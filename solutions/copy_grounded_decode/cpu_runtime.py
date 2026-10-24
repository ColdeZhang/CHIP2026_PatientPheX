"""Actual target HF/tokenizer CPU contract. No model/forward/generate/quality."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from raw_store import publish_json


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    start = time.monotonic()
    p = json.loads(Path(a.plan).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and os.environ['OMP_NUM_THREADS'] == '1'
    for item in p['assets'].values():
        assert ref(item['path']) == item
    assert p['assets']['runner'] == ref(__file__)
    import torch, transformers, peft, tokenizers
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    import transformers.generation.logits_process as lp
    import transformers.generation.utils as gu
    from processors import ordered_processors, FiniteLogitsOnly
    from triple_trie import TripleList
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__, tokenizers=tokenizers.__version__)
    assert versions == p['runtime_versions']
    out = Path(p['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    root = Path(p['storage_parent'])
    model = Path(p['model'])
    for name, wanted in p['tokenizer_assets'].items():
        actual = ref(model/name)
        assert actual['bytes'] == wanted['bytes'] and actual['sha256'] == wanted['sha256']
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    native = Tokenizer.from_file(str(model/'tokenizer.json'))
    prompts = [json.loads(x) for x in Path(p['assets']['prompts']['path']).read_text().splitlines()]
    audits = {r['row_id']:r for r in json.loads(Path(p['assets']['held_audit']['path']).read_text())}
    encoded = json.loads(gzip.decompress(Path(p['assets']['held_ids']['path']).read_bytes()))
    ids = {r['row_id']:r['input_ids'] for r in encoded}
    pools = {r['row_id']:r for r in map(json.loads, Path(p['assets']['candidates']['path']).read_text().splitlines())}
    assert len(prompts) == len(audits) == len(ids) == len(pools) == 39
    assert {r['row_id'] for r in prompts} == set(audits) == set(ids) == set(pools)
    observations = []
    for row in prompts:
        assert set(row) == {'row_id','pmc_id','target_patient_id','messages','units'}
        rid = row['row_id']
        text = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        actual = tok.encode(text, add_special_tokens=False)
        assert actual == native.encode(text, add_special_tokens=False).ids == ids[rid]
        assert tok.decode(actual, skip_special_tokens=False, clean_up_tokenization_spaces=False) == native.decode(actual, skip_special_tokens=False) == text
        assert sha_text(text) == audits[rid]['rendered_prefix_sha256']
        assert hashlib.sha256(json.dumps(actual).encode()).hexdigest() == audits[rid]['input_ids_sha256']
        assert len(actual) == audits[rid]['input_tokens']
        assert (pools[rid]['pmc_id'], pools[rid]['target_patient_id']) == (row['pmc_id'], row['target_patient_id'])
        observations.append(dict(row_id=rid, input_tokens=len(actual), full_native_HF_original_IDs_template_decode_exact=True))
    assert sum(map(len,ids.values())) == 203588 and max(map(len,ids.values())) == 10080
    vocab = json.loads((model/'config.json').read_text())['text_config']['vocab_size']
    assert native.get_vocab_size(with_added_tokens=True) <= vocab
    checks = []
    for row in prompts:
        codec = TripleList(native, pools[row['row_id']]['triples'], [248044,248046])
        width = len(ids[row['row_id']])
        processors = ordered_processors(codec, width)
        assert [type(x).__name__ for x in processors] == ['FiniteLogitsOnly','PrefixConstrainedLogitsProcessor']
        chosen = sorted(codec.triples)[:2]
        suffix = codec.serialize_ids(chosen)
        for k in sorted({0,1,2,len(suffix)//2,len(suffix)-1,len(suffix)}):
            prefix = suffix[:k]
            inp = torch.tensor([ids[row['row_id']] + prefix])
            scores = torch.arange(vocab, dtype=torch.float32).remainder_(19).unsqueeze(0)
            original = scores.clone()
            result = processors(inp, scores)
            allowed = codec.allowed(prefix)
            mask = torch.zeros(vocab, dtype=torch.bool)
            mask[allowed] = True
            assert torch.equal(torch.isfinite(result[0]),mask)
            assert torch.equal(result[0,mask],scores[0,mask]) and torch.equal(scores,original)
            assert torch.equal(processors(inp,scores),result)
            assert torch.equal(result,ordered_processors(codec,width)(inp,scores))
            checks.append(dict(row_id=row['row_id'],prefix_tokens=k,allowed=len(allowed)))
        # Wrong order reproduces the integration failure without any model.
        inp = torch.tensor([ids[row['row_id']]])
        scores = torch.zeros((1,vocab))
        try:
            FiniteLogitsOnly()(inp,processors[1](inp,scores))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Mask before finite guard must fail')
    codec = TripleList(native, sorted(codec.triples)[:2], [248044,248046])
    proc = ordered_processors(codec,2)
    inp = torch.tensor([[7,8]])
    for value in [float('nan'),float('inf'),float('-inf')]:
        scores = torch.zeros((1,vocab));scores[0,0] = value
        try:
            proc(inp,scores)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Original nonfinite logits not rejected')
    for suffix in [codec.serialize_ids([]),codec.serialize_ids([sorted(codec.triples)[0]])]:
        scores = proc(torch.tensor([[7,8]+suffix]),torch.zeros((1,vocab)))
        assert torch.where(torch.isfinite(scores[0]))[0].tolist() == [248044,248046]
    empty = TripleList(native,[],[248044,248046])
    ep = ordered_processors(empty,2)
    scores = ep(inp,torch.zeros((1,vocab)))
    assert torch.where(torch.isfinite(scores[0]))[0].tolist() == empty.empty
    for suffix in [[999999],codec.encode('[["U9999","not-in-pool",0]]')]:
        try:
            proc(torch.tensor([[7,8]+suffix]),torch.zeros((1,vocab)))
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal prefix accepted')
    duplicate = codec.encode(json.dumps([sorted(codec.triples)[0]]*2,separators=(',',':')))
    try:
        proc(torch.tensor([[7,8]+duplicate]),torch.zeros((1,vocab)))
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate triple accepted')
    assert not torch.cuda.is_initialized()
    runtime = publish_json(out/'runtime.json',dict(versions=versions,logits_processor_source=ref(lp.__file__),generation_source=ref(gu.__file__),CUDA_initialized=False,CUDA_available=False),root)
    obs = publish_json(out/'observations.json',observations,root)
    result = dict(status='PASS_ACTUAL_HF_CPU_ORDERED_TRIE39',source=ref(__file__),plan=ref(a.plan),runtime=runtime,observations=obs,
                  rows=39,documents=len({r['pmc_id'] for r in prompts}),input_tokens=203588,max_input_tokens=10080,
                  actual_processor_checks=checks,nonfinite_rejection_cases=3,reversed_order_rejects=True,
                  legal_mask_exact_and_scores_preserved=True,repeat_callbacks_exact=True,empty_pool_and_EOS_and_bad_prefix_and_duplicate_verified=True,
                  all39_native_HF_original_IDs_template_decode_exact=True,no_weights_no_forward_no_generate_no_GPU_no_quality=True,
                  seconds=time.monotonic()-start)
    print(json.dumps(publish_json(out/'result.json',result,root)),flush=True)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=='__main__':
    main()
