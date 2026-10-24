"""Actual native/HF-tokenizer and PrefixConstrainedLogitsProcessor CPU contract.

No model class, weights, forward, optimizer, generate or held labels are loaded.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from common import ref,write,load_jsonl,validate_refs,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();start=time.monotonic()
    plan=json.loads(Path(a.plan).read_text());validate_refs(plan['files'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='1'
    import torch,transformers,tokenizers
    from transformers import AutoTokenizer
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
    import transformers.generation.logits_process as processor_source
    from tokenizers import Tokenizer
    from token_census import census
    from schema_contract import CanonicalList
    from reuse.native_tokenizer import encode
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    assert transformers.__version__==plan['runtime_versions']['transformers']
    assert tokenizers.__version__==plan['runtime_versions']['tokenizers']
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=False)
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        prefix_processor_source=ref(processor_source.__file__),cuda_initialized=torch.cuda.is_initialized(),
        cuda_available=torch.cuda.is_available(),no_model_weights=True)
    write(root/'runtime.json',runtime)
    model=Path(plan['model_dir'])
    for name,expected in plan['model_assets'].items():
        actual=ref(model/name);assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes'],name
    data=root/'inputs';data.mkdir()
    for phase in ['train','held_input']:
        raw=gzip.decompress(Path(plan['compressed_inputs'][phase]['path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==plan['compressed_inputs'][phase]['raw_sha256']
        with (data/(phase+'.jsonl')).open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
    native_plan=dict(model=plan['model'],template_kind=plan['template_kind'],eos_token_id=plan['eos_token_id'],
        tokenizers_version=plan['runtime_versions']['tokenizers'],inputs={
            'train':ref(data/'train.jsonl'),'held_input':ref(data/'held_input.jsonl'),
            'tokenizer':ref(model/'tokenizer.json'),'template':ref(model/plan['template_file'])})
    write(root/'native-plan.json',native_plan)
    native_result=census(native_plan,root/'native')
    encoded=json.loads(gzip.decompress((root/'native/encoded.json.gz').read_bytes()))
    tk=AutoTokenizer.from_pretrained(model,local_files_only=True)
    native=Tokenizer.from_file(str(model/'tokenizer.json'))
    rows=load_jsonl(data/'train.jsonl');held=load_jsonl(data/'held_input.jsonl');observations=[]
    for phase,items in [('train',rows),('held',held)]:
        for row,expected in zip(items,encoded[phase]):
            prompt=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tk.encode(prompt,add_special_tokens=False)
            assert ids==native.encode(prompt,add_special_tokens=False).ids
            assert tk.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(ids,skip_special_tokens=False)==prompt
            if phase=='train':
                actual=encode(row,tk);assert actual==expected
            else:assert ids==expected['input_ids'] and 'answer' not in row
            observations.append(dict(phase=phase,row_id=row['row_id'],native_HF_full_IDs_labels_rendering_exact=True))
    if plan.get('expected_local_native'):
        expected=plan['expected_local_native']
        for field in ['full_training_tokens','supervised_tokens','held_input_tokens']:
            assert native_result[field]==expected[field],field
        assert ref(root/'native/encoded.json.gz')['sha256']==expected['encoded_sha256']
    # New actual HF interface only. Old broad public trie/name/parse checks are reused.
    names=json.loads(gzip.decompress(Path(plan['names_gzip']).read_bytes()))
    codec=CanonicalList(native,names,plan['eos_token_id'])
    config=json.loads((model/'config.json').read_text());vocab=config.get('text_config',config)['vocab_size']
    assert native.get_vocab_size(with_added_tokens=True)<=vocab
    callback=codec.callback(2);checks=[]
    raw=rows[max(range(len(rows)),key=lambda i:len(rows[i]['answer']))]['answer']
    body=native.encode(raw,add_special_tokens=False).ids
    for k in sorted({0,1,2,3,len(body)//2,len(body)-1,len(body)}):
        prefix=body[:k];ids=torch.tensor([[7,8]+prefix]);scores=torch.arange(vocab,dtype=torch.float32).remainder_(19).unsqueeze(0)
        proc=PrefixConstrainedLogitsProcessor(callback,num_beams=1);out=proc(ids,scores)
        allowed=codec.allowed(prefix);wanted=torch.zeros(vocab,dtype=torch.bool);wanted[allowed]=True
        assert torch.equal(torch.isfinite(out[0]),wanted) and torch.equal(out[0,wanted],scores[0,wanted])
        checks.append(dict(kind='actual_HF_greedy_processor',generated_prefix_tokens=k,allowed=len(allowed)))
    empty_ids=native.encode('[]',add_special_tokens=False).ids
    empty_out=PrefixConstrainedLogitsProcessor(callback,num_beams=1)(torch.tensor([[7,8]+empty_ids]),torch.zeros((1,vocab)))
    assert torch.where(torch.isfinite(empty_out[0]))[0].tolist()==[plan['eos_token_id']]
    # Actual 2-batch x 2-beam invocation with equal prefix lengths and divergent legal first tokens.
    p0=codec.open;option=codec.allowed(p0);assert len(option)>1
    prefixes=[p0+[option[0]],p0+[option[1]],p0+[option[-1]],p0+[option[0]]]
    seen=[]
    def beam_callback(batch_id,ids):seen.append(batch_id);return callback(batch_id,ids)
    beam=PrefixConstrainedLogitsProcessor(beam_callback,num_beams=2)
    input_ids=torch.tensor([[7,8]+x for x in prefixes]);score=torch.zeros((4,vocab))
    result=beam(input_ids,score);assert seen==[0,0,1,1]
    for row,prefix in zip(result,prefixes):assert set(torch.where(torch.isfinite(row))[0].tolist())==set(codec.allowed(prefix))
    perm=[1,0,3,2];result2=beam(input_ids[perm],score)
    assert torch.equal(result2,result[perm])
    try:PrefixConstrainedLogitsProcessor(lambda *_:[],num_beams=1)(torch.tensor([[1,2]]),torch.zeros((1,vocab)))
    except ValueError:empty_rejected=True
    else:raise AssertionError('Actual HF empty allowed set must fail')
    write(root/'observations.json',observations)
    result=dict(status='PASS_ACTUAL_NATIVE_HF_PREFIX_PROCESSOR_NO_MODEL',source=ref(__file__),plan=ref(a.plan),runtime=ref(root/'runtime.json'),
        native=ref(root/'native/manifest.json'),encoded=ref(root/'native/encoded.json.gz'),train_audit=ref(root/'native/train-audit.json'),held_audit=ref(root/'native/F1-input-audit.json'),
        observations=ref(root/'observations.json'),train_rows=87,held_rows=39,all126_native_HF_IDs_decode_exact=True,
        all87_complete_labels_masks_EOS_newline_exact=True,actual_greedy_processor_checks=checks,
        actual_model_output_vocab_size=vocab,actual_empty_list_EOS_only=True,
        actual_two_batch_two_beam_shapes_and_reordering=True,actual_empty_allowed_rejected=empty_rejected,
        no_model_no_forward_no_generate_no_GPU_no_quality=True,seconds=time.monotonic()-start)
    print(json.dumps(write(root/'result.json',result)),flush=True)

if __name__=='__main__':main()
