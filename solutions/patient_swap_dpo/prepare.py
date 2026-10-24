"""Train-only, deterministic patient-swap preference supply and token census.

No model, remote machine, held labels, fitting, or evaluator. Keep full records
outside Git. Requires the already cached tokenizers CPU environment.
"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import time
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=sha(data))


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def make_pair(row):
    chosen = json.loads(row['answer'])
    patients = row['patient_ids']
    assert len(set(patients)) == len(patients)
    assert set(chosen) == set(row['code_map'])
    assert all(isinstance(v, list) and len(set(v)) == len(v) and set(v) <= set(patients)
               for v in chosen.values())
    swaps = [(cid, positive, negative) for cid, values in chosen.items()
             for positive in values for negative in patients if negative not in values]
    if not swaps:
        return None, 0
    def key(swap):
        return sha(json.dumps([20261024, row['row_id'], *swap],
                              ensure_ascii=False, separators=(',', ':')).encode())
    cid, positive, negative = min(swaps, key=key)
    rejected = copy.deepcopy(chosen)
    index = rejected[cid].index(positive)
    rejected[cid][index] = negative
    # Preserve native patient ordering; no change to any other CID.
    rejected[cid].sort(key=patients.index)
    old_pairs = {(c, p) for c, values in chosen.items() for p in values}
    new_pairs = {(c, p) for c, values in rejected.items() for p in values}
    assert old_pairs - new_pairs == {(cid, positive)}
    assert new_pairs - old_pairs == {(cid, negative)}
    assert len(old_pairs) == len(new_pairs)
    assert all(chosen[c] == rejected[c] for c in chosen if c != cid)
    answer = json.dumps(rejected, ensure_ascii=False, separators=(',', ':'))
    return dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                messages=row['messages'], chosen=row['answer'], rejected=answer,
                changed_cid=cid, removed_patient=positive, added_patient=negative,
                original_positive_patients=len(chosen[cid]),
                eligible_swaps=len(swaps)), len(swaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--encoded', type=Path, required=True)
    parser.add_argument('--native-census', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    body = args.train.read_bytes()
    assert sha(body) == 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    assert ref(args.tokenizer)['sha256'] == 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
    rows = [json.loads(x) for x in body.splitlines()]
    old = [json.loads(x) for x in gzip.decompress(args.encoded.read_bytes()).splitlines()]
    old_by_id = {x['row_id']: x for x in old}
    native = json.loads(args.native_census.read_text())
    assert native['all204_native_raw_ids_and_roundtrip_exact']
    assert native['train_source_sha256'] == sha(body)
    assert len(rows) == len(old) == len(old_by_id) == 124
    assert {str(x['pmc_id']) for x in rows} == set(native['training_ids'])
    assert not set(native['training_ids']) & set(native['held_ids'])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    pairs, encoded_pairs, audits, skipped = [], [], [], []
    all_swaps = 0
    for row in rows:
        cached = old_by_id[row['row_id']]
        prompt_ids = cached['input_ids'][:cached['prompt_length']]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        full = prompt + row['answer'] + '<|im_end|>\n'
        ids = tokenizer.encode(full, add_special_tokens=False).ids
        assert ids == cached['input_ids']
        assert cached['labels'] == [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        assert tokenizer.decode(ids, skip_special_tokens=False) == full
        pair, n = make_pair(row)
        all_swaps += n
        if pair is None:
            skipped.append(dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                                patients=len(row['patient_ids']), reason='no_positive_to_true_negative_patient_swap'))
            continue
        rejected_text = prompt + pair['rejected'] + '<|im_end|>\n'
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False).ids
        assert rejected_ids[:len(prompt_ids)] == prompt_ids
        assert ids[:len(prompt_ids)] == prompt_ids
        assert tokenizer.decode(rejected_ids, skip_special_tokens=False) == rejected_text
        assert ids[-2:] == rejected_ids[-2:] == [151645, 198]
        # The whole answer including EOS/newline is scored. Full-string tokenization
        # is mandatory; no concatenation of independently encoded answer tokens.
        pairs.append(pair)
        encoded_pairs.append(dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                                  chosen_input_ids=ids, rejected_input_ids=rejected_ids,
                                  prompt_length=len(prompt_ids)))
        audits.append(dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                           prompt_tokens=len(prompt_ids), chosen_tokens=len(ids),
                           rejected_tokens=len(rejected_ids),
                           chosen_supervised=len(ids)-len(prompt_ids),
                           rejected_supervised=len(rejected_ids)-len(prompt_ids),
                           same_cardinality=True, same_prefix=True, eos_exact=True,
                           chosen_ids_sha256=sha(json.dumps(ids,separators=(',',':')).encode()),
                           rejected_ids_sha256=sha(json.dumps(rejected_ids,separators=(',',':')).encode())))
    for name, content in [('pairs.jsonl.gz', pairs), ('encoded-pairs.jsonl.gz', encoded_pairs)]:
        b=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in content).encode()
        (args.output/name).write_bytes(gzip.compress(b,mtime=0))
    dump(args.output/'row-audit.json', audits)
    dump(args.output/'skipped.json', skipped)
    summary=dict(status='PASS_TRAIN_ONLY_SWAP_AND_FULL_NATIVE_TOKEN_CENSUS',
                 at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 seconds=time.monotonic()-started, source=ref(Path(__file__)),
                 inputs={k:ref(getattr(args,k)) for k in ['train','encoded','native_census','tokenizer']},
                 tokenizers_version=tokenizers_version, source_rows=124, source_documents=32,
                 pair_rows=len(pairs), pair_documents=len({x['pmc_id'] for x in pairs}),
                 all_eligible_swaps=all_swaps, selected_one_per_eligible_row=True,
                 selected_multipos_rows=sum(x['original_positive_patients']>1 for x in pairs),
                 skipped_rows=len(skipped), skipped_single_patient=sum(x['patients']==1 for x in skipped),
                 source_train32_excludes_F1=True,
                 total_tokens=sum(x['chosen_tokens']+x['rejected_tokens'] for x in audits),
                 chosen_tokens=sum(x['chosen_tokens'] for x in audits),
                 rejected_tokens=sum(x['rejected_tokens'] for x in audits),
                 chosen_supervised=sum(x['chosen_supervised'] for x in audits),
                 rejected_supervised=sum(x['rejected_supervised'] for x in audits),
                 max_tokens=max(max(x['chosen_tokens'],x['rejected_tokens']) for x in audits),
                 token_length_delta_histogram=dict(collections.Counter(x['rejected_tokens']-x['chosen_tokens'] for x in audits)),
                 original124_chosen_ids_and_masks_exact=True,
                 pair_prefill_and_EOS_exact=True, native_HF_new_reject_check_not_run=True,
                 artifacts={k:ref(args.output/k) for k in ['pairs.jsonl.gz','encoded-pairs.jsonl.gz','row-audit.json','skipped.json']},
                 model_calls=0,GPU_calls=0,held_gold_reads=0,fit_calls=0,quality_calls=0)
    dump(args.output/'result.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','artifacts','source')}))


if __name__ == '__main__':
    main()
