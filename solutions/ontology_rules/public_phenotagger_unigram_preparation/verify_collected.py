"""Check recovered input-only coordinates/token arrays; no gold or model imports."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--input', required=True)
    a = ap.parse_args()
    t = time.monotonic()
    root = Path(a.root)
    out = root/'terminal/output'
    summary = json.loads((out/'summary.json').read_text())
    assert sha(a.input) == summary['input_sha256']
    docs = {str(d['pmc_id']): d for d in map(json.loads,Path(a.input).read_text().splitlines())}
    assert len(docs) == 64 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs.values())
    for name, ref in summary['artifacts'].items():
        assert sha(out/name)==ref['sha256'] and (out/name).stat().st_size==ref['bytes']
    words, spans, dictionary = Counter(), set(), 0
    for r in map(json.loads,gzip.open(out/'input-only-unigram-coordinates.jsonl.gz','rt')):
        p=docs[r['pmc_id']]['full_text'][r['passage_index']]
        assert p['text'][r['local_start']:r['local_end']]==r['text']
        assert p['offset']+r['local_start']==r['offset']
        assert r['local_end']-r['local_start']==r['length']
        assert r['lower_tokens']==[r['native_token'].lower()]
        k=(r['pmc_id'],r['offset'],r['length']);assert k not in spans;spans.add(k)
        words[r['lower_tokens'][0]]+=1
        dictionary+=r['raw_dictionary_span_present']
    expected=Counter()
    model_inputs={m:set() for m in summary['tokenizers']}
    truncs={m:[] for m in model_inputs}
    digests={m:hashlib.sha256() for m in model_inputs}
    last=None
    for r in map(json.loads,gzip.open(out/'input-only-unique-tokens.jsonl.gz','rt')):
        word,=r['lower_tokens'];assert last is None or last<word;last=word
        expected[word]=r['count']
        for model,x in r['models'].items():
            assert len(x['input_ids'])==len(x['token_type_ids'])==len(x['attention_mask'])<=32
            assert set(x['attention_mask'])=={1}
            assert all(0<=v<summary['tokenizers'][model]['vocab_size'] for v in x['input_ids'])
            model_inputs[model].add(tuple(x['input_ids']))
            if x['truncated']:truncs[model].append({'lower_tokens':[word],'coordinate_count':r['count'],'full_tokens':x['full_tokens'],'native_cases':r['native_cases']})
            digests[model].update((json.dumps([word,x],sort_keys=True)+'\n').encode())
    assert words==expected and len(words)==summary['unique_lower_unigrams']
    assert len(spans)==summary['counts']['accepted_coordinates']
    assert dictionary==summary['dictionary']['accepted_spans_present']
    for model in digests:assert digests[model].hexdigest()==summary['tokenizers'][model]['input_record_sha256']
    report=dict(status='PASS_owner_recovered_contract_check',source_sha256=sha(__file__),summary_sha256=sha(out/'summary.json'),
        input_sha256=sha(a.input),coordinates=len(spans),unique_lower_tokens=len(words),raw_dictionary_coordinate_presence=dictionary,
        unique_inputID_sequences={m:len(v) for m,v in model_inputs.items()},truncation_details=truncs,
        no_gold=True,no_NN=True,seconds=time.monotonic()-t,
        scope='All recovered coordinates/original slices and all token-array digests checked. Input-ID collisions do not reduce the planned exact-native-tuple forward count; no old multiword input-ID equivalence search.')
    dest=root/'recovered-contract-check.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
