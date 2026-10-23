#!/usr/bin/env python3
"""Bind a frozen independent Bioformer64 cache to source-HGB input, without labels."""
import argparse
import json
import math
from pathlib import Path
import run as base

WEIGHT = '867819ac932e11250c42ca13c49eba3711a7f24b170e24bfb31b65589d0024a1'
PLAN = '73f7fadc504b08949e8a345ca564fa6c704d626fbda6a6a2adf580dfd215c485'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache-dir', 'cache-plan', 'input', 'hpo', 'output-dir']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    cache, out = Path(a.cache_dir), Path(a.output_dir)
    manifest = json.loads((cache/'manifest.json').read_text())
    plan = json.loads(Path(a.cache_plan).read_text())
    assert base.sha(a.cache_plan) == PLAN == manifest['plan_sha256']
    assert manifest['public_model_training_on_task'] is False
    assert manifest['model_weight_sha256'] == plan['model_weight_sha256'] == WEIGHT
    assert manifest['source_sha256'] == plan['remote_source_sha256']['train64-cache/make_provenance.py']
    assert manifest['fixed_policy'] == plan['policy']
    assert base.sha(a.input) == manifest['input_sha256'] == plan['input64_sha256']
    assert base.sha(a.hpo) == 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
    assert manifest['counts']['documents'] == 64 and manifest['counts']['reused_documents'] == 0
    for name, evidence in manifest['outputs'].items():
        assert base.sha(cache/name) == evidence['sha256']
        assert (cache/name).stat().st_size == evidence['bytes']
    docs = base.read_jsonl(a.input)
    docmap = {d['pmc_id']: d for d in docs}
    assert len(docmap) == 64 and list(docmap) == manifest['input_ids']
    assert all(set(d) == {'pmc_id','pmid','patient','full_text'} for d in docs)
    _, allowed, _, _ = base.parse_hpo(a.hpo)
    allowed.discard('HP:0000118')
    key = lambda r: (r['pmc_id'],r['offset'],r['length'],r['hpo_id'])
    provenance = {}
    for r in base.read_jsonl(cache/'source-provenance.jsonl'):
        k = key(r)
        assert k not in provenance and r['hpo_id'] in allowed
        flags = [r[n] for n in ['raw_neural_match','raw_dictionary_match','abbrev_inferred']]
        assert all(isinstance(v,bool) for v in flags) and any(flags)
        assert all(isinstance(c,int) and 0 <= c < 17266 for c in r['matched_neural_head_columns'])
        assert all(math.isfinite(s) and 0 <= s <= 1 for s in r['actual_hybrid_scores'])
        provenance[k] = r
    rows = {did:[] for did in docmap}
    seen = set()
    for r in base.read_jsonl(cache/'canonical-entities.jsonl'):
        k = key(r)
        assert k in provenance
        evidence = provenance[k]
        assert r['raw_hpo_id'] in evidence['raw_hpo_ids'] and r['score'] in evidence['actual_hybrid_scores']
        passage = docmap[r['pmc_id']]['full_text'][r['passage_index']]
        lo, hi = r['local_start'],r['local_end']
        assert r['offset'] == passage['offset']+lo and r['length'] == hi-lo
        assert r['text'] == passage['text'][lo:hi] and r['length'] > 0
        rows[r['pmc_id']].append(dict(offset=r['offset'],length=r['length'],text=r['text'],
            identifier=r['hpo_id'],bioformer_present=True,bioformer_score=float(r['score']),
            bioformer_neural=evidence['raw_neural_match'],bioformer_dictionary=evidence['raw_dictionary_match'],
            bioformer_abbreviation=evidence['abbrev_inferred'],bioformer_origin_unknown=False,
            bioformer_raw_hpo_ids=evidence['raw_hpo_ids'],
            bioformer_matched_neural_head_columns=evidence['matched_neural_head_columns']))
        seen.add(k)
    assert seen == set(provenance)
    assert len(seen) == manifest['counts']['provenance_distinct_keys']
    assert sum(map(len,rows.values())) == manifest['counts']['canonical_kept']
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists()
    (out/'candidates.jsonl').write_text(''.join(json.dumps(dict(pmc_id=did,candidates=cs),ensure_ascii=False)+'\n' for did,cs in rows.items()))
    result = dict(source_sha256=base.sha(__file__),dependency_sha256=base.sha(base.__file__),
        public_model_task_fit=False,model_weight_sha256=WEIGHT,
        input_path=a.input,input_sha256=base.sha(a.input),hpo_sha256=base.sha(a.hpo),
        native_manifest_path=str(cache/'manifest.json'),native_manifest_sha256=base.sha(cache/'manifest.json'),
        native_plan_path=a.cache_plan,native_plan_sha256=PLAN,
        provenance_sha256=base.sha(cache/'source-provenance.jsonl'),
        canonical_sha256=base.sha(cache/'canonical-entities.jsonl'),
        output_sha256=base.sha(out/'candidates.jsonl'),documents=len(rows),
        rows=sum(map(len,rows.values())),unique_keys=len(seen),
        annotations_read=False,scope='Input-only schema/source join; no inference/fitting/evaluation.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
