#!/usr/bin/env python3
"""Input-only Bioformer80: unchanged prepared64 plus a pinned native dev16 cache."""
import argparse
import json
import math
from pathlib import Path
import run as base

WEIGHT='867819ac932e11250c42ca13c49eba3711a7f24b170e24bfb31b65589d0024a1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','cache-dir','output-dir']:p.add_argument('--'+name,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());cache=Path(a.cache_dir);out=Path(a.output_dir)
    for name in ['prepared64','prepared64_manifest','input64','dev_input','native_dev_plan']:
        assert base.sha(plan[name]['path'])==plan[name]['sha256']
    old=Path(plan['prepared64']['path']);old_manifest=json.loads(Path(plan['prepared64_manifest']['path']).read_text())
    assert old_manifest['output_sha256']==base.sha(old) and old_manifest['public_model_task_fit'] is False
    native_plan=json.loads(Path(plan['native_dev_plan']['path']).read_text())
    manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['plan_sha256']==plan['native_dev_plan']['sha256']
    assert manifest['model_weight_sha256']==old_manifest['model_weight_sha256']==WEIGHT
    assert manifest['public_model_training_on_task'] is False
    assert manifest['input_sha256']==plan['dev_input']['sha256']
    assert manifest['fixed_policy']==native_plan['policy']
    assert manifest['source_sha256']==native_plan['remote_source_sha256']['dev16-cache/make_provenance.py']
    for name,item in manifest['outputs'].items():
        assert base.sha(cache/name)==item['sha256'] and (cache/name).stat().st_size==item['bytes']
    train=base.read_jsonl(plan['input64']['path']);dev=base.read_jsonl(plan['dev_input']['path'])
    trainids={d['pmc_id'] for d in train};devmap={d['pmc_id']:d for d in dev}
    assert len(trainids)==64 and len(devmap)==16 and not trainids&devmap.keys()
    assert list(devmap)==manifest['input_ids']
    assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in train+dev)
    assert {d['pmc_id'] for d in base.read_jsonl(old)}==trainids
    _,allowed,_,_=base.parse_hpo(plan['hpo']);allowed.discard('HP:0000118')
    key=lambda r:(r['pmc_id'],r['offset'],r['length'],r['hpo_id'])
    provenance={}
    for r in base.read_jsonl(cache/'source-provenance.jsonl'):
        k=key(r);assert k not in provenance and r['pmc_id'] in devmap and r['hpo_id'] in allowed
        flags=[r[n] for n in ['raw_neural_match','raw_dictionary_match','abbrev_inferred']]
        assert all(isinstance(v,bool) for v in flags) and any(flags)
        assert all(isinstance(v,int) and 0<=v<17266 for v in r['matched_neural_head_columns'])
        assert all(math.isfinite(v) and 0<=v<=1 for v in r['actual_hybrid_scores'])
        provenance[k]=r
    rows={did:[] for did in devmap};seen=set()
    for r in base.read_jsonl(cache/'canonical-entities.jsonl'):
        k=key(r);evidence=provenance[k]
        assert r['raw_hpo_id'] in evidence['raw_hpo_ids'] and r['score'] in evidence['actual_hybrid_scores']
        passage=devmap[r['pmc_id']]['full_text'][r['passage_index']];lo,hi=r['local_start'],r['local_end']
        assert 0<=lo<hi<=len(passage['text']) and r['offset']==passage['offset']+lo and r['length']==hi-lo
        assert r['text']==passage['text'][lo:hi]
        rows[r['pmc_id']].append(dict(offset=r['offset'],length=r['length'],text=r['text'],identifier=r['hpo_id'],
            bioformer_present=True,bioformer_score=float(r['score']),bioformer_neural=evidence['raw_neural_match'],
            bioformer_dictionary=evidence['raw_dictionary_match'],bioformer_abbreviation=evidence['abbrev_inferred'],
            bioformer_origin_unknown=False,bioformer_raw_hpo_ids=evidence['raw_hpo_ids'],
            bioformer_matched_neural_head_columns=evidence['matched_neural_head_columns']))
        seen.add(k)
    assert seen==set(provenance)
    assert sum(map(len,rows.values()))==manifest['counts']['canonical_kept']
    out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists()
    oldbytes=old.read_bytes();assert oldbytes.endswith(b'\n')
    devbytes=''.join(json.dumps(dict(pmc_id=did,candidates=cs),ensure_ascii=False)+'\n' for did,cs in rows.items()).encode()
    (out/'candidates.jsonl').write_bytes(oldbytes+devbytes)
    (out/'input80.jsonl').write_bytes(Path(plan['input64']['path']).read_bytes()+Path(plan['dev_input']['path']).read_bytes())
    assert (out/'candidates.jsonl').read_bytes()[:len(oldbytes)]==oldbytes
    result=dict(source_sha256=base.sha(__file__),dependency_sha256=base.sha(base.__file__),
        plan_sha256=base.sha(a.plan),public_model_task_fit=False,model_weight_sha256=WEIGHT,
        input_path=str(out/'input80.jsonl'),input_sha256=base.sha(out/'input80.jsonl'),
        output_sha256=base.sha(out/'candidates.jsonl'),documents=80,train64_rows_byteexact=True,
        train64_prepared_manifest_sha256=base.sha(plan['prepared64_manifest']['path']),
        native_dev_manifest_path=str(cache/'manifest.json'),native_dev_manifest_sha256=base.sha(cache/'manifest.json'),
        dev_rows=sum(map(len,rows.values())),dev_unique_keys=len(seen),annotations_read=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
