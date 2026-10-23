#!/usr/bin/env python3
"""Fixed two-arm entity assembly; no gold access, no patient reassignment."""
from pathlib import Path
import argparse, copy, datetime, gzip, hashlib, json, math

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);raw=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return [json.loads(line) for line in raw.splitlines()]

def decode(candidates):
    # Original calibrate.py ordering/tie stability; preserve candidate input order.
    selected=[]
    for c in sorted(candidates,key=lambda c:(-c['slot_probability'],-c['length'],c['offset'])):
        if c['slot_probability']<.5:continue
        if any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in selected):continue
        selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
    return sorted(selected,key=lambda e:e['offset'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory
    plan=json.loads((b/'plan.json').read_text());cfg=plan['assembly']
    assert sha(__file__)==cfg['source_sha256'] and cfg['cut']==.5
    for item in cfg['references'].values():assert sha(item['path'])==item['sha256']
    templates=read(cfg['references']['baseline']['path']);prompts=read(cfg['references']['heldout_prompts']['path'])
    assert len(templates)==16 and len(prompts)==398
    inputs={str(d['pmc_id']):d for d in read(cfg['references']['input']['path'])}
    assert all(not {'entities','association'}&d.keys() for d in inputs.values())
    completion=json.loads((b/'terminal-infer/inference/complete.json').read_text());assert completion['no_labels'] and completion['no_quality']
    all_keys=[];grouped={str(d['pmc_id']):[] for d in templates}
    for row in prompts:
        for c in row['candidates']:
            key=(str(row['pmc_id']),c['offset'],c['length'],c['identifier']);all_keys.append(key)
            document=inputs[key[0]]
            assert any(c['offset']>=p['offset'] and c['offset']+c['length']<=p['offset']+len(p['text'])
                and p['text'][c['offset']-p['offset']:c['offset']-p['offset']+c['length']]==c['text'] for p in document['full_text'])
            grouped[key[0]].append(c)
    assert len(all_keys)==1589 and len(set(all_keys))==1589
    outputs={};output=b/'assembled';output.mkdir(exist_ok=False)
    for arm in ['base','final']:
        path=b/'terminal-infer/inference'/f'{arm}.probabilities.jsonl';assert sha(path)==completion['arms'][arm]['probability_sha256']
        scores=read(path);assert len(scores)==len(prompts);lookup={}
        for row,prob in zip(prompts,scores):
            assert row['row_id']==prob['row_id'] and str(row['pmc_id'])==str(prob['pmc_id'])
            expected=[[c['offset'],c['length'],c['identifier']] for c in row['candidates']]
            assert expected==prob['keys'] and len(expected)==len(prob['probabilities'])
            for key,value in zip(expected,prob['probabilities']):
                assert math.isfinite(value) and 0<=value<=1;lookup[(str(row['pmc_id']),*key)]=value
        assert set(lookup)==set(all_keys)
        result=[]
        for template in templates:
            pmc=str(template['pmc_id']);candidates=[dict(c,slot_probability=lookup[(pmc,c['offset'],c['length'],c['identifier'])]) for c in grouped[pmc]]
            d=copy.deepcopy(template);d['entities']=decode(candidates)
            assert {k:v for k,v in d.items() if k!='entities'}=={k:v for k,v in template.items() if k!='entities'}
            result.append(d)
        path=output/(arm+'.pred.jsonl');path.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in result))
        outputs[arm]=dict(path=str(path),sha256=sha(path),entities=sum(len(d['entities']) for d in result))
    freeze=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outputs=outputs,
        inference_complete_sha256=sha(b/'terminal-infer/inference/complete.json'),source_sha256=sha(__file__),
        baseline_sha256=cfg['references']['baseline']['sha256'],plan_sha256=sha(b/'plan.json'),
        gold_read=False,all_nonentities_preserved=True,threshold=.5,no_extra_negation_or_NIL_filter=True,
        policy='old calibrate greedy (-p,-length,offset), stable input-order tie; no overlap; candidate IDs unchanged')
    (output/'freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n');print(json.dumps(freeze))

if __name__=='__main__':main()
