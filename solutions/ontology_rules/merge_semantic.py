#!/usr/bin/env python3
"""Combine ontology predictions with cached pretrained semantic candidates.

This executable never reads gold. Selection/calibration runs use separate tools.
"""
import argparse,json,gzip,time
from pathlib import Path
import run as baseline
import enhance


def assign(doc,entities):
    anchors=baseline.patient_anchors(doc);states=enhance.active_by_passage(doc,anchors)
    associations={p['patient_id']:set() for p in doc.get('patient',[])}
    for e in entities:
        pi,p,a,b=enhance.location(doc,e['offset'],e['offset']+e['length'])
        ids=tuple(e['identifier'].split(';'))
        mention=baseline.Mention(a,b,baseline.token_key(e['text']),ids,0)
        pids=baseline.assign_patients(doc,p,mention,anchors,states.get(pi))
        for pid in pids:associations[pid].update([e['text']] if ids==('-1',) else ids)
    return [{'patient_id':p['patient_id'],'phenotype':sorted(associations[p['patient_id']])} for p in doc.get('patient',[])]


def merge(doc,base,candidates,prob=.8,similarity=.87,policy='augment'):
    accepted=[]
    for c in candidates:
        if c['exact'] and policy=='augment':continue
        if c['probability']<prob or c['similarity']<similarity:continue
        _,p,a,b=enhance.location(doc,c['offset'],c['offset']+c['length'])
        if baseline.is_negated(p['text'],a,b):continue
        c=dict(c);c['utility']=.65*c['probability']+.30*c['similarity']+.025*min(4,len(baseline.TOKEN.findall(c['text'])))
        accepted.append(c)
    result=list(base['entities']) if policy=='augment' else []
    for c in sorted(accepted,key=lambda x:(-x['utility'],-x['length'],x['offset'])):
        a,b=c['offset'],c['offset']+c['length']
        if any(a<e['offset']+e['length'] and e['offset']<b for e in result):continue
        result.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
    result.sort(key=lambda e:(e['offset'],e['length'],e['identifier']))
    return {'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':result,'association':assign(doc,result)}


def main():
    ap=argparse.ArgumentParser()
    for k in ['input','base','candidates','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--probability',type=float,default=.8);ap.add_argument('--similarity',type=float,default=.87);ap.add_argument('--policy',choices=['augment','replace'],default='augment')
    args=ap.parse_args();inputs=baseline.read_jsonl(args.input);bases={d['pmc_id']:d for d in baseline.read_jsonl(args.base)}
    assert all('entities' not in x and 'association' not in x for x in inputs)
    with gzip.open(args.candidates,'rt') as f:cs={d['pmc_id']:d['candidates'] for d in map(json.loads,f)}
    assert {d['pmc_id'] for d in inputs}==set(bases)==set(cs)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    preds=[merge(d,bases[d['pmc_id']],cs[d['pmc_id']],args.probability,args.similarity,args.policy) for d in inputs]
    output.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in preds))
    manifest={'config':vars(args),'source_sha256':baseline.sha256(__file__),'enhance_source_sha256':baseline.sha256(enhance.__file__),'baseline_source_sha256':baseline.sha256(baseline.__file__),'inputs':{k:baseline.sha256(getattr(args,k)) for k in ['input','base','candidates']},'output':{'path':str(output),'bytes':output.stat().st_size,'sha256':baseline.sha256(output)}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
