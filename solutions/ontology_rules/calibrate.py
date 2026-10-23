#!/usr/bin/env python3
"""Second-stage span calibration fitted on paper-disjoint OOF candidate records.

The supplied training manifest must enumerate disjoint validation papers whose
union is --train. Each candidate cache records a generator trained without that
paper. The original dev is never included in calibration fitting.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,json,gzip,math,time,platform
import sklearn
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
import run as baseline
import enhance
import merge_semantic


def read_candidates(path):
    with gzip.open(path,'rt') as f:return {d['pmc_id']:d['candidates'] for d in map(json.loads,f)}


def pool(doc,base,raw):
    cs={(c['offset'],c['length'],c['identifier']):dict(c,is_base=False) for c in raw}
    for e in base['entities']:
        k=(e['offset'],e['length'],e['identifier'])
        if k in cs:cs[k]['is_base']=True
        else:cs[k]=dict(e,probability=.9,similarity=1.,exact=True,is_base=True)
    return list(cs.values())


def features(doc,c,base):
    pi,p,s,e=enhance.location(doc,c['offset'],c['offset']+c['length']);text=p['text'];left,right=baseline.clause_bounds(text,s,e);phrase=text[s:e];words=baseline.TOKEN.findall(phrase)
    before=baseline.TOKEN.findall(text[max(left,s-60):s].lower());after=baseline.TOKEN.findall(text[e:min(right,e+60)].lower())
    overlap=[b for b in base['entities'] if c['offset']<b['offset']+b['length'] and b['offset']<c['offset']+c['length']]
    key=baseline.token_key(phrase);support=Counter(e['identifier'] for e in base['entities'])
    f={'prob':c['probability'],'similarity':c['similarity'],'exact':c['exact'],'base':c['is_base'],'length':min(c['length'],150),'word_count':len(words),'uppercase':phrase.isupper(),'titlecase':phrase.istitle(),'unlinked':c['identifier']=='-1','has_digit':any(ch.isdigit() for ch in phrase),'has_dash':'-' in phrase,'has_slash':'/' in phrase,'negated':baseline.is_negated(text,s,e),'clinical':bool(baseline.CLINICAL.search(text[left:right])),'family':bool(baseline.FAMILY.search(text[left:right])),'general':bool(baseline.GENERAL.search(text[left:right])),'section':str(p.get('section_type','')),'ptype':str(p.get('type','')),'early':pi<3,'n_overlap':min(4,len(overlap)),'covers_base':any(c['offset']<=b['offset'] and b['offset']+b['length']<=c['offset']+c['length'] for b in overlap),'contained_base':any(b['offset']<=c['offset'] and c['offset']+c['length']<=b['offset']+b['length'] for b in overlap),'same_base_id':any(c['identifier']==b['identifier'] for b in overlap),'log_doc_frequency':math.log1p(support[c['identifier']]),'has_modifier':any(w.lower() in enhance.MODIFIERS for w in words),'starts_connector':bool(words and words[0].lower() in enhance.CONNECTORS),'ends_connector':bool(words and words[-1].lower() in enhance.CONNECTORS)}
    # Small controlled context alphabet avoids document IDs or surface memorization.
    for token in ['with','without','of','and','or','no','normal','severe','mild','a','the','had','have','was','were','showed']:
        f['left_'+token]=bool(before and before[-1]==token);f['right_'+token]=bool(after and after[0]==token)
    return f


def main():
    ap=argparse.ArgumentParser()
    for k in ['training-manifest','train','input','base','candidates','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--threshold',type=float,default=.4)
    a=ap.parse_args();started=time.monotonic();source_sha=baseline.sha256(__file__);training=baseline.read_jsonl(a.train);inputs=baseline.read_jsonl(a.input);trainids={d['pmc_id'] for d in training};inputids={d['pmc_id'] for d in inputs};assert not trainids&inputids;assert all('entities' not in d and 'association' not in d for d in inputs)
    specs=json.loads(Path(a.training_manifest).read_text());seen=set();x=[];y=[];training_by_id={d['pmc_id']:d for d in training}
    for spec in specs:
        docs=baseline.read_jsonl(spec['gold']);base={d['pmc_id']:d for d in baseline.read_jsonl(spec['base'])};raw=read_candidates(spec['candidates']);generator_manifest=json.loads(Path(spec['candidate_manifest']).read_text())
        assert generator_manifest['inputs']['input']==baseline.sha256(spec['input'])
        assert generator_manifest['inputs']['train']==baseline.sha256(spec['generator_train'])
        generated_train_ids={d['pmc_id'] for d in baseline.read_jsonl(spec['generator_train'])}
        dids={d['pmc_id'] for d in docs};assert not dids&generated_train_ids;assert not seen&dids;seen|=dids
        for d in docs:
            assert d==training_by_id[d['pmc_id']], 'OOF labels differ from original train64'
            truth={(e['offset'],e['length']):set(e['identifier'].split(';')) for e in d['entities'] if str(e.get('note','')).upper()!='NO'}
            for c in pool(d,base[d['pmc_id']],raw[d['pmc_id']]):
                ids=truth.get((c['offset'],c['length']),set());target=bool(ids and (set(c['identifier'].split(';'))<=ids or '-1' in ids))
                x.append(features(d,c,base[d['pmc_id']]));y.append(target)
    assert seen==trainids
    vec=DictVectorizer(sparse=False);matrix=vec.fit_transform(x)
    model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=5.,learning_rate=.06,random_state=20261023,early_stopping=False).fit(matrix,y)
    raw=read_candidates(a.candidates);base={d['pmc_id']:d for d in baseline.read_jsonl(a.base)};pred=[]
    for d in inputs:
        cs=pool(d,base[d['pmc_id']],raw[d['pmc_id']]);probs=model.predict_proba(vec.transform([features(d,c,base[d['pmc_id']]) for c in cs]))[:,1]
        selected=[]
        for c,p in sorted(zip(cs,probs),key=lambda cp:(-cp[1],-cp[0]['length'],cp[0]['offset'])):
            if p<a.threshold:continue
            start,end=c['offset'],c['offset']+c['length']
            if any(start<e['offset']+e['length'] and e['offset']<end for e in selected):continue
            selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
        selected.sort(key=lambda e:e['offset']);pred.append({'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':selected,'association':merge_semantic.assign(d,selected)})
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in pred))
    meta={'config':vars(a),'source_sha256':source_sha,'dependency_sha256':{m.__name__:baseline.sha256(m.__file__) for m in [baseline,enhance,merge_semantic]},'runtime':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},'input_hashes':{k:baseline.sha256(getattr(a,k)) for k in ['training_manifest','train','input','base','candidates']},'training_examples':len(y),'training_positive':sum(y),'training_documents':len(trainids),'candidate_origins':'4 fold OOF, each document excluded from its generating model','feature_count':matrix.shape[1],'estimator':model.get_params(),'actual_iterations':model.n_iter_,'early_stopping_performed':model.do_early_stopping_,'threshold_selection':'fixed a priori .4; no dev threshold search','output_sha256':baseline.sha256(output),'seconds':time.monotonic()-started}
    output.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
