#!/usr/bin/env python3
"""Fold-local patient relation LR with explicit rule, negation and cohort evidence."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import run as base
import combine
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

COHORT=re.compile(r'\b(both|all(?:\s+(?:two|three|four|five|six|seven|eight|nine|ten|\d+))?|these|our)\s+(?:affected\s+)?(?:patients|siblings|brothers|sisters|children|individuals|cases)\b',re.I)
NUMBERS={'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}

def cohort_evidence(text,patient_count,section):
    clinical=section.upper() in {'CASE','RESULTS','CASE_REPORT','CASE PRESENTATION'}
    for m in COHORT.finditer(text):
        word=m.group(1).lower();parts=word.split();n=2 if word=='both' else NUMBERS.get(parts[-1],int(parts[-1]) if parts[-1].isdigit() else None)
        if n is not None and n!=patient_count:continue
        if word=='our' or clinical:return True
    return False

def features(doc,entity,patient,rule_pids,anchors,counts):
    f=base.pair_features(doc,entity,patient);pi,p=base.passage_for(doc,entity['offset']);off=int(p.get('offset',0));text=p.get('text','');a=entity['offset']-off;b=a+entity['length']
    left,right=combine.ontology.clause_bounds(text,a,b);clause=text[left:right]
    keys=base.association_keys(entity);frequency=max(counts[k] for k in keys)
    negative=combine.ontology.is_negated(text,a,b)
    own=[x for x in anchors if x[2]==patient['patient_id']];before=[x for x in anchors if x[0]<=entity['offset']]
    preceding=before[-1][2] if before else None
    healthy=0
    for m in patient.get('mention',[]):
        _,passage=base.passage_for(doc,int(m['offset']));pos=int(m['offset'])-int(passage.get('offset',0));window=passage.get('text','')[max(0,pos-45):pos+int(m.get('length',0))+65]
        healthy+=bool(re.search(r'\b(?:unaffected|asymptomatic|clinically normal|healthy)\b',window,re.I))
    f.update({'rule_selected':patient['patient_id'] in rule_pids,'rule_patients':len(rule_pids),
        'negated':negative,'negated_near':negative and f['nearest'],'cohort_explicit':cohort_evidence(clause,len(doc.get('patient',[])),str(p.get('section_type',''))),
        'cohort_passage':cohort_evidence(text,len(doc.get('patient',[])),str(p.get('section_type',''))),
        'nearest_preceding_patient':preceding==patient['patient_id'],'expanded_anchors':math.log1p(len(own)),
        'healthy_mention_fraction':healthy/max(len(patient.get('mention',[])),1),'concept_mention_logcount':math.log1p(frequency),
        'same_sentence_negative':negative and f['same_sentence']})
    return f

def rows_for(doc,entities,linker,training=False,include_negative=False):
    gold=base.gold_associations(doc,linker) if training else None
    counts=defaultdict(int);prepared=[]
    for e in entities:
        if str(e.get('note') or '').upper()=='NO' and not include_negative:continue
        hid=linker.canonical(e.get('identifier','-1'))
        if not hid:continue
        e=dict(e,identifier=hid);prepared.append(e)
        for key in base.association_keys(e):counts[key]+=1
    anchors=combine.ontology.patient_anchors(doc);rows=[];pairs=[];ys=[]
    for e in prepared:
        rule=combine.assign(doc,[e]);rule_pids={a['patient_id'] for a in rule if a['phenotype']}
        for patient in doc.get('patient',[]):
            rows.append(features(doc,e,patient,rule_pids,anchors,counts));pairs.append((e,patient['patient_id']))
            if training:ys.append(str(e.get('note') or '').upper()!='NO' and any(k in gold.get(patient['patient_id'],set()) for k in base.association_keys(e)))
    return rows,pairs,ys

def main():
    p=argparse.ArgumentParser()
    for k in ['train','input','hpo','predictions','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--include-negative-examples',action='store_true');p.add_argument('--threshold',type=float,default=.55)
    a=p.parse_args();train=base.read_jsonl(a.train);docs=base.read_jsonl(a.input);preds={str(d['pmc_id']):d for d in base.read_jsonl(a.predictions)}
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in docs}
    assert all('entities' not in d and 'association' not in d for d in docs)
    linker=base.Linker(train,a.hpo);rows=[];ys=[]
    for d in train:
        xx,_,yy=rows_for(d,d.get('entities',[]),linker,True,a.include_negative_examples);rows.extend(xx);ys.extend(yy)
    vec=DictVectorizer();clf=LogisticRegression(C=1.,solver='liblinear',class_weight='balanced',random_state=20261023,max_iter=400);clf.fit(vec.fit_transform(rows),ys)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as h:
        for d in docs:
            pred=preds[str(d['pmc_id'])];xx,pairs,_=rows_for(d,pred['entities'],linker);scores=clf.predict_proba(vec.transform(xx))[:,1] if xx else []
            vals={p['patient_id']:set() for p in d.get('patient',[])}
            for (e,pid),score in zip(pairs,scores):
                if score>=a.threshold:vals[pid].update(base.association_keys(e))
            pred=dict(pred,association=[{'patient_id':p['patient_id'],'phenotype':sorted(vals[p['patient_id']])} for p in d.get('patient',[])])
            h.write(json.dumps(pred,ensure_ascii=False)+'\n')
    coefs=dict(zip(vec.feature_names_,clf.coef_[0].tolist()))
    manifest={'method':'rule, cohort, negation evidence logistic patient attribution','config':vars(a),'command':sys.argv,'source_sha256':base.sha(__file__),
      'common_source_sha256':base.sha(base.__file__),'combine_source_sha256':base.sha(combine.__file__),'ontology_source_sha256':base.sha(combine.ontology.__file__),
      'inputs':{k:base.sha(getattr(a,k)) for k in ['train','input','hpo','predictions']},'output_sha256':base.sha(out),
      'training_examples':len(ys),'training_positive':sum(ys),'added_feature_coefficients':{k:v for k,v in coefs.items() if k in ['rule_selected','negated','cohort_explicit','cohort_passage','healthy_mention_fraction','nearest_preceding_patient']},'input_annotations_used':False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
