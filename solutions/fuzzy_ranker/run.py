#!/usr/bin/env python3
"""HPO candidate retrieval + sparse supervised mention and patient rankers.

Uses only the designated ontology and --train annotations. Input labels are never
read. No archived code, resource, or prediction is imported. Thread caps apply
before importing numerical libraries so five independent jobs can coexist.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time

import joblib
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:\.\d+)?")
GAP = re.compile(r'^[\s\-‐‑–—/]*$')
STOP = set('a an and are as at be been but by can case clinical did do for from had has have his her he she in into is it its of on or our patient patients that the their these they this those to was were which with without we not normal'.split())
NEG = re.compile(r'\b(?:no|not|without|denies|denied|negative for|free of)\b',re.I)
FAMILY = re.compile(r'\b(?:mother|father|sister|brother|sibling|parents?|unaffected|healthy)\b',re.I)

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def norm(text):return ' '.join(m.group().lower() for m in TOKEN.finditer(text))

def stem(word):
    if len(word)>5 and word.endswith('ies'):return word[:-3]+'y'
    if len(word)>5 and word.endswith('es') and word[-3] in 'sxz':return word[:-2]
    if len(word)>4 and word.endswith('s') and not word.endswith(('ss','is','us')):return word[:-1]
    return word

def grams(text):
    text=' '+text+' '
    return frozenset(text[i:i+3] for i in range(len(text)-2))

def ids(entity):
    return [s.strip() for s in str(entity.get('identifier','-1')).split(';') if s.strip()]

def parse_hpo(path):
    terms={};current=None;version=None
    for line in Path(path).read_text().splitlines():
        if line.startswith('data-version:'):version=line.partition(':')[2].strip()
        if line=='[Term]':current={'parents':[],'aliases':[],'alt':[]};continue
        if line.startswith('['):current=None;continue
        if current is None:continue
        if line.startswith('id: HP:'):
            current['id']=line[4:];terms[current['id']]=current
        elif line.startswith('name: '):current['name']=line[6:];current['aliases'].append(line[6:])
        elif line.startswith('synonym: '):
            m=re.match(r'synonym: "(.*?)" (EXACT|BROAD|NARROW|RELATED)',line)
            if m and m.group(2) in ('EXACT','BROAD','NARROW'):current['aliases'].append(m.group(1))
        elif line.startswith('is_a: '):current['parents'].append(line.split()[1])
        elif line.startswith('alt_id: '):current['alt'].append(line.split()[1])
        elif line=='is_obsolete: true':current['obsolete']=True
    if version!='hp/releases/2026-06-23':raise ValueError(f'Unexpected HPO version: {version}')
    children=defaultdict(list)
    for hid,t in terms.items():
        for parent in t['parents']:children[parent].append(hid)
    allowed=set();pending=['HP:0000118']
    while pending:
        hid=pending.pop()
        if hid in allowed:continue
        allowed.add(hid);pending.extend(children[hid])
    allowed={x for x in allowed if x in terms and not terms[x].get('obsolete')}
    alt={a:hid for hid in allowed for a in terms[hid]['alt']}
    return {hid:terms[hid] for hid in sorted(allowed)},alt,version

@dataclass
class Candidate:
    pid:int
    start:int
    end:int
    text:str
    normalized:str
    hid:str
    matched:str
    similarity:float
    exact:bool
    probability:float=0.0

class Retriever:
    def __init__(self,hpo,alt,train):
        self.hpo=hpo;self.alt=alt;self.alias_ids=defaultdict(set)
        self.official=set();self.training_docs=defaultdict(set);self.id_freq=Counter()
        for hid,term in hpo.items():
            for text in term['aliases']:
                alias=norm(text)
                if 3<=len(alias)<=120 and 1<=len(alias.split())<=10:
                    self.alias_ids[alias].add(hid);self.official.add((alias,hid))
        for doc in train:
            docid=str(doc['pmc_id'])
            for ent in doc.get('entities',[]):
                if str(ent.get('note','')).upper()=='NO':continue
                alias=norm(ent.get('text',''))
                if len(alias)<2:continue
                for raw in ids(ent):
                    hid=alt.get(raw,raw)
                    if hid not in hpo and hid!='-1':continue
                    self.alias_ids[alias].add(hid);self.training_docs[(alias,hid)].add(docid)
                    self.id_freq[hid]+=1
        self.aliases=sorted(self.alias_ids)
        self.heads=defaultdict(list);self.stem_aliases=defaultdict(list)
        self.alias_tokens=[];self.alias_grams=[]
        df=Counter()
        for i,a in enumerate(self.aliases):
            tokens=a.split();self.alias_tokens.append(tokens)
            self.heads[stem(tokens[-1])].append(i)
            self.stem_aliases[' '.join(map(stem,tokens))].append(i)
            g=grams(a);self.alias_grams.append(g);df.update(g)
        n=len(self.aliases)
        self.idf={g:math.log((n+1)/(freq+1))+1 for g,freq in df.items()}
        self.alias_norms=[math.sqrt(sum(self.idf[g]**2 for g in gs)) for gs in self.alias_grams]
        self.cache={};self.query_count=0
    def similarity(self,q,idx):
        qg=grams(q);ag=self.alias_grams[idx]
        top=sum(self.idf.get(g,1.)**2 for g in qg&ag)
        denominator=math.sqrt(sum(self.idf.get(g,1.)**2 for g in qg))*self.alias_norms[idx]
        return top/max(denominator,1e-9)
    def retrieve(self,query):
        if query in self.cache:return self.cache[query]
        self.query_count+=1
        toks=query.split();matches=[]
        if query in self.alias_ids:
            matches.extend((hid,query,1.,True) for hid in sorted(self.alias_ids[query]))
        elif len(query)>=5:
            stemquery=' '.join(map(stem,toks))
            indices=self.stem_aliases.get(stemquery,[])
            for idx in indices:
                a=self.aliases[idx]
                matches.extend((hid,a,self.similarity(query,idx),False) for hid in sorted(self.alias_ids[a]))
        if not matches and len(query)>=7 and len(toks)<=7:
            candidates=self.heads.get(stem(toks[-1]),[])
            if len(candidates)>250:
                first=stem(toks[0]);candidates=[i for i in candidates if first in map(stem,self.alias_tokens[i])]
            fuzzy=[]
            qset=set(map(stem,toks))
            for idx in candidates[:500]:
                a=self.aliases[idx];at=self.alias_tokens[idx]
                if abs(len(at)-len(toks))>1:continue
                if len(toks)==1 and a!=query and ratio(a,query)<92:continue
                overlap=len(qset&set(map(stem,at)))/max(len(qset|set(map(stem,at))),1)
                if overlap<.5:continue
                char=ratio(query,a)/100
                order=token_sort_ratio(query,a)/100
                if max(char,order)<.82:continue
                cosine=self.similarity(query,idx)
                if cosine<.68:continue
                score=.55*cosine+.25*char+.2*order
                fuzzy.append((score,idx))
            for score,idx in sorted(fuzzy,reverse=True)[:3]:
                a=self.aliases[idx]
                matches.extend((hid,a,score,False) for hid in sorted(self.alias_ids[a]))
        # Different aliases can name the same concept; keep its strongest retrieval.
        best={}
        for match in matches:
            if match[0] not in best or match[2]>best[match[0]][2]:best[match[0]]=match
        result=list(best.values())[:6];self.cache[query]=result
        return result
    def candidates(self,doc):
        result=[]
        for pid,passage in enumerate(doc.get('full_text',[])):
            text=passage.get('text','');offset=int(passage.get('offset',0));tokens=list(TOKEN.finditer(text))
            for end in range(1,len(tokens)+1):
                head=stem(tokens[end-1].group().lower())
                if head not in self.heads:continue
                for size in range(1,min(8,end)+1):
                    start=end-size;window=tokens[start:end]
                    if window[0].group().lower() in STOP and size>1:continue
                    if any(not GAP.match(text[window[k].end():window[k+1].start()]) for k in range(size-1)):continue
                    begin=window[0].start();finish=window[-1].end()
                    surface=text[begin:finish];query=' '.join(m.group().lower() for m in window)
                    if len(query)<3 and not surface.isupper():continue
                    for hid,matched,similarity,exact in self.retrieve(query):
                        result.append(Candidate(pid,offset+begin,offset+finish,surface,query,hid,matched,similarity,exact))
        return result

def nearby(doc,c):
    passage=doc['full_text'][c.pid];text=passage['text'];local=c.start-int(passage['offset']);end=c.end-int(passage['offset'])
    left=text[max(0,local-80):local];right=text[end:end+60]
    clause=re.split(r'[.;!?\n]',left)[-1]
    neg=bool(NEG.search(clause[-60:])) and 'not only' not in clause.lower()
    return passage,left,right,neg

def mention_features(doc,c,r,training=False):
    passage,left,right,neg=nearby(doc,c)
    support=r.training_docs.get((c.normalized,c.hid),set())
    if training:support=support-{str(doc['pmc_id'])}
    matched_support=r.training_docs.get((c.matched,c.hid),set())
    if training:matched_support=matched_support-{str(doc['pmc_id'])}
    words=c.normalized.split();mt=c.matched.split()
    feats={
      'bias':1.,'similarity':c.similarity,'exact':float(c.exact),
      'official_exact':float((c.normalized,c.hid) in r.official),
      'official_candidate':float((c.matched,c.hid) in r.official),
      'training_support':math.log1p(len(support)),'matched_training_support':math.log1p(len(matched_support)),
      'length':min(len(c.text),80)/40,'words':len(words)/4,
      'word_difference':abs(len(words)-len(mt)),'uppercase':float(c.text.isupper()),
      'titlecase':float(c.text.istitle()),'short':float(len(c.text)<=4),
      'negated':float(neg),'family_context':float(bool(FAMILY.search(left[-60:]))),
      'section='+str(passage.get('section_type','')):1.,'type='+str(passage.get('type','')):1.,
      'concept_frequency':math.log1p(r.id_freq[c.hid]),
      'identifier_unknown':float(c.hid=='-1'),
    }
    for i,t in enumerate(norm(left).split()[-3:][::-1]):feats[f'left{i}={t}']=1.
    for i,t in enumerate(norm(right).split()[:2]):feats[f'right{i}={t}']=1.
    # Surface identity helps suppress common nonclinical polysemy. Capped per-feature
    # regularization and explicit train-support cross-document counts limit memorization.
    feats['surface='+c.normalized]=1.
    return feats

def gold_mentions(doc,alt):
    result=set()
    for e in doc.get('entities',[]):
        if str(e.get('note','')).upper()=='NO':continue
        for hid in ids(e):result.add((int(e['offset']),int(e['offset'])+int(e['length']),alt.get(hid,hid)))
    return result

def fit_classifier(features,labels):
    if len(set(labels))<2:raise ValueError('Training candidates must contain positive and negative samples')
    model=make_pipeline(DictVectorizer(dtype=np.float64),LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023))
    model.fit(features,np.array(labels));return model

def score_candidates(doc,cs,r,model,training=False):
    if not cs:return
    values=model.predict_proba([mention_features(doc,c,r,training) for c in cs])[:,1]
    for c,value in zip(cs,values):c.probability=float(value)

def select_mentions(doc,cs,threshold):
    eligible=[c for c in cs if c.probability>=threshold and not nearby(doc,c)[3]]
    # One normalized concept per span; overlapping alternatives are resolved by
    # calibrated classifier evidence rather than unconditionally taking longest.
    ordered=sorted(eligible,key=lambda c:(-(c.probability+.012*min(len(c.normalized.split()),6)),-(c.end-c.start),c.start,c.hid))
    result=[];occupied=defaultdict(list)
    for c in ordered:
        if any(c.start<other.end and other.start<c.end for other in occupied[c.pid]):continue
        result.append(c);occupied[c.pid].append(c)
    return sorted(result,key=lambda c:(c.start,c.end,c.hid))

def patient_anchors(doc):
    anchors=defaultdict(list);owners=defaultdict(set)
    for patient in doc.get('patient',[]):
        key=str(patient['patient_id'])
        for m in patient.get('mention',[]):
            a=int(m['offset']);z=a+int(m['length']);surface=m.get('text','')
            anchors[key].append((a,z))
            if len(surface)>=4:owners[surface.lower()].add(key)
    for surface,keys in owners.items():
        if len(keys)!=1:continue
        key=next(iter(keys))
        expression=re.compile(r'(?<!\w)'+re.escape(surface)+r'(?!\w)',re.I)
        for passage in doc.get('full_text',[]):
            off=int(passage['offset'])
            anchors[key].extend((off+m.start(),off+m.end()) for m in expression.finditer(passage['text']))
    return {key:sorted(set(values)) for key,values in anchors.items()}

def relation_features(doc,c,patient,anchors):
    key=str(patient['patient_id']);passage,left,right,neg=nearby(doc,c)
    begin=int(passage['offset']);end=begin+len(passage['text']);center=(c.start+c.end)/2
    own=anchors.get(key,[]);all_anchors=[(k,a,z) for k,v in anchors.items() for a,z in v]
    distance=min((abs(center-(a+z)/2) for a,z in own),default=50000)
    own_inside=[(a,z) for a,z in own if begin<=a<end]
    other_inside=[(k,a,z) for k,a,z in all_anchors if k!=key and begin<=a<end]
    before=[(k,a,z) for k,a,z in all_anchors if a<=c.start]
    nearest_before=max(before,key=lambda x:x[1])[0] if before else None
    nearest=min(all_anchors,key=lambda x:abs(center-(x[1]+x[2])/2))[0] if all_anchors else None
    local=c.start-begin;sentence_left=max(passage['text'].rfind(mark,0,local) for mark in '.!?\n')+begin+1
    sentence_end_candidates=[passage['text'].find(mark,local) for mark in '.!?\n']
    sentence_right=begin+min([v for v in sentence_end_candidates if v>=0] or [len(passage['text'])])
    same_sentence=any(sentence_left<=a<=sentence_right for a,z in own)
    others_sentence=sum(sentence_left<=a<=sentence_right for k,a,z in all_anchors if k!=key)
    feats={'bias':1.,'single_patient':float(len(doc.get('patient',[]))==1),'patient_count':min(len(doc.get('patient',[])),10)/5,
      'entity_probability':c.probability,'negated':float(neg),'log_distance':math.log1p(distance)/10,
      'near_100':float(distance<100),'near_300':float(distance<300),'same_passage':float(bool(own_inside)),
      'same_sentence':float(same_sentence),'other_sentence':float(others_sentence>0),'only_patient_in_passage':float(bool(own_inside) and not other_inside),
      'nearest_anchor':float(nearest==key),'nearest_preceding':float(nearest_before==key),
      'has_any_anchor':float(bool(own)),'family_context':float(bool(FAMILY.search(left[-60:]))),
      'section='+str(passage.get('section_type','')):1.,'type='+str(passage.get('type','')):1.}
    for t in norm(left).split()[-5:]:feats['context='+t]=1.
    for t in norm(right).split()[:3]:feats['context='+t]=1.
    return feats

def gold_associations(doc,alt):
    result={str(p['patient_id']):set() for p in doc.get('patient',[])}
    for row in doc.get('association',[]):
        result.setdefault(str(row['patient_id']),set()).update(alt.get(x,x) for x in row.get('phenotype',[]))
    return result

def predict_doc(doc,retriever,mention_model,relation_model,mention_threshold,relation_threshold,stats):
    cs=retriever.candidates(doc);stats['candidates']+=len(cs)
    score_candidates(doc,cs,retriever,mention_model)
    chosen=select_mentions(doc,cs,mention_threshold);stats['entities']+=len(chosen)
    anchors=patient_anchors(doc);features=[];pairs=[]
    for c in chosen:
        for patient in doc.get('patient',[]):
            features.append(relation_features(doc,c,patient,anchors));pairs.append((str(patient['patient_id']),c.hid if c.hid!='-1' else c.text))
    scores=relation_model.predict_proba(features)[:,1] if features else []
    association={str(p['patient_id']):set() for p in doc.get('patient',[])}
    for (key,concept),score in zip(pairs,scores):
        if score>=relation_threshold:association[key].add(concept)
    stats['associations']+=sum(map(len,association.values()))
    return {'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':[{'identifier':c.hid,'type':'Phenotype','offset':c.start,'length':c.end-c.start,'text':c.text,'note':None} for c in chosen],
      'association':[{'patient_id':p['patient_id'],'phenotype':sorted(association[str(p['patient_id'])])} for p in doc.get('patient',[])]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('train','input','hpo','output','work-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--mention-threshold',type=float,default=.60)
    parser.add_argument('--relation-threshold',type=float,default=.55)
    parser.add_argument('--reuse-model',action='store_true')
    args=parser.parse_args();start=time.monotonic();work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True)
    train=load_jsonl(args.train);inputs=load_jsonl(args.input)
    assert len({str(d['pmc_id']) for d in train})==len(train),'Duplicate training document'
    train_hash=digest(args.train);hpo_hash=digest(args.hpo)
    cache=work/'model.joblib';fingerprint={'train_sha256':train_hash,'hpo_sha256':hpo_hash,'implementation_sha256':digest(__file__)}
    if args.reuse_model and cache.exists():
        saved=joblib.load(cache)
        if saved['fingerprint']!=fingerprint:raise ValueError('Model cache does not match train, ontology, or implementation')
        retriever,mention_model,relation_model=saved['model'];training_stats=saved['training_stats']
    else:
        hpo,alt,version=parse_hpo(args.hpo);retriever=Retriever(hpo,alt,train)
        print(json.dumps({'stage':'retrieval_ready','hpo_terms':len(hpo),'aliases':len(retriever.aliases)}),flush=True)
        rows=[];labels=[];doc_candidates=[]
        for doc in train:
            cs=retriever.candidates(doc);gold=gold_mentions(doc,alt);doc_candidates.append(cs)
            for c in cs:
                rows.append(mention_features(doc,c,retriever,training=True));labels.append(int((c.start,c.end,c.hid) in gold))
        mention_model=fit_classifier(rows,labels)
        training_stats={'mention_candidates':len(labels),'mention_positives':sum(labels),'training_documents':len(train),'hpo_terms':len(hpo),'aliases':len(retriever.aliases)}
        print(json.dumps({'stage':'mention_model_trained',**training_stats}),flush=True)
        del rows,labels
        rows=[];labels=[]
        for doc,cs in zip(train,doc_candidates):
            score_candidates(doc,cs,retriever,mention_model,training=True)
            chosen=select_mentions(doc,cs,.4);anchors=patient_anchors(doc);gold=gold_associations(doc,alt)
            for c in chosen:
                concept=c.hid if c.hid!='-1' else c.text
                for patient in doc.get('patient',[]):
                    rows.append(relation_features(doc,c,patient,anchors));labels.append(int(concept in gold.get(str(patient['patient_id']),set())))
        relation_model=fit_classifier(rows,labels)
        training_stats.update({'relation_candidates':len(labels),'relation_positives':sum(labels)})
        retriever.cache={}
        joblib.dump({'fingerprint':fingerprint,'model':(retriever,mention_model,relation_model),'training_stats':training_stats},cache,compress=3)
        print(json.dumps({'stage':'relation_model_trained','rows':len(labels),'positive':sum(labels)}),flush=True)
        del rows,labels,doc_candidates
    prediction_stats=Counter();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for index,doc in enumerate(inputs):
            pred=predict_doc(doc,retriever,mention_model,relation_model,args.mention_threshold,args.relation_threshold,prediction_stats)
            f.write(json.dumps(pred,ensure_ascii=False)+'\n')
            if (index+1)%10==0:print(json.dumps({'stage':'predict','completed':index+1,'total':len(inputs)}),flush=True)
    metadata={'method':'HPO fuzzy candidate retrieval + sparse mention classifier + supervised patient attribution','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
      'training':training_stats,'prediction':dict(prediction_stats),'input_documents':len(inputs),'input_sha256':digest(args.input),'output_sha256':digest(out),'output_bytes':out.stat().st_size,
      'fingerprint':fingerprint,'command':sys.argv,'mention_threshold':args.mention_threshold,'relation_threshold':args.relation_threshold,
      'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'evaluation_note':'Input labels are never used. Train and validation interpretation is determined by supplied files. Model cache requires exact data, HPO and implementation hashes.'}
    write_json(work/'run.json',metadata);print(json.dumps({'stage':'complete',**prediction_stats,'seconds':round(metadata['elapsed_seconds'],2)}),flush=True)
if __name__=='__main__':main()
