#!/usr/bin/env python3
"""Ontology-derived lexical variants, local abbreviation resolution and evidence aggregation.

No input annotations are used. Candidate learning excludes the current training
paper's alias support. HPO synonyms are the only external semantic resource.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(key,'1')
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import run as baseline

CONNECTORS={'of','the','in','and','with','to','a','an'}
MORPH={'hypoplastic':'hypoplasia','dysplastic':'dysplasia','hypertrophic':'hypertrophy','atrophic':'atrophy','aplastic':'aplasia','stenotic':'stenosis','fibrotic':'fibrosis','anomalous':'anomaly','abnormal':'abnormality','abnormalities':'abnormality','defective':'defect','deficient':'deficiency','reduced':'decreased','elevated':'increased'}
MODIFIERS={'bilateral','unilateral','severe','mild','moderate','recurrent','progressive','congenital','generalized','multiple','early','late'}
PURE_HPO=re.compile(r'HP:\d{7}(?:;HP:\d{7})*\Z')


def key(text,mode='basic'):
    parts=baseline.token_key(text)
    if mode=='basic':return parts
    parts=tuple(MORPH.get(t,t) for t in parts if t not in CONNECTORS)
    return tuple(sorted(parts))

@dataclass
class Candidate:
    start:int
    end:int
    ids:tuple
    key:tuple
    source:str
    support:int
    ontology:bool
    negated:bool
    probability:float=1.0


class OntologyLexicon:
    def __init__(self,hpo,train,mode='basic',synonyms='exact',abbreviations=True):
        self.mode=mode;self.abbreviations=abbreviations
        self.terms,self.allowed,self.alt,self.version=baseline.load_hpo(hpo)
        self.options=defaultdict(dict)
        self.by_doc=defaultdict(Counter)
        def add(surface,ids,official=False,doc=None):
            k=key(surface,mode)
            if not k or len(k)>16 or len(''.join(k))<2:return
            if ids not in self.options[k]:self.options[k][ids]={'official':False,'documents':Counter()}
            entry=self.options[k][ids]
            entry['official'] |= official
            if doc is not None:entry['documents'][doc]+=1
        if synonyms!='exact':
            current=None
            for line in Path(hpo).read_text().splitlines():
                if line.startswith('id: '):current=line[4:]
                elif line.startswith('synonym: ') and current in self.allowed:
                    m=re.match(r'synonym: "((?:[^"\\]|\\.)*)"\s+(\w+)',line)
                    if m and (synonyms=='all' or m[2] in {'EXACT','RELATED'}):add(m[1].replace('\\"','"'),(current,),True)
        for hid in sorted(self.allowed):
            t=self.terms[hid]
            for s in [t.get('name','')]+t['synonyms']:add(s,(hid,),True)
        for d in train:
            did=str(d['pmc_id'])
            for e in d.get('entities',[]):
                if str(e.get('note','')).upper()=='NO':continue
                ids=self.canonical(e['identifier'])
                if ids:add(e['text'],ids,doc=did)
        self.lasts=defaultdict(set)
        self.trie={}
        for k in self.options:
            if mode=='basic':
                n=self.trie
                for t in k:n=n.setdefault(t,{})
                n[None]=k
            else:
                for t in set(k):self.lasts[t].add(len(k))

    def canonical(self,value):
        if str(value)=='-1':return ('-1',)
        return tuple(sorted({self.alt.get(x,x) for x in str(value).split(';') if self.alt.get(x,x) in self.allowed}))

    def choose(self,k,exclude=None):
        values=[]
        for ids,v in self.options.get(k,{}).items():
            support=sum(n for doc,n in v['documents'].items() if doc!=exclude)
            if support or v['official']:values.append((support,len(v['documents'])-(exclude in v['documents']),v['official'],ids))
        if not values:return None
        count,docs,official,ids=min(values,key=lambda x:(-x[0],-x[2],len(x[3]),x[3]))
        return ids,count,official

    def local_abbreviations(self,text,exclude=None):
        # Schwartz/Hearst-style right-to-left character alignment. The expansion
        # must itself resolve against an allowed, fold-local lexicon entry.
        local={};other=set()
        for m in re.finditer(r'\(([A-Za-z][A-Za-z0-9-]{1,9})\)',text):
            short=m[1];letters=re.sub(r'[^A-Za-z0-9]','',short)
            if not short.isupper() or len(letters)<2:continue
            prior=text[max(0,m.start()-180):m.start()].rstrip()
            toks=list(baseline.TOKEN.finditer(prior))
            if not toks:continue
            j=len(prior)-1;ok=True
            for i in range(len(letters)-1,-1,-1):
                while j>=0 and (prior[j].lower()!=letters[i].lower() or (i==0 and j>0 and prior[j-1].isalnum())):j-=1
                if j<0:ok=False;break
                j-=1
            if not ok:continue
            long=prior[j+1:]
            if len(baseline.TOKEN.findall(long))>min(len(letters)+5,len(letters)*2):continue
            resolved=self.choose(key(long,self.mode),exclude)
            if resolved:local[short]=resolved
            else:other.add(short)
        return local,other

    def candidates(self,doc,exclude=None):
        alltext='\n'.join(p.get('text','') for p in doc.get('full_text',[]))
        abbrev,other=self.local_abbreviations(alltext,exclude) if self.abbreviations else ({},set())
        out=[]
        for passage in doc.get('full_text',[]):
            if str(passage.get('section_type','')).upper() in baseline.STOP_SECTIONS:continue
            text=passage.get('text','');base=int(passage.get('offset',0));toks=list(baseline.TOKEN.finditer(text))
            for i,tok in enumerate(toks):
                if tok.group() in abbrev:
                    ids,count,official=abbrev[tok.group()]
                    out.append(Candidate(base+tok.start(),base+tok.end(),ids,key(tok.group(),self.mode),'abbreviation',count,official,baseline.is_negated(text,tok.start(),tok.end())))
                    continue
                if tok.group() in other:continue
                node=self.trie
                for j in range(i,min(i+16,len(toks))):
                    if j>i and not re.fullmatch(r'[\s\-/‐‑–—]*',text[toks[j-1].end():toks[j].start()]):break
                    if self.mode=='basic':
                        node=node.get(baseline.normalize(toks[j].group()))
                        if node is None:break
                        if None not in node:continue
                        k=node[None]
                    else:
                        surface=text[tok.start():toks[j].end()]
                        if toks[j].group().lower() in CONNECTORS:continue
                        k=key(surface,self.mode)
                        if not k or len(k)>10:continue
                        if k not in self.options:continue
                    chosen=self.choose(k,exclude)
                    if chosen is None:continue
                    ids,count,official=chosen
                    start,end=tok.start(),toks[j].end();surface=text[start:end]
                    if len(k)==1 and len(''.join(k))<=3 and (not surface.isupper() or len(''.join(k))<2):continue
                    source='basic' if self.mode=='basic' else 'morph'
                    out.append(Candidate(base+start,base+end,ids,k,source,count,official,baseline.is_negated(text,start,end)))
        return list({(c.start,c.end,c.ids):c for c in out}.values())


def location(doc,start,end):
    for i,p in enumerate(doc['full_text']):
        base=int(p.get('offset',0))
        if base<=start and end<=base+len(p.get('text','')):return i,p,start-base,end-base
    raise ValueError('Invalid span')


def entity_features(doc,c):
    pi,p,s,e=location(doc,c.start,c.end);text=p['text'];surface=text[s:e]
    left,right=baseline.clause_bounds(text,s,e);before=text[max(left,s-80):s];after=text[e:min(right,e+80)]
    feats={'bias':1.,'official':c.ontology,'support':math.log1p(c.support),'has_support':c.support>0,'negated':c.negated,'length':e-s,'tokens':len(c.key),'source':c.source,'section':str(p.get('section_type','')),'ptype':str(p.get('type','')),'allcaps':surface.isupper(),'single':len(doc.get('patient',[]))==1,'early':pi<3,'family':bool(baseline.FAMILY.search(text[left:right])),'clinical':bool(baseline.CLINICAL.search(text[left:right])),'general':bool(baseline.GENERAL.search(text[left:right])),'id_unlinked':c.ids==('-1',)}
    for tag,tokens in [('left',baseline.TOKEN.findall(before)[-3:]),('right',baseline.TOKEN.findall(after)[:3])]:
        for n,t in enumerate(tokens):feats[f'{tag}{n}']=t.lower()
    # Token identities generalize ordinary lexical noise; no document/patient ID.
    for t in c.key:feats['token='+t]=1.
    return feats


def relation_features(doc,entity,patient,rule=False):
    off=int(entity['offset']);pi,p,s,e=location(doc,off,off+int(entity['length']));txt=p['text'];base=int(p['offset']);left,right=baseline.clause_bounds(txt,s,e)
    positions=[int(m['offset']) for m in patient.get('mention',[]) if int(m.get('offset',-1))>=0]
    others=[int(m['offset']) for q in doc.get('patient',[]) if q['patient_id']!=patient['patient_id'] for m in q.get('mention',[]) if int(m.get('offset',-1))>=0]
    near=min([abs(t-off) for t in positions] or [100000]);other=min([abs(t-off) for t in others] or [100000]);prev=min([off-t for t in positions if t<=off] or [100000]);otherprev=min([off-t for t in others if t<=off] or [100000])
    sentence=txt[left:right]
    return {'bias':1.,'patients':len(doc.get('patient',[])),'single':len(doc.get('patient',[]))==1,'log_dist':math.log1p(near),'log_prev':math.log1p(prev),'nearest':near<=other,'previous':prev<=otherprev,'margin':max(-7.,min(7.,math.log1p(other)-math.log1p(near))),'same_sentence':any(base+left<=t<base+right for t in positions),'same_passage':any(base<=t<base+len(txt) for t in positions),'other_sentence':any(base+left<=t<base+right for t in others),'other_passage':any(base<=t<base+len(txt) for t in others),'family':bool(baseline.FAMILY.search(sentence)),'clinical':bool(baseline.CLINICAL.search(sentence)),'general':bool(baseline.GENERAL.search(sentence)),'section':str(p.get('section_type','')),'ptype':str(p.get('type','')),'early':pi<3,'rule':rule}


def relation_gold(doc):
    return {a['patient_id']:{x for v in a['phenotype'] for x in (v.split(';') if PURE_HPO.fullmatch(v) else [v])} for a in doc.get('association',[])}


def surface(doc,c):
    _,p,s,e=location(doc,c.start,c.end);return p['text'][s:e]


def to_entity(doc,c):return {'identifier':';'.join(c.ids),'type':'Phenotype','offset':c.start,'length':c.end-c.start,'text':surface(doc,c),'note':None}


def nonoverlap(candidates,threshold=.5,model=None,doc=None):
    if model and candidates:
        vec,clf=model
        probs=clf.predict_proba(vec.transform([entity_features(doc,c) for c in candidates]))[:,1]
        for c,p in zip(candidates,probs):c.probability=float(p)
    valid=[c for c in candidates if (c.probability>=threshold if model else not c.negated)]
    # Prefer posterior confidence, then widest evidence; resolve after scoring.
    selected=[];occupied=[]
    for c in sorted(valid,key=lambda c:(-c.probability,-(c.end-c.start),-c.support,c.start,c.ids)):
        if any(c.start<b and c.end>a for a,b in occupied):continue
        selected.append(c);occupied.append((c.start,c.end))
    return sorted(selected,key=lambda c:c.start)


def fit_entity(train,lexicon):
    x=[];y=[]
    for d in train:
        truth={(int(e['offset']),int(e['offset'])+int(e['length'])):set(str(e['identifier']).split(';')) for e in d.get('entities',[]) if str(e.get('note','')).upper()!='NO'}
        for c in lexicon.candidates(d,exclude=str(d['pmc_id'])):
            truthids=truth.get((c.start,c.end),set());label=bool(truthids and (set(c.ids)<=truthids or '-1' in truthids))
            x.append(entity_features(d,c));y.append(label)
    vec=DictVectorizer();mat=vec.fit_transform(x)
    clf=LogisticRegression(C=.3,class_weight=None,solver='liblinear',max_iter=500,random_state=20261023)
    clf.fit(mat,y)
    return (vec,clf),{'examples':len(y),'positive':sum(y)}


def fit_relation(train,lexicon):
    x=[];y=[]
    for d in train:
        gold=relation_gold(d);anchors=baseline.patient_anchors(d)
        for e in d.get('entities',[]):
            if str(e.get('note','')).upper()=='NO':continue
            ids=lexicon.canonical(e['identifier'])
            if not ids:continue
            _,p,s,end=location(d,e['offset'],e['offset']+e['length'])
            m=baseline.Mention(s,end,(),ids,0)
            rules=baseline.assign_patients(d,p,m,anchors,None)
            vals={e['text']} if ids==('-1',) else set(ids)
            for patient in d.get('patient',[]):
                x.append(relation_features(d,e,patient,patient['patient_id'] in rules));y.append(bool(vals&gold.get(patient['patient_id'],set())))
    vec=DictVectorizer();mat=vec.fit_transform(x)
    clf=LogisticRegression(C=.5,class_weight='balanced',solver='liblinear',max_iter=500,random_state=20261023).fit(mat,y)
    return (vec,clf),{'examples':len(y),'positive':sum(y)}


def predict(doc,lexicon,entity_model,relation_model,entity_threshold,relation_threshold,relation_mode):
    candidates=nonoverlap(lexicon.candidates(doc),entity_threshold,entity_model,doc)
    entities=[to_entity(doc,c) for c in candidates]
    associations={p['patient_id']:set() for p in doc.get('patient',[])}
    anchors=baseline.patient_anchors(doc)
    prev=None;active=None;rows=[];pairs=[]
    for c,e in zip(candidates,entities):
        pi,p,s,end=location(doc,c.start,c.end)
        if pi!=prev:
            preceding=[a for a in anchors if a[0]<int(p.get('offset',0))]
            active=preceding[-1][2] if preceding else None
            prev=pi
        rules=baseline.assign_patients(doc,p,baseline.Mention(s,end,c.key,c.ids,c.support),anchors,active)
        vals={e['text']} if c.ids==('-1',) else set(c.ids)
        if relation_mode=='rules':
            for pid in rules:associations[pid].update(vals)
        else:
            for patient in doc.get('patient',[]):
                rows.append(relation_features(doc,e,patient,patient['patient_id'] in rules));pairs.append((patient['patient_id'],vals))
    if rows:
        vec,clf=relation_model;probs=clf.predict_proba(vec.transform(rows))[:,1]
        for prob,(pid,vals) in zip(probs,pairs):
            if prob>=relation_threshold:associations[pid].update(vals)
    return {'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':entities,'association':[{'patient_id':p['patient_id'],'phenotype':sorted(associations[p['patient_id']])} for p in doc.get('patient',[])]}


def main():
    ap=argparse.ArgumentParser()
    for k in ['train','input','hpo','output','work-dir']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--mode',choices=['basic','morph'],default='basic');ap.add_argument('--synonyms',choices=['exact','related','all'],default='exact');ap.add_argument('--no-abbreviations',action='store_true');ap.add_argument('--entity-model',action='store_true');ap.add_argument('--entity-threshold',type=float,default=.45);ap.add_argument('--relation',choices=['rules','learned'],default='learned');ap.add_argument('--relation-threshold',type=float,default=.5)
    args=ap.parse_args();started=time.monotonic();src=baseline.sha256(__file__)
    train=baseline.read_jsonl(args.train);inputs=baseline.read_jsonl(args.input)
    assert not {str(x['pmc_id']) for x in train}&{str(x['pmc_id']) for x in inputs},'Training/inference overlap'
    assert all('entities' not in x and 'association' not in x for x in inputs),'Use label-free input'
    lex=OntologyLexicon(args.hpo,train,args.mode,args.synonyms,not args.no_abbreviations)
    em,es=fit_entity(train,lex) if args.entity_model else (None,{})
    rm,rs=fit_relation(train,lex) if args.relation=='learned' else (None,{})
    fitted=time.monotonic();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    preds=[predict(d,lex,em,rm,args.entity_threshold,args.relation_threshold,args.relation) for d in inputs]
    out.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds))
    work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True)
    manifest={'method':'ontology-expanded-evidence-v2','configuration':vars(args),'source_sha256':src,'baseline_source_sha256':baseline.sha256(baseline.__file__),'inputs':{k:{'path':getattr(args,k),'sha256':baseline.sha256(getattr(args,k))} for k in ['train','input','hpo']},'output':{'path':str(out.resolve()),'sha256':baseline.sha256(out),'bytes':out.stat().st_size},'training_documents':len(train),'inference_documents':len(inputs),'candidate_model':es,'relation_model':rs,'fit_seconds':fitted-started,'predict_seconds':time.monotonic()-fitted,'input_annotations_used':False,'training_candidate_alias_policy':'leave-one-document-out'}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':main()
