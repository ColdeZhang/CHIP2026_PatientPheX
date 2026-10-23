#!/usr/bin/env python3
"""OOF-calibrated CRF/dictionary spans and patient-concept aggregation.

All supervised resources are built within --train. No gold input labels are read.
The two internal partitions are for candidate calibration, not reported evaluation.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
from collections import defaultdict, Counter
import copy
import json
import math
from pathlib import Path
import pickle
import random
import re
import sys
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import run as base

SEED=20261023
NEG=re.compile(r'\b(?:no|not|without|denied|denies|negative for|free of|unaffected)\b',re.I)

def gold_spans(doc, linker):
    values=defaultdict(set)
    for e in doc.get('entities',[]):
        if str(e.get('note') or '').upper()=='NO':continue
        hid=linker.canonical(e.get('identifier','-1'))
        if hid:values[(int(e['offset']),int(e['length']))].update(hid.split(';'))
    return values

def candidate_entities(doc, crf, linker):
    """Build CRF spans and ontology/fold-alias spans with out-of-sample evidence."""
    found={}
    for pi,p in enumerate(doc.get('full_text',[])):
        toks=base.tokenize(p)
        if not toks:continue
        feats=base.features(toks,p,linker)
        marg=[None]*len(toks);decoded=['O']*len(toks)
        for a,b in base.chunks(toks):
            marg[a:b]=crf.predict_marginals_single(feats[a:b]);decoded[a:b]=crf.predict_single(feats[a:b])
        spans={};start=None
        for i in range(len(toks)+1):
            label=decoded[i] if i<len(toks) else 'O'
            if start is not None and label!='I':spans[(start,i)]=True;start=None
            if label=='B' or (label=='I' and start is None):start=i
        for a in range(len(toks)):
            node=linker.trie
            for b in range(a,min(a+14,len(toks))):
                node=node.get(toks[b][0].lower())
                if node is None:break
                if None in node:spans.setdefault((a,b+1),False)
        for (a,b),viterbi in spans.items():
            begin,end=toks[a][1],toks[b-1][2]
            text=p['text'][begin-int(p.get('offset',0)):end-int(p.get('offset',0))]
            if not text.strip() or not any(x.isalpha() for x in text):continue
            hid,score=linker.link(text)
            normalized=base.norm(text)
            probs=[marg[a].get('B',0.)]+[marg[k].get('I',0.) for k in range(a+1,b)]
            positive=[marg[k].get('B',0.)+marg[k].get('I',0.) for k in range(a,b)]
            cand={'identifier':hid,'type':'Phenotype','offset':begin,'length':end-begin,'text':text,'note':None,
                  '_viterbi':bool(viterbi),'_prob':float(np.mean(probs)), '_min':min(probs),
                  '_positive':float(np.mean(positive)), '_link':float(score),'_pi':pi,
                  '_alias_support':float(sum(linker.aliases.get(normalized,{}).values())),
                  '_alias_count':len(linker.aliases.get(normalized,{}))}
            found[(begin,end,hid)]=cand
    return sorted(found.values(),key=lambda e:(e['offset'],e['length'],e['identifier']))

def entity_features(doc,e):
    _,p=base.passage_for(doc,e['offset']);text=p.get('text','');start=e['offset']-int(p.get('offset',0));end=start+e['length']
    left=text[max(0,start-100):start];right=text[end:end+80]
    clause=re.split(r'[.;!?\n]',left)[-1]
    return {'viterbi':float(e['_viterbi']),'ner_prob':e['_prob'],'ner_min':e['_min'],'ner_positive':e['_positive'],
        'link_score':e['_link'],'alias_support':math.log1p(e['_alias_support']),'alias_count':e['_alias_count'],
        'length':len(e['text']),'tokens':len(base.norm(e['text']).split()),'uppercase':float(e['text'].isupper()),
        'short':float(len(e['text'])<5),'unknown':float(e['identifier']=='-1'),'section':str(p.get('section_type','')),
        'type':str(p.get('type','')),'negative':float(bool(NEG.search(clause[-70:])) and 'not only' not in clause.lower()),
        'family':float(bool(base.FAMILY.search(clause))),'number_context':float(bool(re.search(r'\b\d+\s*%|\b\d+/\d+',left[-40:]+right[:20]))),
        'in_parentheses':float(left.rstrip().endswith('(') and right.lstrip().startswith(')')),
        'phenotypic_suffix':float(e['text'].lower().endswith(('ia','ism','osis','pathy','plasia','cephaly','dactyly'))),
        'presence_context':float(bool(re.search(r'\b(?:had|has|with|showed|presented|revealed|exhibited)\b',left[-50:],re.I)))}

def fit_classifier(rows,y,kind='hist'):
    vec=DictVectorizer(sparse=False,dtype=np.float32)
    x=vec.fit_transform(rows)
    if kind=='logistic':
        clf=LogisticRegression(C=.3,max_iter=400,class_weight='balanced',solver='liblinear',random_state=SEED)
        clf.fit(x,y)
    else:
        clf=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,max_depth=5,min_samples_leaf=20,
                l2_regularization=5.,learning_rate=.07,random_state=SEED,early_stopping=False)
        clf.fit(x,y)
    return vec,clf

def probabilities(model,rows):
    return model[1].predict_proba(model[0].transform(rows))[:,1] if rows else []

def select_entities(doc,cands,model,threshold,baseline=False):
    scores=probabilities(model,[entity_features(doc,e) for e in cands])
    for e,s in zip(cands,scores):e['_entity_probability']=float(s)
    if baseline:
        eligible=[e for e in cands if e['_viterbi'] and e['_prob']>=.55]
    else:eligible=[e for e in cands if e['_entity_probability']>=threshold]
    # Resolve conflicting nested candidates using learned evidence; leave disjoint spans.
    kept=[]
    for e in sorted(eligible,key=lambda e:(-e['_entity_probability'],-e['length'])):
        if any(e['offset']<o['offset']+o['length'] and o['offset']<e['offset']+e['length'] for o in kept):continue
        kept.append(e)
    return sorted(kept,key=lambda e:(e['offset'],e['length'],e['identifier']))

def expanded_doc(doc):
    """Expand only unambiguous supplied patient names, never generic case nouns."""
    result=copy.deepcopy({k:v for k,v in doc.items() if k not in ('entities','association')})
    names=defaultdict(set)
    for p in result.get('patient',[]):
        for m in p.get('mention',[]):
            name=m.get('text','')
            if len(name)>=3 and not re.fullmatch(r'(?i)(?:the |a |an )?(?:patient|proband|case|child|boy|girl|male|female|man|woman|individual)s?',name):names[name.lower()].add(p['patient_id'])
    for p in result.get('patient',[]):
        offsets={int(m['offset']) for m in p.get('mention',[])}
        for name,owners in names.items():
            if owners!={p['patient_id']}:continue
            pat=re.compile(r'(?<!\w)'+re.escape(name)+r'(?!\w)',re.I)
            for passage in result.get('full_text',[]):
                for m in pat.finditer(passage.get('text','')):
                    offset=int(passage.get('offset',0))+m.start()
                    if offset not in offsets:
                        p.setdefault('mention',[]).append({'offset':offset,'length':len(m.group()),'text':m.group()});offsets.add(offset)
    return result

def concept_groups(entities):
    groups=defaultdict(list)
    for e in entities:
        for key in base.association_keys(e):groups[key].append(e)
    return groups

def group_features(doc,entities,patient,key):
    mention_features=[base.pair_features(doc,e,patient) for e in entities]
    f={'mentions':math.log1p(len(entities)),'patients':len(doc.get('patient',[])),
       'single':len(doc.get('patient',[]))==1,'unknown':not key.startswith('HP:'),
       'unique_passages':len(set(e['_pi'] for e in entities)),
       'entity_probability_max':max(e.get('_entity_probability',1.) for e in entities),
       'entity_probability_mean':float(np.mean([e.get('_entity_probability',1.) for e in entities])),
       'ner_prob_max':max(e['_prob'] for e in entities),'link_min':min(e['_link'] for e in entities),
       'text_length_min':min(len(e['text']) for e in entities)}
    numeric=set(k for row in mention_features for k,v in row.items() if isinstance(v,(int,float,bool)))
    for name in numeric:
        vals=[float(row.get(name,0)) for row in mention_features]
        f[name+'_max']=max(vals);f[name+'_min']=min(vals);f[name+'_mean']=float(np.mean(vals))
    for row in mention_features:
        f['section='+row['section']]=1.;f['ptype='+row['ptype']]=1.
    stats=Counter()
    for e,row in zip(entities,mention_features):
        ef=entity_features(doc,e)
        _,p=base.passage_for(doc,e['offset'])
        case=bool(re.search(r'case|patient|clinical|result',str(p.get('section_type','')),re.I))
        stats['case_mentions']+=case
        stats['nonnegative_near']+=not ef['negative'] and row['nearest']
        stats['nonnegative_same_sentence']+=not ef['negative'] and row['same_sentence']
        stats['nonnegative_same_passage']+=not ef['negative'] and row['same_passage']
        stats['negative_mentions']+=bool(ef['negative'])
        stats['exclusive_passage']+=row['same_passage'] and not row['other_same_passage']
        stats['case_near']+=case and row['nearest'] and not ef['negative']
    for k,v in stats.items():f[k]=math.log1p(v);f[k+'_fraction']=v/len(entities)
    return f

def relation_rows(doc,entities):
    doc=expanded_doc(doc);rows=[];pairs=[]
    for key,group in concept_groups(entities).items():
        for patient in doc.get('patient',[]):
            rows.append(group_features(doc,group,patient,key));pairs.append((patient['patient_id'],key))
    return rows,pairs

def export(doc,entities,relation,threshold):
    rows,pairs=relation_rows(doc,entities)
    assoc={p['patient_id']:set() for p in doc.get('patient',[])}
    for (pid,key),prob in zip(pairs,probabilities(relation,rows)):
        if prob>=threshold:assoc[pid].add(key)
    return {'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),
      'entities':[{k:v for k,v in e.items() if not k.startswith('_')} for e in entities],
      'association':[{'patient_id':p['patient_id'],'phenotype':sorted(assoc[p['patient_id']])} for p in doc.get('patient',[])]}

def train_pipeline(train,hpo,work,iterations):
    order=list(range(len(train)));random.Random(SEED).shuffle(order)
    partitions=[set(order[::2]),set(order[1::2])];oof={};training_stats={}
    for fi,indices in enumerate(partitions):
        subset=[d for i,d in enumerate(train) if i not in indices]; held=[d for i,d in enumerate(train) if i in indices]
        local=work/f'calibration-{fi}';local.mkdir(exist_ok=True)
        crf,linker,_,_,stats=base.fit(subset,hpo,local,iterations)
        for doc in held:oof[str(doc['pmc_id'])]=candidate_entities(doc,crf,linker)
        training_stats[f'calibration_{fi}']={'train_pmcs':[d['pmc_id'] for d in subset],'heldout_pmcs':[d['pmc_id'] for d in held],**stats}
        print(json.dumps({'stage':'oof_candidates','fold':fi,'documents':len(held)}),flush=True)
    gold_linker=base.Linker([],hpo)
    ex,ey=[],[]
    for doc in train:
        gold=gold_spans(doc,gold_linker)
        for e in oof[str(doc['pmc_id'])]:
            truth=gold.get((e['offset'],e['length']),set());ids=set(e['identifier'].split(';'))
            ex.append(entity_features(doc,e));ey.append(bool(truth and (truth&ids or '-1' in truth)))
    entity_model=fit_classifier(ex,ey)
    rx,ry=[],[]
    for doc in train:
        # Broad threshold covers both strict and permissive inference variants.
        selected=select_entities(doc,oof[str(doc['pmc_id'])],entity_model,.10)
        rows,pairs=relation_rows(doc,selected);gold=base.gold_associations(doc,gold_linker)
        rx.extend(rows);ry.extend(key in gold.get(pid,set()) for pid,key in pairs)
    relation_model=fit_classifier(rx,ry)
    crf,linker,_,_,stats=base.fit(train,hpo,work,iterations)
    training_stats.update({'entity_candidates':len(ey),'entity_positive':sum(ey),'relation_candidates':len(ry),'relation_positive':sum(ry),'final':stats})
    return {'crf':crf,'linker':linker,'entity_model':entity_model,'relation_model':relation_model,'training_stats':training_stats}

def main():
    parser=argparse.ArgumentParser()
    for key in ('train','input','hpo','output','work-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--entity-threshold',type=float,default=.5)
    parser.add_argument('--relation-threshold',type=float,default=.5)
    parser.add_argument('--baseline-entities',action='store_true')
    parser.add_argument('--reuse-model',action='store_true')
    args=parser.parse_args();start=time.monotonic();work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True)
    train=base.read_jsonl(args.train);inputs=base.read_jsonl(args.input)
    assert not ({str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs}),'Training and prediction documents overlap'
    fp={'train_sha256':base.sha(args.train),'hpo_sha256':base.sha(args.hpo),'iterations':args.iterations,
        'source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__)}
    cache=work/'round2.pkl'
    if args.reuse_model and cache.exists():
        with cache.open('rb') as h:saved=pickle.load(h)
        if saved['fingerprint']!=fp:raise ValueError('Cache fingerprint mismatch')
        model=saved['pipeline']
    else:
        model=train_pipeline(train,args.hpo,work,args.iterations)
        with cache.open('wb') as h:pickle.dump({'fingerprint':fp,'pipeline':model},h,protocol=5)
    fitted=time.monotonic();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter();raw_candidates=[]
    with output.open('w') as h:
        for doc in inputs:
            cands=candidate_entities(doc,model['crf'],model['linker']);raw_candidates.append({'pmc_id':doc['pmc_id'],'candidates':cands})
            entities=select_entities(doc,cands,model['entity_model'],args.entity_threshold,args.baseline_entities)
            pred=export(doc,entities,model['relation_model'],args.relation_threshold)
            h.write(json.dumps(pred,ensure_ascii=False)+'\n');counts['entities']+=len(entities);counts['relations']+=sum(len(a['phenotype']) for a in pred['association'])
    # Outside Git. Candidate evidence supports controlled threshold comparison without refitting.
    candidate_path=output.with_suffix('.candidates.jsonl')
    with candidate_path.open('w') as h:
        for row in raw_candidates:h.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'method':'OOF CRF+dictionary candidate calibration; patient-concept aggregation','fingerprint':fp,
      'command':sys.argv,'train_pmcs':[d['pmc_id'] for d in train],'input_pmcs':[d['pmc_id'] for d in inputs],
      'input_sha256':base.sha(args.input),'output_sha256':base.sha(output),'output_bytes':output.stat().st_size,
      'config':vars(args),'counts':dict(counts),'fit_seconds':fitted-start,'predict_seconds':time.monotonic()-fitted,
      'training_stats':model['training_stats'],'runtime_versions':base.runtime_versions(),'input_annotations_used':False}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='training_stats'}),flush=True)
if __name__=='__main__':main()
