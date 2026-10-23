#!/usr/bin/env python3
"""Patient-concept evidence classification with fold-local BiomedBERT supervision.

Gold relation labels are weak document-level labels; multiple phenotype mentions
are jointly presented with explicit target/other patient markers. No input gold
annotations are consumed. The input prediction supplies phenotype hypotheses.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
import run as base


def make_examples(doc,entities,linker,training=False):
    groups=defaultdict(list)
    for e in entities:
        if str(e.get('note') or '').upper()=='NO':continue
        hid=linker.canonical(e.get('identifier','-1'))
        if not hid:continue
        e=dict(e,identifier=hid)
        for key in base.association_keys(e):groups[key].append(e)
    gold=base.gold_associations(doc,linker) if training else None
    rows=[]
    for key,group in groups.items():
        name=linker.terms.get(key,{}).get('names',[key])[0]
        for patient in doc.get('patient',[]):
            pid=patient['patient_id'];names=sorted({m.get('text','') for m in patient.get('mention',[]) if m.get('text')})
            title='Patient '+pid+': '+', '.join(names[:8])+'. Phenotype: '+name+'.'
            ranked=[]
            for e in group:
                pf=base.pair_features(doc,e,patient)
                rank=4*pf['same_sentence']+2*pf['same_passage']+pf['nearest']-.04*pf['log_dist']-pf['general_sentence']
                ranked.append((rank,e))
            contexts=[];seen=set()
            for _,e in sorted(ranked,key=lambda x:-x[0]):
                pi,p=base.passage_for(doc,e['offset']);text=p.get('text','');baseoff=int(p.get('offset',0));loc=e['offset']-baseoff
                a=max(0,loc-220);b=min(len(text),loc+e['length']+220)
                if any(oi==pi and abs(oa-a)<120 for oi,oa in seen):continue
                seen.add((pi,a));marks=[]
                marks.append((loc-a,loc-a+e['length'],'[unused5]','[unused6]'))
                for other in doc.get('patient',[]):
                    for m in other.get('mention',[]):
                        pos=int(m['offset'])-baseoff
                        if a<=pos and pos+int(m.get('length',len(m.get('text',''))))<=b:
                            own=other['patient_id']==pid
                            marks.append((pos-a,pos-a+int(m.get('length',len(m.get('text','')))),'[unused1]' if own else '[unused3]','[unused2]' if own else '[unused4]'))
                snippet=text[a:b]
                for x,y,left,right in sorted(marks,reverse=True):snippet=snippet[:y]+' '+right+' '+snippet[y:];snippet=snippet[:x]+' '+left+' '+snippet[x:]
                contexts.append(str(p.get('section_type',''))+': '+snippet)
                if len(contexts)>=3:break
            row={'pmc_id':doc['pmc_id'],'patient_id':pid,'concept':key,'header':title,'evidence':' [SEP] '.join(contexts)}
            if training:row['label']=int(key in gold.get(pid,set()))
            rows.append(row)
    return rows

class Collator:
    def __init__(self,tokenizer,max_length):self.tokenizer=tokenizer;self.length=max_length
    def __call__(self,rows):
        batch=self.tokenizer([r['header'] for r in rows],[r['evidence'] for r in rows],padding=True,truncation='only_second',max_length=self.length,return_tensors='pt')
        if 'label' in rows[0]:batch['labels']=torch.tensor([r['label'] for r in rows])
        return batch

def main():
    p=argparse.ArgumentParser()
    for k in ('train','input','predictions','hpo','model','work-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--reuse-model',action='store_true')
    a=p.parse_args();start=time.monotonic();work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    train=base.read_jsonl(a.train);inputs=base.read_jsonl(a.input);preds=base.read_jsonl(a.predictions)
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    lookup={str(d['pmc_id']):d for d in preds};assert set(lookup)=={str(d['pmc_id']) for d in inputs}
    linker=base.Linker(train,a.hpo)
    fp={'train_sha256':base.sha(a.train),'hpo_sha256':base.sha(a.hpo),'source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__),
        'epochs':a.epochs,'batch_size':a.batch_size,'max_length':a.max_length,'learning_rate':a.learning_rate,'seed':20261023}
    rows=[]
    for d in train:rows.extend(make_examples(d,d.get('entities',[]),linker,training=True))
    if a.reuse_model and (work/'fingerprint.json').exists():
        if json.loads((work/'fingerprint.json').read_text())!=fp:raise ValueError('Relation checkpoint fingerprint mismatch')
        tok=AutoTokenizer.from_pretrained(work/'model',use_fast=True);model=AutoModelForSequenceClassification.from_pretrained(work/'model').cuda()
        stats=json.loads((work/'training-stats.json').read_text())
    else:
        tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).cuda()
        loader=DataLoader(rows,batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=Collator(tok,a.max_length))
        optim=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01);steps=len(loader)*a.epochs
        scheduler=get_linear_schedule_with_warmup(optim,int(.1*steps),steps);scaler=torch.amp.GradScaler('cuda')
        stats={'examples':len(rows),'positive':sum(r['label'] for r in rows),'epochs':[]}
        counts=np.bincount([r['label'] for r in rows],minlength=2);weights=torch.tensor(len(rows)/(2*counts),dtype=torch.float32,device='cuda')
        for epoch in range(a.epochs):
            model.train();loss_sum=0.;es=time.monotonic()
            for batch in loader:
                batch={k:v.cuda() for k,v in batch.items()};y=batch.pop('labels');optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(**batch).logits;loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=weights)
                scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optim);scaler.update();scheduler.step();loss_sum+=float(loss.detach())
            stat={'epoch':epoch+1,'loss':loss_sum/len(loader),'seconds':time.monotonic()-es};stats['epochs'].append(stat);print(json.dumps(stat),flush=True)
        model.save_pretrained(work/'model');tok.save_pretrained(work/'model');(work/'fingerprint.json').write_text(json.dumps(fp,indent=2)+'\n');(work/'training-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    fit_end=time.monotonic();test=[]
    for d in inputs:test.extend(make_examples(d,lookup[str(d['pmc_id'])]['entities'],linker))
    loader=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=Collator(tok,a.max_length));prob=[];model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.float16):logits=model(**batch).logits
            prob.extend(torch.softmax(logits.float(),-1)[:,1].cpu().tolist())
    predicted=defaultdict(set);evidence=[]
    for row,s in zip(test,prob):
        evidence.append({**row,'probability':s})
        if s>=a.threshold:predicted[(str(row['pmc_id']),row['patient_id'])].add(row['concept'])
    out=work/'pred.jsonl'
    with out.open('w') as h:
        for d in inputs:
            pr=lookup[str(d['pmc_id'])];pr=dict(pr,association=[{'patient_id':p['patient_id'],'phenotype':sorted(predicted[(str(d['pmc_id']),p['patient_id'])])} for p in d.get('patient',[])])
            h.write(json.dumps(pr,ensure_ascii=False)+'\n')
    (work/'pair-scores.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in evidence))
    manifest={'method':'BiomedBERT patient-concept multi-evidence classification','fingerprint':fp,'config':vars(a),'command':sys.argv,
       'train_pmcs':[d['pmc_id'] for d in train],'input_pmcs':[d['pmc_id'] for d in inputs],'input_sha256':base.sha(a.input),'entities_sha256':base.sha(a.predictions),
       'output_sha256':base.sha(out),'output_bytes':out.stat().st_size,'parameters':sum(p.numel() for p in model.parameters()),'training_stats':stats,
       'fit_seconds':fit_end-start,'total_seconds':time.monotonic()-start,'input_annotations_used':False,'source_note':'Only competition train labels; domain-pretrained BiomedBERT base; no external phenotype annotations'}
    (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'stage':'complete','seconds':manifest['total_seconds']}),flush=True)
if __name__=='__main__':main()
