#!/usr/bin/env python3
"""Fold-local context cross-encoder ranking of SapBERT HPO candidates.

Training retrieval excludes aliases originating only in the example's document.
The held-out input is label-free. Existing unambiguous exact aliases are protected.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
from collections import Counter,defaultdict
import importlib.util
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
import run_gpu_round2 as gpu
from run_gpu_round2 import associations


def alias_sources(train,linker):
    ontology=defaultdict(Counter);documents=defaultdict(lambda:defaultdict(Counter))
    for hid in sorted(linker.allowed):
        for name in linker.terms[hid]['names']:
            if len(base.norm(name))>=3:ontology[base.norm(name)][hid]+=.05
    for d in train:
        for e in d.get('entities',[]):
            hid=linker.canonical(e.get('identifier','-1'))
            if hid and hid!='-1' and e.get('text'):
                documents[base.norm(e['text'])][str(d['pmc_id'])][hid]+=1
    return ontology,documents


def candidate_bank(train,linker):
    ontology,documents=alias_sources(train,linker)
    names=sorted(set(ontology)|set(documents));all_counts={}
    for name in names:
        c=ontology[name].copy()
        for counts in documents[name].values():c.update(counts)
        all_counts[name]=c
    return names,all_counts,documents


def concept_text(hid,linker):
    return '; '.join(' / '.join(linker.terms[x]['names'][:3]) for x in hid.split(';') if x in linker.terms)


def context(doc,e):
    _,p=base.passage_for(doc,e['offset']);text=p.get('text','');pos=e['offset']-int(p.get('offset',0));end=pos+e['length']
    return str(p.get('section_type',''))+': '+text[max(0,pos-180):pos]+' [unused1] '+text[pos:end]+' [unused2] '+text[end:end+180]


def rows_for(doc,entities,linker,retrieved,names,counts,sources,topk,training=False):
    rows=[];groups=[];seen=set();stats=Counter()
    for ei,e in enumerate(entities):
        key=base.norm(e['text']);gold=linker.canonical(e.get('identifier','-1'))
        if training and (not gold or gold=='-1'):continue
        if training and (key,gold) in seen:continue
        seen.add((key,gold));candidates=[];used=set();pid=str(doc['pmc_id'])
        def add(hid,similarity):
            if hid and hid!='-1' and hid not in used:used.add(hid);candidates.append((hid,float(similarity)))
        # Existing ontology/supervised exact mappings only from allowed sources.
        exact=counts.get(key,Counter()).copy()
        if training:exact.subtract(sources.get(key,{}).get(pid,Counter()));exact=+exact
        for hid,_ in exact.most_common():add(hid,1.)
        for index,score in retrieved[key]:
            name=names[index];c=counts[name].copy()
            if training:c.subtract(sources[name].get(pid,Counter()));c=+c
            for hid,_ in c.most_common():add(hid,score)
            if len(candidates)>=topk:break
        candidates=candidates[:topk]
        if training:
            stats['examples']+=1;stats['gold_retrieved']+=gold in {x[0] for x in candidates}
            if gold not in {x[0] for x in candidates}:candidates.append((gold,0.))
        else:
            # Preserve a base candidate so the cross-encoder can leave it unchanged.
            if gold and gold!='-1' and gold not in {x[0] for x in candidates}:candidates.append((gold,0.))
        if not candidates:continue
        start=len(rows)
        for hid,similarity in candidates:
            row={'context':context(doc,e),'concept':concept_text(hid,linker),'identifier':hid,'similarity':similarity}
            if training:row['label']=int(hid==gold)
            rows.append(row)
        groups.append({'pmc_id':pid,'entity_index':ei,'start':start,'end':len(rows),'protected_exact':len(exact)==1,'exact_identifier':next(iter(exact)) if len(exact)==1 else None,'original_identifier':gold})
    return rows,groups,stats


class Collator:
    def __init__(self,tok,length):self.tok=tok;self.length=length
    def __call__(self,rows):
        x=self.tok([r['context'] for r in rows],[r['concept'] for r in rows],padding=True,truncation=True,max_length=self.length,return_tensors='pt')
        if 'label' in rows[0]:x['labels']=torch.tensor([r['label'] for r in rows])
        return x


def main():
    p=argparse.ArgumentParser()
    for k in ('train','input','predictions','hpo','model','sapbert','semantic-source','work-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--reuse-model',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();start=time.monotonic();work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    train=base.read_jsonl(a.train);inputs=base.read_jsonl(a.input);preds=base.read_jsonl(a.predictions)
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    lookup={str(d['pmc_id']):d for d in preds};assert set(lookup)=={str(d['pmc_id']) for d in inputs}
    linker=base.Linker(train,a.hpo);names,counts,sources=candidate_bank(train,linker)
    spec=importlib.util.spec_from_file_location('sapbert_encoder',a.semantic_source);sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    encoder=sem.Encoder(a.sapbert,'sapbert','cuda',256,work/'embeddings',1)
    bank=encoder.encode(names,'crossencoder-hpo-alias-bank')
    surfaces=sorted({base.norm(e['text']) for d in train for e in d.get('entities',[])}|{base.norm(e['text']) for d in preds for e in d['entities']})
    queries=encoder.encode(surfaces,'crossencoder-queries');retrieved={};bank_tensor=torch.tensor(bank,dtype=torch.float16,device='cuda').T
    with torch.inference_mode():
        for j in range(0,len(queries),128):
            q=torch.tensor(queries[j:j+128],dtype=torch.float16,device='cuda');scores,indices=(q@bank_tensor).topk(min(128,len(names)),dim=1)
            for surface,idx,score in zip(surfaces[j:j+128],indices.cpu().tolist(),scores.float().cpu().tolist()):retrieved[surface]=list(zip(idx,score))
    del encoder,bank_tensor,q,scores,indices;torch.cuda.empty_cache()
    training=[];retrieval_stats=Counter()
    for d in train:
        rows,_,stats=rows_for(d,d.get('entities',[]),linker,retrieved,names,counts,sources,a.top_k,True);training.extend(rows);retrieval_stats.update(stats)
    test=[];groups=[]
    for d in inputs:
        rows,gg,_=rows_for(d,lookup[str(d['pmc_id'])]['entities'],linker,retrieved,names,counts,sources,a.top_k)
        for g in gg:g['start']+=len(test);g['end']+=len(test)
        test.extend(rows);groups.extend(gg)
    fp={'dependency_sha256':gpu.source_fingerprint.source_hashes({'gpu_relation_helper':gpu.__file__,'semantic_encoder':a.semantic_source,'fingerprint_helper':gpu.source_fingerprint.__file__}), 'source_sha256':base.sha(__file__),'base_sha256':base.sha(base.__file__),'train_sha256':base.sha(a.train),'hpo_sha256':base.sha(a.hpo),
        'model_sha256':{f.name:base.sha(f) for f in sorted(Path(a.model).iterdir()) if f.is_file() and (f.suffix in {'.bin','.safetensors'} or f.name in {'config.json','vocab.txt','tokenizer.json'})},
        'sapbert_manifest':json.loads((Path(a.sapbert)/'manifest.json').read_text()),'epochs':a.epochs,'batch_size':a.batch_size,'max_length':a.max_length,'learning_rate':a.learning_rate,'top_k':a.top_k}
    prepared={'fingerprint':fp,'training_pairs':len(training),'test_pairs':len(test),'training_retrieval':dict(retrieval_stats),'input_sha256':base.sha(a.input),'predictions_sha256':base.sha(a.predictions),'input_annotations_used':False}
    (work/'preparation.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (work/'retrieval.jsonl').write_text(''.join(json.dumps({**g,'candidates':[{k:v for k,v in x.items() if k not in {'context','concept'}} for x in test[g['start']:g['end']]]})+'\n' for g in groups))
    print(json.dumps({'stage':'prepared','training_pairs':len(training),'test_pairs':len(test),'training_retrieval':dict(retrieval_stats)}),flush=True)
    if a.prepare_only:return
    if a.reuse_model and (work/'fingerprint.json').exists():
        assert json.loads((work/'fingerprint.json').read_text())==fp,'Cross-encoder checkpoint fingerprint mismatch'
        tok=AutoTokenizer.from_pretrained(work/'model',use_fast=True);model=AutoModelForSequenceClassification.from_pretrained(work/'model').cuda();stats=json.loads((work/'training-stats.json').read_text())
    else:
        tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2).cuda()
        loader=DataLoader(training,batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=Collator(tok,a.max_length));optim=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
        steps=len(loader)*a.epochs;schedule=get_linear_schedule_with_warmup(optim,int(.1*steps),steps);scaler=torch.amp.GradScaler('cuda');stats=[]
        label_counts=np.bincount([x['label'] for x in training],minlength=2);weights=torch.tensor(len(training)/(2*label_counts),device='cuda',dtype=torch.float32)
        for epoch in range(a.epochs):
            model.train();loss_sum=0.;t=time.monotonic()
            for batch in loader:
                batch={k:v.cuda() for k,v in batch.items()};labels=batch.pop('labels');optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(**batch).logits.float(),labels,weight=weights)
                scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optim);scaler.update();schedule.step();loss_sum+=float(loss.detach())
            row={'epoch':epoch+1,'loss':loss_sum/len(loader),'seconds':time.monotonic()-t};stats.append(row);print(json.dumps(row),flush=True)
        model.save_pretrained(work/'model');tok.save_pretrained(work/'model');(work/'fingerprint.json').write_text(json.dumps(fp,indent=2)+'\n');(work/'training-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    loader=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=Collator(tok,a.max_length));prob=[];model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.float16):logits=model(**batch).logits
            prob.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
    changes=0;scored=[]
    for g in groups:
        candidates=[dict(row,probability=prob[i]) for i,row in enumerate(test[g['start']:g['end']],g['start'])]
        best=max(candidates,key=lambda x:x['probability']);chosen=g['original_identifier'] if g['protected_exact'] or g['original_identifier']=='-1' else best['identifier']
        e=lookup[g['pmc_id']]['entities'][g['entity_index']];changes+=chosen!=e['identifier'];e['identifier']=chosen
        scored.append({**g,'chosen':chosen,'candidates':[{k:v for k,v in x.items() if k not in {'context','concept'}} for x in candidates]})
    vec,rel,relstats=base.fit_relation(train,linker);out=work/'pred.jsonl'
    with out.open('w') as h:
        for d in inputs:
            pr=lookup[str(d['pmc_id'])];pr=dict(pr,association=associations(d,pr['entities'],vec,rel,.55));h.write(json.dumps(pr,ensure_ascii=False)+'\n')
    (work/'scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in scored))
    manifest={**prepared,'config':vars(a),'command':sys.argv,'training_stats':stats,'relation_stats':relstats,'changed_identifiers':changes,'output_sha256':base.sha(out),'total_seconds':time.monotonic()-start}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'stage':'finished','changed_identifiers':changes,'seconds':manifest['total_seconds']}),flush=True)
if __name__=='__main__':main()
