#!/usr/bin/env python3
"""Paper-disjoint candidate preparation and a short-context biomedical verifier.

Preparation accepts labeled training papers and label-free inference papers.
GPU fitting consumes the frozen examples only; assembly never reads gold.
"""
import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def dump(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def zipped(path):
    with gzip.open(path,'rt') as f:return list(map(json.loads,f))


def prepare(a):
    import run as baseline
    import enhance
    from calibrate import pool,read_candidates
    terms,_,_,_=baseline.load_hpo(a.hpo)
    training={d['pmc_id']:d for d in read(a.train)}
    inputs=read(a.input);assert not set(training)&{d['pmc_id'] for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    def example(doc,c):
        _,p,s,e=enhance.location(doc,c['offset'],c['offset']+c['length'])
        names='; '.join(terms.get(h,{}).get('name','unlinked phenotype') for h in c['identifier'].split(';'))
        left=p['text'][max(0,s-180):s];right=p['text'][e:e+180]
        return {'pmc_id':doc['pmc_id'],'phrase':c['text']+' | '+names,
                'context':left+' [ '+c['text']+' ] '+right,'candidate':c}
    examples=[];seen=set();origins=[];rng=random.Random(20261023)
    for spec in json.loads(Path(a.training_manifest).read_text()):
        docs=read(spec['gold']);raw=read_candidates(spec['candidates']);base={d['pmc_id']:d for d in read(spec['base'])}
        m=json.loads(Path(spec['candidate_manifest']).read_text());assert m['inputs']['input']==sha(spec['input']);assert m['inputs']['train']==sha(spec['generator_train'])
        generator_ids={d['pmc_id'] for d in read(spec['generator_train'])};dids={d['pmc_id'] for d in docs}
        assert not dids&generator_ids and not dids&seen;seen|=dids
        origins.append({'validation_ids':sorted(dids),'generator_train_sha256':sha(spec['generator_train']),'candidate_sha256':sha(spec['candidates']),'generator_source_sha256':m['source_sha256']})
        for doc in docs:
            assert doc==training[doc['pmc_id']]
            truth={}
            for e in doc['entities']:
                if str(e.get('note','')).upper()!='NO':truth.setdefault((e['offset'],e['length']),set()).update(e['identifier'].split(';'))
            pos=[];neg=[]
            for c in pool(doc,base[doc['pmc_id']],raw[doc['pmc_id']]):
                ids=truth.get((c['offset'],c['length']),set());label=bool(ids and (set(c['identifier'].split(';'))<=ids or '-1' in ids))
                row=example(doc,c);row['label']=int(label);(pos if label else neg).append(row)
            count=min(len(neg),max(12,3*len(pos)))
            examples.extend(pos+rng.sample(neg,count))
    assert seen==set(training)
    inference=[];base={d['pmc_id']:d for d in read(a.base)};raw=read_candidates(a.candidates)
    for doc in inputs:
        inference.extend(example(doc,c) for c in pool(doc,base[doc['pmc_id']],raw[doc['pmc_id']]))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    dump(out/'train.jsonl.gz',examples);dump(out/'input.jsonl.gz',inference)
    meta={'config':vars(a),'source_sha256':sha(__file__),'inputs':{k:sha(getattr(a,k)) for k in ['training_manifest','train','input','hpo','base','candidates']},
          'training_documents':len(training),'input_documents':len(inputs),'training_examples':len(examples),'training_positive':sum(r['label'] for r in examples),
          'inference_examples':len(inference),'negative_sampling':'up to3 negatives per positive per paper, at least12 if available; seed20261023',
          'candidate_origins':origins,'input_annotations_used':False,'output_hashes':{k:sha(out/k) for k in ['train.jsonl.gz','input.jsonl.gz']}}
    (out/'preparation.manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


def fit(a):
    import torch
    import numpy as np
    import transformers
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    started=time.monotonic();source=sha(__file__);torch.set_num_threads(1)
    random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    assert torch.cuda.device_count()==1
    train=zipped(a.train_examples);inputs=zipped(a.input_examples)
    assert not {r['pmc_id'] for r in train}&{r['pmc_id'] for r in inputs}
    assert all('label' not in r for r in inputs)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2,local_files_only=True).cuda()
    parameter_count=sum(p.numel() for p in model.parameters());assert parameter_count<=10_000_000_000
    def tokens(rows):
        return tokenizer([r['phrase'] for r in rows],[r['context'] for r in rows],padding='max_length',truncation='only_second',max_length=a.max_length,return_tensors='pt')
    train_tokens=tokens(train);test_tokens=tokens(inputs);labels=torch.tensor([r['label'] for r in train],dtype=torch.long)
    positive=int(labels.sum());weight=torch.tensor([1.,(len(train)-positive)/max(positive,1)],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
    batches=(len(train)+a.batch_size-1)//a.batch_size;steps=batches*a.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(steps*.1)),steps)
    losses=[];model.train()
    for epoch in range(a.epochs):
        order=torch.randperm(len(train));total=0.
        for start in range(0,len(train),a.batch_size):
            ix=order[start:start+a.batch_size];batch={k:v[ix].cuda() for k,v in train_tokens.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**batch).logits;loss=torch.nn.functional.cross_entropy(logits.float(),labels[ix].cuda(),weight=weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);total+=float(loss.detach())*len(ix)
        losses.append(total/len(train));print(json.dumps({'epoch':epoch+1,'training_loss':losses[-1]}),flush=True)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out/'model',safe_serialization=True);tokenizer.save_pretrained(out/'model')
    model.eval();probabilities=[]
    with torch.inference_mode():
        for start in range(0,len(inputs),a.batch_size*2):
            batch={k:v[start:start+a.batch_size*2].cuda() for k,v in test_tokens.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**batch).logits
            probabilities.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
    grouped={}
    for row,p in zip(inputs,probabilities):grouped.setdefault(row['pmc_id'],[]).append(dict(row['candidate'],probability=p,previous_probability=row['candidate']['probability']))
    dump(out/'scored-candidates.jsonl.gz',[{'pmc_id':k,'candidates':v} for k,v in grouped.items()])
    files=[p for p in Path(a.model).iterdir() if p.suffix in {'.json','.bin','.safetensors','.txt'}]
    meta={'config':vars(a),'source_sha256':source,'input_hashes':{k:sha(getattr(a,k)) for k in ['train_examples','input_examples']},'base_model_files':{p.name:sha(p) for p in files},
          'parameter_count':parameter_count,'training_examples':len(train),'training_positive':positive,'training_losses':losses,'epochs':a.epochs,
          'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
          'trained_model_sha256':sha(out/'model/model.safetensors'),'output_sha256':sha(out/'scored-candidates.jsonl.gz'),'seconds':time.monotonic()-started}
    (out/'training.manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


def assemble(a):
    import merge_semantic
    docs=read(a.input);candidates={d['pmc_id']:d['candidates'] for d in zipped(a.candidates)};pred=[]
    assert all('entities' not in d and 'association' not in d for d in docs)
    for d in docs:
        selected=[]
        for c in sorted(candidates[d['pmc_id']],key=lambda c:(-c['probability'],-c['length'],c['offset'])):
            if c['probability']<a.threshold:continue
            if any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in selected):continue
            selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
        selected.sort(key=lambda e:e['offset']);pred.append({'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':selected,'association':merge_semantic.assign(d,selected)})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in pred))
    out.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(a),'source_sha256':sha(__file__),'input_sha256':sha(a.input),'candidates_sha256':sha(a.candidates),'output_sha256':sha(out)},indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare')
    for k in ['training-manifest','train','input','hpo','base','candidates','output-dir']:p.add_argument('--'+k,required=True)
    p=sub.add_parser('fit')
    for k in ['train-examples','input-examples','model','output-dir']:p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-length',type=int,default=160);p.add_argument('--learning-rate',type=float,default=2e-5)
    p=sub.add_parser('assemble')
    for k in ['input','candidates','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--threshold',type=float,default=.5)
    a=ap.parse_args();{'prepare':prepare,'fit':fit,'assemble':assemble}[a.mode](a)


if __name__=='__main__':main()
