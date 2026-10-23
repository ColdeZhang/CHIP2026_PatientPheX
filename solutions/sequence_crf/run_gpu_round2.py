#!/usr/bin/env python3
"""BiomedBERT sequence labels with a frozen SapBERT normalization comparison.

The ontology bank includes official HPO and --train aliases only. Exact aliases
retain the supervised mapping; novel expressions use dense concept retrieval.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
import run as base
import run_biomedbert as ner


def associations(doc,entities,vectorizer,relation,threshold):
    rows=[];pairs=[];result={p['patient_id']:set() for p in doc.get('patient',[])}
    for e in entities:
        for p in doc.get('patient',[]):rows.append(base.pair_features(doc,e,p));pairs.append((e,p))
    probs=relation.predict_proba(vectorizer.transform(rows))[:,1] if rows else []
    for prob,(e,p) in zip(probs,pairs):
        if prob>=threshold:result[p['patient_id']].update(base.association_keys(e))
    return [{'patient_id':p['patient_id'],'phenotype':sorted(result[p['patient_id']])} for p in doc.get('patient',[])]


def main():
    p=argparse.ArgumentParser()
    for k in ('train','input','hpo','work-dir','model','sapbert','semantic-source'):p.add_argument('--'+k,required=True)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--ner-threshold',type=float,default=.55);p.add_argument('--relation-threshold',type=float,default=.55)
    p.add_argument('--semantic-threshold',type=float,default=.65);p.add_argument('--reuse-model',action='store_true')
    a=p.parse_args();start=time.monotonic();work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    hashes={Path(x).name:base.sha(x) for x in [__file__,base.__file__,ner.__file__,a.semantic_source]}
    train=base.read_jsonl(a.train);inputs=base.read_jsonl(a.input)
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    torch.set_num_threads(1);random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    device=torch.device('cuda');linker=base.Linker(train,a.hpo);vec,rel,relstats=base.fit_relation(train,linker)
    fp={'train':base.sha(a.train),'hpo':base.sha(a.hpo),'source_hashes':hashes,
        'epochs':a.epochs,'learning_rate':a.learning_rate,'batch_size':a.batch_size,'max_length':a.max_length,'seed':20261023}
    if a.reuse_model and (work/'ner-fingerprint.json').exists():
        if json.loads((work/'ner-fingerprint.json').read_text())!=fp:raise ValueError('NER cache mismatch')
        tokenizer=AutoTokenizer.from_pretrained(work/'ner',use_fast=True)
        model=AutoModelForTokenClassification.from_pretrained(work/'ner').to(device)
        stats=json.loads((work/'training-stats.json').read_text())
    else:
        tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
        model=AutoModelForTokenClassification.from_pretrained(a.model,num_labels=3,id2label=ner.LABELS,label2id={v:k for k,v in ner.LABELS.items()})
        torch.manual_seed(20261023);torch.nn.init.normal_(model.classifier.weight,mean=0.,std=model.config.initializer_range);torch.nn.init.zeros_(model.classifier.bias)
        stats=ner.train_model(train,linker,tokenizer,model,a,device)
        model.save_pretrained(work/'ner');tokenizer.save_pretrained(work/'ner')
        (work/'ner-fingerprint.json').write_text(json.dumps(fp,indent=2)+'\n');(work/'training-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    fitted=time.monotonic();raw=[]
    for d in inputs:raw.append(ner.predict_entities(d,tokenizer,model,linker,a,device))
    params=sum(x.numel() for x in model.parameters());del model;torch.cuda.empty_cache()
    spec=importlib.util.spec_from_file_location('sapbert_encoder',a.semantic_source);sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    encoder=sem.Encoder(a.sapbert,'sapbert','cuda',256,work/'embeddings',1)
    names=linker.names;ids=[linker.aliases[n].most_common(1)[0][0] for n in names];bank=encoder.encode(names,'hpo-and-fold-aliases')
    surfaces=sorted({base.norm(e['text']) for es in raw for e in es if base.norm(e['text']) not in linker.aliases})
    vectors=encoder.encode(surfaces,'novel-sequence-spans');idx,sim=encoder.nearest(vectors,bank)
    semantic={s:(ids[int(i)],float(v),names[int(i)]) for s,i,v in zip(surfaces,idx,sim)}
    outputs={};comparisons=[]
    for variant in ('char','sapbert'):
        out=work/(variant+'.pred.jsonl');count=0
        with out.open('w') as h:
            for d,es in zip(inputs,raw):
                ents=[]
                for e in es:
                    e=dict(e);key=base.norm(e['text'])
                    if variant=='sapbert' and key in semantic:
                        hid,score,alias=semantic[key]
                        if score>=a.semantic_threshold:e['identifier']=hid
                        comparisons.append({'pmc_id':d['pmc_id'],'offset':e['offset'],'text':e['text'],'char_id':next(x['identifier'] for x in es if x['offset']==e['offset'] and x['length']==e['length']),
                            'sapbert_id':hid,'similarity':score,'matched_alias':alias})
                    ents.append(e)
                pred={'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':ents,'association':associations(d,ents,vec,rel,a.relation_threshold)}
                h.write(json.dumps(pred,ensure_ascii=False)+'\n');count+=len(ents)
        outputs[variant]={'path':str(out),'sha256':base.sha(out),'bytes':out.stat().st_size,'entities':count}
    (work/'link-comparison.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
    manifest={'method':'BiomedBERT NER + exact-fold-alias / SapBERT dense normalization','config':vars(a),'command':sys.argv,
        'fingerprint':fp,'input_sha256':base.sha(a.input),'train_pmcs':[d['pmc_id'] for d in train],'input_pmcs':[d['pmc_id'] for d in inputs],
        'model_parameters':params,'sapbert_manifest':json.loads((Path(a.sapbert)/'manifest.json').read_text()),
        'training_stats':stats,'relation_stats':relstats,'outputs':outputs,'fit_seconds':fitted-start,'total_seconds':time.monotonic()-start,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'input_annotations_used':False,
        'runtime_versions':{**base.runtime_versions(),'torch':torch.__version__,'transformers':__import__('transformers').__version__}}
    (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'stage':'finished','seconds':manifest['total_seconds'],'outputs':outputs}),flush=True)
if __name__=='__main__':main()
