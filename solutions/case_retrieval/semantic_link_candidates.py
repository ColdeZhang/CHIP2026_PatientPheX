#!/usr/bin/env python3
"""Frozen SapBERT nearest ontology/within-fold alias candidates for extracted quotes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--aliases',required=True);ap.add_argument('--queries');ap.add_argument('--output',required=True);ap.add_argument('--work-dir',required=True)
    ap.add_argument('--top-k',type=int,default=8);ap.add_argument('--batch-size',type=int,default=256)
    args=ap.parse_args();torch.set_num_threads(1);start=time.time();assert torch.cuda.device_count()==1 and torch.cuda.is_available()
    modelpath=Path(args.model);aliases=json.loads(Path(args.aliases).read_text());queries=json.loads(Path(args.queries).read_text()) if args.queries else []
    signature={'model_manifest':digest(modelpath/'manifest.json'),'aliases_sha256':digest(args.aliases),'code_sha256':digest(__file__),'pooling':'CLS_L2','max_length':32}
    work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True);key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();cache=work/(key+'.npy')
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModel.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float16).eval().cuda()
    def encode(texts):
        values=[]
        for i in range(0,len(texts),args.batch_size):
            batch=tokenizer(texts[i:i+args.batch_size],padding=True,truncation=True,max_length=32,return_tensors='pt').to('cuda')
            with torch.inference_mode():v=model(**batch).last_hidden_state[:,0].float();v=torch.nn.functional.normalize(v,dim=1)
            values.append(v.cpu().numpy())
        return np.concatenate(values) if values else np.empty((0,768),np.float32)
    if cache.exists():bank=np.load(cache)
    else:bank=encode([x['text'] for x in aliases]);np.save(cache,bank)
    vectors=encode([x['quote'] for x in queries]);bank_tensor=torch.from_numpy(bank).cuda();results=[]
    for start_index in range(0,len(queries),128):
        v=torch.from_numpy(vectors[start_index:start_index+128]).cuda()
        with torch.inference_mode():scores,indices=(v@bank_tensor.T).topk(min(100,len(aliases)),dim=1)
        for row,ss,ii in zip(queries[start_index:start_index+128],scores.cpu().tolist(),indices.cpu().tolist()):
            candidates=[];seen=set()
            for score,index in zip(ss,ii):
                alias=aliases[index]
                for identifier in alias['identifiers']:
                    if identifier in seen:continue
                    seen.add(identifier);candidates.append({'identifier':identifier,'alias':alias['text'],'score':score,'source':'SapBERT'})
                    if len(candidates)>=args.top_k:break
                if len(candidates)>=args.top_k:break
            results.append({'code':row['code'],'quote':row['quote'],'candidates':candidates})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,ensure_ascii=False))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(args),'signature':signature,'aliases':len(aliases),'queries':len(queries),'output_sha256':digest(out),'seconds':time.time()-start,'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__},indent=2)+'\n');print(json.dumps({'aliases':len(aliases),'queries':len(queries),'seconds':time.time()-start}))
if __name__=='__main__':main()
