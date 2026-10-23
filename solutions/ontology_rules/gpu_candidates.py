#!/usr/bin/env python3
"""Cache ontology-linked span evidence using a frozen, locally supplied encoder.

All fit labels are --train only; --input must be disjoint and label free. This
script produces scored textual candidates, not gold-conditioned predictions.
"""
import argparse, importlib.util, json, time, gzip, hashlib
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for k in ['train','input','hpo','model-dir','semantic-source','work-dir']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--max-candidates',type=int,default=6000)
    a=ap.parse_args();work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();source=sha(__file__)
    spec=importlib.util.spec_from_file_location('semantic_encoder',a.semantic_source);sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    train=sem.read_jsonl(a.train);inputs=sem.read_jsonl(a.input)
    assert not {d['pmc_id'] for d in train}&{d['pmc_id'] for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    encoder=sem.Encoder(a.model_dir,'sapbert','cuda',256,work/'embeddings',1)
    class UniqueEncoder:
        def __init__(self,e):self.e=e
        def encode(self,texts,label):
            import numpy as np
            names=sorted(set(texts));lookup={s:i for i,s in enumerate(names)}
            values=self.e.encode(names,label+'-unique')
            return values[np.array([lookup[t] for t in texts])]
        def nearest(self,*args):return self.e.nearest(*args)
    args=argparse.Namespace(seed=20261023,max_candidates=a.max_candidates,negatives_per_doc=500)
    terms,canonical=sem.hpo_terms(a.hpo)
    system=sem.System(train,terms,canonical,UniqueEncoder(encoder),args)
    output=work/'scored-candidates.jsonl.gz';counts={}
    with gzip.open(output,'wt',encoding='utf-8') as f:
        for d in inputs:
            cs=system.candidates(d)
            vectors=system.encoder.encode([sem.norm(c['text']) for c in cs],'candidates-'+str(d['pmc_id']))
            probs=system.detector.predict_proba(vectors)[:,1]
            indices,sims=system.encoder.nearest(vectors,system.bank)
            out=[]
            for c,p,idx,sim in zip(cs,probs,indices,sims):
                ident=system.best_id(sem.norm(c['text'])) if c['exact'] else system.label_ids[int(idx)]
                out.append(dict(c,identifier=ident,probability=float(p),similarity=float(sim)))
            f.write(json.dumps({'pmc_id':d['pmc_id'],'candidates':out})+'\n');f.flush()
            counts[str(d['pmc_id'])]=len(out)
            print(json.dumps({'event':'scored','pmc_id':d['pmc_id'],'candidates':len(out)}),flush=True)
    manifest={'source_sha256':source,'semantic_source_sha256':sha(a.semantic_source),'config':vars(a),'inputs':{k:sha(getattr(a,k)) for k in ['train','input','hpo']},'training':system.training_stats,'input_annotations_used':False,'model_manifest':json.loads((Path(a.model_dir)/'manifest.json').read_text()),'counts':counts,'output':{'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output)},'elapsed_seconds':time.monotonic()-started}
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'event':'finished','seconds':manifest['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
