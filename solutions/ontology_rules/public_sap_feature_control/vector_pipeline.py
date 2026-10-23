"""Bounded public SapBERT vector reuse; no task labels, new spans or IDs."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def array(path,expected=None):
    import numpy as np
    digest=sha(path)
    if expected is not None:assert digest==expected
    a=np.load(path,allow_pickle=False)
    assert sha(path)==digest and a.dtype==np.float32 and a.ndim==2 and a.shape[1]==768
    assert np.isfinite(a).all()
    norms=np.linalg.norm(a,axis=1)
    assert len(norms)==0 or np.max(np.abs(norms-1))<2e-5
    return a,dict(**ref(path),shape=list(a.shape),dtype=str(a.dtype),max_norm_error=float(np.max(np.abs(norms-1))) if len(norms) else 0)


def prepare(plan):
    import numpy as np
    out=Path('prepared');out.mkdir(exist_ok=False)
    root=Path(plan['remote_root']);sem=root/'semantic-neural/run.py'
    assert sha(sem)==plan['semantic_source_sha256']
    module=ast.parse(sem.read_text());nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','hpo_terms']]
    scope={'Path':Path,'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(sem),'exec'),scope)
    norm,hpo_terms=scope['norm'],scope['hpo_terms']
    model=root/'semantic-neural/model';assert sha(model/'manifest.json')==plan['model_manifest_sha256']
    mm=read(model/'manifest.json');assert mm['revision']==plan['model_revision']
    files={}
    for name,entry in mm['files'].items():
        expected=mm['derived_weights']['sha256'] if name=='model.safetensors' else entry['sha256']
        assert sha(model/name)==expected
        files[name]=ref(model/name)
    signature='sapbert'+sha(model/'config.json')+sha(model/'manifest.json')
    assert signature==plan['encoder_signature']
    assert sha('request.json.gz')==plan['request_sha256']
    with gzip.open('request.json.gz','rt') as f:req=json.load(f)
    names=req['phrases'];assert names==sorted(set(names)) and all(norm(s)==s for s in names)
    pairs=req['pairs'];assert pairs==sorted({tuple(p) for p in pairs}) or pairs==[list(p) for p in sorted({tuple(p) for p in pairs})]
    assert {p for p,h in pairs}<=set(names)
    labels=req['bank_labels'];assert labels==sorted(set(labels)) and len(labels)==42236
    digest=hashlib.sha256((signature+'\0'+'\0'.join(labels)).encode()).hexdigest()
    assert digest==plan['public_bank_key']
    bankpath=root/plan['public_bank'];bank,bankproof=array(bankpath,plan['public_bank_sha256']);assert len(bank)==len(labels)
    hpo=root/'semantic-neural/data/hp.obo';assert sha(hpo)==plan['hpo_sha256'];terms,canonical=hpo_terms(hpo)
    public={hid:sorted({norm(t) for t in [term['name']]+term['synonyms'] if len(t)>=3}) for hid,term in terms.items() if hid!='HP:0000118'}
    indexes={s:i for i,s in enumerate(labels)};alias_indices={}
    for hid,canon in canonical.items():
        if canon in public:
            assert all(s in indexes for s in public[canon])
            alias_indices[hid]=[indexes[s] for s in public[canon]]
    lookup={s:i for i,s in enumerate(names)};values=np.zeros((len(names),768),np.float32);seen=set();evidence=[]
    # Deterministic first cache wins. Cached vectors are frozen text-only public
    # encoder output; no training-frequency label assignment is consulted.
    for item in sorted(req['query_caches'],key=lambda x:(x['path'],x['pmc_id'])):
        ordered=item['texts'];assert ordered==sorted(set(ordered)) and all(norm(s)==s for s in ordered)
        key=hashlib.sha256((signature+'\0'+'\0'.join(ordered)).encode()).hexdigest()
        path=root/item['path'];assert path.name==key+'.npy'
        cached,proof=array(path);assert len(cached)==len(ordered)
        proof.update(ordered_texts_sha256=hashlib.sha256(json.dumps(ordered,ensure_ascii=False).encode()).hexdigest(),cache_key=key,pmc_id=item['pmc_id'])
        evidence.append(proof)
        for i,s in enumerate(ordered):
            if s in lookup and s not in seen:values[lookup[s]]=cached[i];seen.add(s)
    public_words={s for ss in public.values() for s in ss}
    for s in sorted(public_words & set(names)-seen):values[lookup[s]]=bank[indexes[s]];seen.add(s)
    missing=sorted(set(names)-seen);assert len(missing)==plan['expected_missing_phrases']==799
    np.save(out/'known-vectors.npy',values);write(out/'missing.json',missing);write(out/'alias-indices.json',alias_indices)
    write(out/'manifest.json',dict(source_sha256=sha(__file__),plan_sha256=sha('plan.json'),model_files=files,encoder_signature=signature,
        query_caches=evidence,public_bank=bankproof,public_alias_policy='current canonical and declared valid alt IDs receive only public name/EXACT aliases; old training best_id/extra aliases ignored',
        phrases=len(names),pairs=len(pairs),missing=len(missing),outputs={n:ref(out/n) for n in ['known-vectors.npy','missing.json','alias-indices.json']},no_labels=True,no_model_load=True))


def encode(plan):
    root=Path(plan['remote_root']);out=Path('new-embeddings');assert not out.exists();out.mkdir()
    prep=read('prepared/manifest.json');assert prep['plan_sha256']==sha('plan.json')
    missing=read('prepared/missing.json');assert sha('prepared/missing.json')==prep['outputs']['missing.json']['sha256'] and len(missing)==799
    gpu=plan['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
    status=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in status.splitlines() if x.split(',')[0].strip()==gpu);assert int(row.split(',')[1])<64
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    assert not any(x.split(',')[0].strip()==gpu for x in apps.splitlines())
    write(out/'fresh-GPU.json',dict(at=datetime.now(timezone.utc).isoformat(),row=row,compute_apps=apps))
    sempath=root/'semantic-neural/run.py';assert sha(sempath)==plan['semantic_source_sha256']
    spec=importlib.util.spec_from_file_location('frozen_semantic',sempath);sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    for name,item in prep['model_files'].items():assert sha(item['path'])==item['sha256']
    encoder=sem.Encoder(str(root/'semantic-neural/model'),'sapbert','cuda',256,out,1)
    assert encoder.signature==plan['encoder_signature'];encoded=encoder.encode(missing,'public-sap-only-missing799')
    encoder.torch.cuda.synchronize();key=hashlib.sha256((encoder.signature+'\0'+'\0'.join(missing)).encode()).hexdigest()
    path=out/(key+'.npy');actual,proof=array(path);assert len(actual)==len(encoded)==799
    write(out/'manifest.json',dict(source_sha256=sha(__file__),plan_sha256=sha('plan.json'),prepared_sha256=sha('prepared/manifest.json'),
          missing_texts_sha256=sha('prepared/missing.json'),vectors=proof,cache_key=key,phrases=799,torch=encoder.torch.__version__,
          representation='same source FP16 CUDA forward CLS/max32 then float32 L2 normalization',model=prep['model_files'],no_labels=True))


def cosine(plan):
    import numpy as np
    prep=read('prepared/manifest.json');new=read('new-embeddings/manifest.json')
    assert prep['plan_sha256']==new['plan_sha256']==sha('plan.json') and new['prepared_sha256']==sha('prepared/manifest.json')
    with gzip.open('request.json.gz','rt') as f:req=json.load(f)
    assert sha('request.json.gz')==plan['request_sha256'];names=req['phrases'];lookup={s:i for i,s in enumerate(names)}
    known=Path('prepared/known-vectors.npy');assert sha(known)==prep['outputs']['known-vectors.npy']['sha256'];vectors=np.load(known,allow_pickle=False)
    missing=read('prepared/missing.json');newvectors,_=array(new['vectors']['path'],new['vectors']['sha256'])
    for phrase,row in zip(missing,newvectors):vectors[lookup[phrase]]=row
    assert np.isfinite(vectors).all() and np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))<2e-5
    bank,_=array(prep['public_bank']['path'],prep['public_bank']['sha256'])
    assert sha('prepared/alias-indices.json')==prep['outputs']['alias-indices.json']['sha256'];aliases=read('prepared/alias-indices.json')
    atoms=defaultdict(set)
    for phrase,hid in req['pairs']:atoms[phrase].update(hid.split(';'))
    matches={};dotproducts=0
    for phrase,ids in atoms.items():
        vector=vectors[lookup[phrase]]
        for hid in sorted(ids):
            indices=aliases.get(hid,[])
            if indices:matches[phrase,hid]=float((bank[indices]@vector).max());dotproducts+=len(indices)
    out=Path('features');out.mkdir(exist_ok=False);nobs=0;missing_count=0
    with gzip.open(out/'lookup.jsonl.gz','wt') as f:
        for phrase,hid in req['pairs']:
            values=[matches.get((phrase,atom)) for atom in hid.split(';')]
            observed=bool(values) and all(value is not None for value in values)
            score=min(values) if observed else 0.;assert -1.0001<=score<=1.0001
            f.write(json.dumps(dict(text=phrase,identifier=hid,public_sap_cosine=score,public_sap_observed=observed),ensure_ascii=False)+'\n')
            nobs+=observed;missing_count+=not observed
    write(out/'manifest.json',dict(source_sha256=sha(__file__),plan_sha256=sha('plan.json'),prepared_sha256=sha('prepared/manifest.json'),new_embedding_sha256=sha('new-embeddings/manifest.json'),lookup=ref(out/'lookup.jsonl.gz'),pairs=len(req['pairs']),observed=nobs,missing=missing_count,alias_dot_products=dotproducts,
          math='Per atomic HPO: max cosine over public name/EXACT aliases. Compound: min of per-atom maxima. NIL/no aliases: observed=false, cosine numeric placeholder0. No full-bank nearest retrieval, no new span/ID.',no_labels=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','encode','cosine']);args=ap.parse_args()
    start=time.monotonic();plan=read('plan.json');assert sha(__file__)==plan['vector_source_sha256']
    globals()[args.stage](plan)
    write(args.stage+'-complete.json',dict(stage=args.stage,seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),plan_sha256=sha('plan.json')))
