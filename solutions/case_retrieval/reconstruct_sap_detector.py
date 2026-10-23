#!/usr/bin/env python3
"""CPU-only exact recipe reconstruction using existing immutable embedding files."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, importlib.util, json, gzip, platform, time
from pathlib import Path
import numpy as np
import sklearn


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for k in ['source','train','input','hpo','model-dir','frozen-dir','output-dir']:
        ap.add_argument('--'+k,required=True)
    a=ap.parse_args();started=time.monotonic();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((Path(a.frozen_dir)/'manifest.json').read_text())
    assert sha(a.source)==old['semantic_source_sha256']
    for k in ['train','input','hpo']:assert sha(getattr(a,k))==old['inputs'][k],k
    candidate_file=Path(a.frozen_dir)/'scored-candidates.jsonl.gz'
    assert sha(candidate_file)==old['output']['sha256']
    spec=importlib.util.spec_from_file_location('frozen_semantic',a.source)
    sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    train=sem.read_jsonl(a.train);inputs=sem.read_jsonl(a.input)
    assert len(train)==48==old['training']['training_docs']
    assert not {d['pmc_id'] for d in train}&{d['pmc_id'] for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    model_files={}
    for name,record in old['model_manifest']['files'].items():
        expected=old['model_manifest']['derived_weights']['sha256'] if name=='model.safetensors' else record['sha256']
        actual=sha(Path(a.model_dir)/name);assert actual==expected,name
        model_files[name]=actual
    signature='sapbert'+sha(Path(a.model_dir)/'config.json')+sha(Path(a.model_dir)/'manifest.json')
    class CacheOnlyEncoder:
        def __init__(self):self.reads=[]
        def encode(self,texts,label):
            names=sorted(set(texts));lookup={s:i for i,s in enumerate(names)}
            digest=hashlib.sha256((signature+'\0'+'\0'.join(names)).encode()).hexdigest()
            file=Path(a.frozen_dir)/'embeddings'/(digest+'.npy')
            if not file.exists():raise FileNotFoundError('Cache miss; encoding forbidden: '+label+' '+digest)
            values=np.load(file,allow_pickle=False)
            assert values.shape==(len(names),768) and values.dtype==np.float32
            self.reads.append({'label':label,'path':str(file),'sha256':sha(file),'rows':len(texts),'unique':len(names),'ordered_texts_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()})
            return values[np.asarray([lookup[t] for t in texts])]
    encoder=CacheOnlyEncoder();terms,canonical=sem.hpo_terms(a.hpo)
    config=argparse.Namespace(seed=20261023,max_candidates=old['config']['max_candidates'],negatives_per_doc=500)
    system=sem.System(train,terms,canonical,encoder,config)
    assert system.training_stats==old['training'],(system.training_stats,old['training'])
    with gzip.open(candidate_file,'rt') as f:reference={str(d['pmc_id']):d for d in map(json.loads,f)}
    report={'candidate_count':0,'max_abs_probability':0.,'max_abs_similarity':0.,'probability_decision_mismatches':0,'similarity_decision_mismatches':0,'joint_decision_mismatches':0,'nearest_id_mismatches':0,'documents':[]}
    for doc in inputs:
        pmc=str(doc['pmc_id']);cs=system.candidates(doc);oldcs=reference[pmc]['candidates']
        assert len(cs)==len(oldcs)
        assert all(all(c[k]==oldc[k] for k in c) for c,oldc in zip(cs,oldcs)),'Candidate reconstruction differs '+pmc
        vectors=encoder.encode([sem.norm(c['text']) for c in cs],'candidates-'+pmc)
        probs=system.detector.predict_proba(vectors)[:,1]
        sims=[];indices=[]
        for start in range(0,len(vectors),128):
            dots=vectors[start:start+128]@system.bank.T
            idx=dots.argmax(1);indices.extend(idx.tolist());sims.extend(dots[np.arange(len(idx)),idx].tolist())
        probs=np.asarray(probs);sims=np.asarray(sims)
        oldp=np.asarray([c['probability'] for c in oldcs]);olds=np.asarray([c['similarity'] for c in oldcs])
        dp=float(np.max(np.abs(probs-oldp),initial=0));ds=float(np.max(np.abs(sims-olds),initial=0))
        pbad=int(np.count_nonzero((probs>=.65)!=(oldp>=.65)));sbad=int(np.count_nonzero((sims>=.92)!=(olds>=.92)))
        jointbad=int(np.count_nonzero(((probs>=.65)&(sims>=.92))!=((oldp>=.65)&(olds>=.92))))
        idbad=sum((system.best_id(sem.norm(c['text'])) if c['exact'] else system.label_ids[idx])!=oc['identifier'] for c,idx,oc in zip(cs,indices,oldcs))
        row={'pmc_id':pmc,'candidates':len(cs),'max_abs_probability':dp,'max_abs_similarity':ds,'probability_decision_mismatches':pbad,'similarity_decision_mismatches':sbad,'joint_decision_mismatches':jointbad,'nearest_id_mismatches':idbad}
        report['documents'].append(row);report['candidate_count']+=len(cs)
        report['max_abs_probability']=max(report['max_abs_probability'],dp);report['max_abs_similarity']=max(report['max_abs_similarity'],ds)
        for k in ['probability_decision_mismatches','similarity_decision_mismatches','joint_decision_mismatches','nearest_id_mismatches']:report[k]+=row[k]
        print(json.dumps({'event':'replayed',**row}),flush=True)
    passed=(report['max_abs_probability']<=1e-8 and report['max_abs_similarity']<=2e-6 and report['probability_decision_mismatches']==report['similarity_decision_mismatches']==report['joint_decision_mismatches']==0)
    target=out/'reconstructed-detector.npz'
    np.savez(target,coef=system.detector.coef_,intercept=system.detector.intercept_,classes=system.detector.classes_)
    bank=encoder.reads[0]
    result={'passed':bool(passed),'method':'exact frozen training recipe replay from existing embeddings; classifier was not originally serialized','inputs':old['inputs'],'reference_candidate_sha256':sha(candidate_file),'reference_manifest_sha256':sha(Path(a.frozen_dir)/'manifest.json'),'source_sha256':sha(__file__),'semantic_source_sha256':sha(a.source),'training_ids':sorted(str(d['pmc_id']) for d in train),'input_ids':sorted(str(d['pmc_id']) for d in inputs),'configuration':vars(config),'training':system.training_stats,'encoder_signature':signature,'model_manifest':old['model_manifest'],'actual_model_files_sha256':model_files,'bank_cache':bank,'cache_reads':encoder.reads,'detector':{'path':str(target),'sha256':sha(target)},'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'threads':1,'cuda_visible_devices':''},'replay':report,'acceptance':{'max_abs_probability_lte':1e-8,'max_abs_similarity_lte':2e-6,'decision_mismatches':0},'elapsed_seconds':time.monotonic()-started,'new_encoding_performed':False,'gold_validation_used':False}
    (out/'reconstruction.manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'finished','passed':passed,'candidate_count':report['candidate_count'],'max_abs_probability':report['max_abs_probability'],'max_abs_similarity':report['max_abs_similarity']}),flush=True)
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
