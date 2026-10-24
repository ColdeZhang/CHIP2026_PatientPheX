"""Bind complete necessary held queries to unchanged public Sap assets and 180+30."""
import argparse,hashlib,json,os
from pathlib import Path
S=Path(__file__).resolve().parent
N=S.parent/'normalization'
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
OLD=PROJECT+'/grounded-generation-v1/normalization-v1/query-v1'
ROOT=PROJECT+'/grounded-generation-full64-v1/normalization-dev36-v1'

def ref(p):p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preparation-manifest',required=True);ap.add_argument('--release-evidence',required=True);a=ap.parse_args();prep=json.loads(Path(a.preparation_manifest).read_text())
 assert prep['schema']=='grounded-held-normalization-preparation-v1' and prep['no_gold_fit_or_score'] and prep['documents']==16 and prep['patients']==36
 queries=Path(prep['queries']['path']);assert ref(queries)==prep['queries'];n=len(json.loads(queries.read_text()));assert n==prep['necessary_public_query_count']
 if n==0:print(json.dumps(dict(no_GPU_query_required=True,preparation=ref(a.preparation_manifest))));return
 # Conservative measured proxy: prior complete624 took15.35s including loading.
 # Use 20s fixed + .02s per query; never silently truncate a larger set.
 estimate=20+.02*n;assert estimate<=180,'Report larger resource estimate to root; no truncated query subset'
 oldplan=json.loads((N/'evidence/query-plan.json').read_text());old=B/'normalization-v1/deployment';collection=json.loads((old/'collection.json').read_text());assert collection['all_actual_SHA_verified']
 actual=old/'terminal/actual';runtime=json.loads((actual/'encoder-runtime.json').read_text());complete=json.loads((actual/'complete.json').read_text());assert complete['status']=='COMPLETE_FIXED_PUBLIC_TRAIN_QUERIES' and complete['public_bank_rows']==41099 and complete['query_count']==624
 inputs={name:ref(N/name) for name in ['strict.py','semantic.py','public_bank.py','policy_v1.json','held_query.py','held_driver.py']};inputs['queries.json']=ref(queries)
 assert inputs['policy_v1.json']['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
 files={ROOT+'/'+k:ident(v) for k,v in inputs.items()}
 for path,item in oldplan['files'].items():
  if path.startswith(oldplan['model_dir']+'/') or path==oldplan['hpo']:files[path]=item
 for name in ['public-bank.npy','public-labels.json','public-bank-manifest.json']:files[OLD+'/actual/'+name]=complete['assets'][name]
 reuse={}
 for name in ['strict.py','semantic.py','public_bank.py','policy_v1.json']:
  assert inputs[name]['sha256']==oldplan['files'][OLD+'/'+name]['sha256'];reuse[name]=dict(path=OLD+'/'+name,original_root=OLD,original_member=name,**ident(inputs[name]),actual_collection=ref(old/'collection.json'))
 env=dict(PYTHONPATH=ROOT,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=oldplan['gpu_uuid'],PATIENTPHEX_DURABILITY_ROOT=PROJECT)
 plan=dict(schema='grounded-fixed-held-public-queries-v1',authorized=True,authorization='Parent /root approved all necessary originaldev36 fixed-policy queries <=180+TERM30 after actual coordinated release; train64 aliases, no new fit or bank encoding',budget_seconds=180,authorized_budget_seconds=180,grace_seconds=30,query_count=n,complete_necessary_queries=True,cost_estimate_seconds=estimate,cost_basis='Prior624=15.3473s; conservative20s+.02s/query, actual deadline authoritative',remote_root=ROOT,output_dir=ROOT+'/output',python=oldplan['python'],environment=env,gpu_uuid=oldplan['gpu_uuid'],storage_parent=PROJECT,files=files,payload_sources=inputs,reuse_packages=[],reuse_actual_artifacts=reuse,operations_source=ref(S/'held_remote.py'),collection_source=ref(S/'held_collect.py'),binding_source=ref(__file__),input_sources=dict(preparation=ref(a.preparation_manifest),original_public_bank_collection=ref(old/'collection.json'),release_evidence=ref(a.release_evidence),fixed_policy=inputs['policy_v1.json']),queries=ROOT+'/queries.json',policy=ROOT+'/policy_v1.json',hpo=oldplan['hpo'],bank=OLD+'/actual/public-bank.npy',labels=OLD+'/actual/public-labels.json',bank_manifest=OLD+'/actual/public-bank-manifest.json',model_dir=oldplan['model_dir'],expected_encoder_runtime={k:runtime[k] for k in ['torch','transformers','parameters','pooling','max_length']},argv=[oldplan['python'],'-u',ROOT+'/held_query.py','--plan',ROOT+'/held-query-plan.json'])
 path=S/'held-query-plan.json';assert not path.exists()
 with path.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(ref(path)))
if __name__=='__main__':main()
