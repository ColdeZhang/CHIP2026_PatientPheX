from pathlib import Path
import sys,json,time,hashlib,os,traceback,datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'reuse'))
from strict import HPO
from semantic import PublicConceptBank,PublicSapEncoder
D=Path(__file__).parent;plan=json.loads((D/'plan.json').read_text());root=Path(plan['remote_root']);out=root/'actual';out.mkdir(exist_ok=False)
def save(name,value):
 with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic()
try:
 for n,h in plan['source_files'].items():assert sha(D/'reuse'/n)==h
 a=plan['assets']
 for p,r in a['files'].items():assert Path(p).stat().st_size==r['bytes'] and sha(p)==r['sha256']
 q=D/'queries.json';assert sha(q)==plan['query_sha256'];quotes=json.loads(q.read_text());assert len(quotes)==1034
 save('started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':4347,'queries':1034,'no_held_labels_or_fit':True})
 hpo=HPO(a['hpo']);bank=PublicConceptBank(hpo,a['bank'],a['labels'],a['bank_manifest']);assert len(bank.rows)==41099
 enc=PublicSapEncoder(a['model_dir'],'cuda:0');assert enc.manifest['parameters']==109482240 and enc.manifest['transformers']=='4.51.3'
 save('encoder.json',enc.manifest)
 vectors,counts=enc.encode(quotes);np.save(out/'vectors.npy',vectors)
 results=[]
 for i,(quote,vec,n) in enumerate(zip(quotes,vectors,counts)):
  candidates=bank.rank(vec,top_k=8) if n<=32 else []
  assert len(candidates)==(8 if n<=32 else 0) and len({r['identifier'] for r in candidates})==len(candidates)
  results.append({'quote':quote,'token_count':n,'candidates':candidates,'status':'TOP8_PUBLIC_DISTINCT_HPO' if n<=32 else 'UNRETRIEVABLE_NO_TRUNCATION'})
 save('results.json',results)
 save('complete.json',{'status':'ALL_MISSING_PUBLIC_TOP8_QUERIES_COMPLETE','queries':len(results),'source_rows_accounted':4347,'long_unique_queries':sum(n>32 for n in counts),'max_tokens':max(counts),'results_sha256':sha(out/'results.json'),'vectors_sha256':sha(out/'vectors.npy'),'vectors_bytes':(out/'vectors.npy').stat().st_size,'elapsed_seconds':time.monotonic()-start,'peak_allocated':enc.torch.cuda.max_memory_allocated(),'encoder':enc.manifest,'held_pure_queries_present':True,'held_gold_read':False,'no_llm_fit_or_quality':True})
except BaseException as exc:
 save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start});raise
