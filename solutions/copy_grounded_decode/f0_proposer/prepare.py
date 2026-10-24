"""Bind only input-only F0 proposer jobs; no model, gold or remote actions."""
import argparse,hashlib,json,os,sys
from pathlib import Path
S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/copy-grounded-decode-v1/F0-proposer-v1';P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
OLD=R/'research/gliner25-task1-v1';G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/normalization')
INPUT=Path('/workspace/patientphex-data/round2-inner4/fold-0/validation16.input.jsonl');TRAIN=R/'ontology_rules/nested-outer0-inner1/train32.jsonl';HPO=Path('/workspace/patientphex-data/hp.obo');HELD=R/'stage8-schema-extraction/f0-prepared/held.jsonl'
def ref(p):p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(v,h,ensure_ascii=False,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 q=p.parent
 while True:
  fd=os.open(q,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if q.parent==q:break
  q=q.parent

def checked(v):assert ref(v['path'])==v;return Path(v['path'])
def gold_guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and '.gold.' in os.fsdecode(args[0]):raise PermissionError('No held gold in proposer preparation')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['infer','normalize']);a=ap.parse_args();sys.addaudithook(gold_guard)
 assert ref(INPUT)['sha256']=='ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37'
 assert ref(HELD)['sha256']=='17f9607e5a76753e4147f0a8978c889bf3700dfe92e23c4256e17aa2b1c10d72'
 assert ref(TRAIN)['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
 docs=[json.loads(x) for x in INPUT.read_text().splitlines()];assert len(docs)==16 and sum(len(d['patient']) for d in docs)==47
 tids={str(json.loads(x)['pmc_id']) for x in TRAIN.read_text().splitlines()};assert len(tids)==32 and not tids&{str(d['pmc_id']) for d in docs}
 from coordinates import build,contract
 rebuilt=b''.join((json.dumps(build(d),ensure_ascii=False)+'\n').encode() for d in docs);assert rebuilt==HELD.read_bytes();contract()
 root=P+'/copy-grounded-decode-v1/F0-proposer-v1/'+('infer-v1' if a.phase=='infer' else 'normalize-v1');cap=600 if a.phase=='infer' else 180
 sources={n:ref(S/n) for n in ['job_driver.py','raw_store.py','evidence_io.py']};external={};extra={}
 sources['held.jsonl']=ref(HELD)
 host=read(S.parent/'normalization100/target-assets.json');env=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=host['gpu_uuid'],PATIENTPHEX_DURABILITY_ROOT=P)
 common=dict(schema='copy-grounded-F0-proposer-job-v1',phase=a.phase,authorized=True,authorization='Root explicitly authorized one complete same GLiNER2.5 final10 F0 proposer600+30 then unchanged acceptance with necessaryFP16Sap180+30; no fit or score',budget_seconds=cap,authorized_budget_seconds=cap,grace_seconds=30,complete_all_inputs=True,expected_documents=16,expected_patients=47,remote_root=root,output_dir=root+'/output',python=host['python'],storage_parent=P,gpu_uuid=host['gpu_uuid'],operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),binding_source=ref(__file__),reuse_packages=[],reuse_actual_artifacts={},input_sources=dict(input=ref(INPUT),prepared=ref(HELD),train32=ref(TRAIN),CPU_lineage=ref(S/'lineage.json')))
 if a.phase=='infer':
  sources.update({n:ref(S/n) for n in ['infer.py','coordinates.py']});sources['training-complete.json']=ref(OLD/'formal-v1-collected/actual/complete.json')
  state=read(OLD/'formal-v1-collected/actual/native/final/state-manifest.json');model=P+'/research/gliner25-task1-v1/formal-v1/actual/native/final'
  external.update({model+'/'+n:v for n,v in state['files'].items() if n!='training-state.pt'})
  old=read(OLD/'inference-start-intent.json');env.update(old['arguments']['env']);env['PATIENTPHEX_DURABILITY_ROOT']=P;env['PYTHONPATH']=root+':'+env['PYTHONPATH']
  argv=[host['python'],'-u',root+'/infer.py','--model',model,'--published-tokenizer',model+'/tokenizer.json','--input',root+'/held.jsonl','--training-complete',root+'/training-complete.json','--output',root+'/output']
  common.update(expected_status='COMPLETE_F0_NATIVE_TASK1_EXTRACTION',runner=sources['infer.py'],expected_windows=172,model_sha256=state['files']['model.safetensors']['sha256'],cost_proxy_seconds=23.94204335194081*172/164,predecessor_release=dict(confirmed_by_owner=True,source=ref(R/'research/copy-grounded-decode-v1/fallback-v2/Sap100-v1/terminal/wrapper-complete.json')))
 else:
  col=read(D/'infer-v1/collection.json');assert col['job_reported_success'] and col['all_actual_SHA_verified'];files=col['files']
  complete=read(checked(files['output/complete.json']));assert complete['status']=='COMPLETE_F0_NATIVE_TASK1_EXTRACTION' and complete['windows']==172 and complete['documents']==16 and complete['patients']==47
  selected=read(checked(files['output/selected.json']));assert ref(files['output/selected.json']['path'])['sha256']==complete['selected_sha256']
  sys.path.insert(0,str(G));from strict import StrictNormalizer,verify_span;from semantic import load_policy,precheck
  strict=StrictNormalizer(str(HPO),str(TRAIN));strict.assert_disjoint_input(str(INPUT));policy=load_policy(G/'policy_v1.json');by={str(d['pmc_id']):d for d in docs};spans={}
  for row in selected:
   if row['rejection'] is not None:continue
   s=row['source'];verify_span(by[row['pmc_id']],s['text'],s['offset']);key=(row['pmc_id'],s['offset'],s['length'],s['text'])
   if key not in spans or row['confidence']>spans[key]['confidence']:spans[key]=row
  quotes=sorted({k[3] for k in spans});norm=[strict.normalize(q) for q in quotes];pending=sorted({x['quote'] for x in norm if x['status']=='abstain' and x['reason']==policy['semantic_fallback_reason'] and not precheck(x['quote'],policy)})
  preparation=dict(input=ref(INPUT),extraction_collection=ref(D/'infer-v1/collection.json'),unique_spans=len(spans),unique_quotes=len(quotes),necessary_queries=len(pending),policy=ref(G/'policy_v1.json'),train32=ref(TRAIN),no_gold_fit_GPU_quality=True)
  save(D/'normalization-preparation.json',preparation)
  sources.update({n:ref(S/n) for n in ['normalize_entities.py','normalization-refs.json']});sources['input.jsonl']=ref(INPUT);sources['train32.jsonl']=ref(TRAIN)
  sources.update({'normalizer/'+n:ref(G/n) for n in ['strict.py','semantic.py','public_bank.py','policy_v1.json']});sources['extraction/complete.json']=files['output/complete.json'];sources['extraction/selected.json']=files['output/selected.json']
  external.update(host['files']);env.update(PYTHONPATH=root+':'+host['pythonpath'],OMP_NUM_THREADS='1')
  argv=[host['python'],'-u',root+'/normalize_entities.py','--extraction',root+'/extraction','--normalizer-source',root+'/normalizer','--references',root+'/normalization-refs.json','--input',root+'/input.jsonl','--train',root+'/train32.jsonl','--hpo',host['hpo'],'--bank',str(Path(host['bank']).parent),'--model',host['model_dir'],'--output',root+'/output','--device','cuda']
  common.update(expected_status='COMPLETE_FIXED_TASK1_NORMALIZATION',runner=sources['normalize_entities.py'],necessary_query_count=len(pending),cost_proxy_seconds=20+.02*len(pending),predecessor_release=dict(confirmed_by_owner=True,source=files['wrapper-complete.json']))
  common['input_sources'].update(extraction_collection=ref(D/'infer-v1/collection.json'),normalization_preparation=ref(D/'normalization-preparation.json'),host_assets=ref(S.parent/'normalization100/target-assets.json'))
  if not pending:
   # Original normalizer will never load/encode public model when pending is empty.
   localargv=[sys.executable,str(S/'normalize_entities.py'),'--extraction',str(D/'infer-v1/terminal/output'),'--normalizer-source',str(G),'--references',str(S/'normalization-refs.json'),'--input',str(INPUT),'--train',str(TRAIN),'--hpo',str(HPO),'--bank','UNUSED','--model','UNUSED','--output',str(D/'accepted-cpu'),'--device','cpu']
   save(S/'normalize-cpu-zero-query-plan.json',dict(no_GPU_required=True,argv=localargv,environment=dict(CUDA_VISIBLE_DEVICES='',PATIENTPHEX_DURABILITY_ROOT=str(D)),preparation=ref(D/'normalization-preparation.json')));print('ZERO_QUERIES_CPU_ONLY');return
  assert common['cost_proxy_seconds']<=180
 common.update(environment=env,argv=argv,payload_sources=sources,files={**external,**{root+'/'+k:{n:v[n] for n in ['bytes','sha256']} for k,v in sources.items()}})
 save(S/(a.phase+'-plan.json'),common);print(json.dumps(ref(S/(a.phase+'-plan.json'))))
if __name__=='__main__':main()
