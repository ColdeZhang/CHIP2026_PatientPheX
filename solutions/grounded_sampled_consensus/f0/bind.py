"""Bind complete native47 to one same-model, same-policy full94 inference."""
import copy,hashlib,json,os
from pathlib import Path
T=Path(__file__).resolve().parent;S=T.parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/grounded-sampled-consensus-v1/infer-F0-A30-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def identity(x):return {k:x[k] for k in ['bytes','sha256']}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 decision=json.load(open(T/'decision.json'));assert decision['baseline_score']==.7281956289113009 and decision['fresh_rows']==94 and decision['budget_seconds']==3600
 c=json.load(open(D/'native-F0-cpu-v1/collection.json'));assert c['all_actual_SHA_verified'] and c['CPU_reported_success']
 for v in c['files'].values():assert ref(v['path'])==v
 n=json.load(open(D/'native-F0-cpu-v1/terminal/actual/result.json'));assert n['status']=='PASS_NATIVE_COMPLETE47_QUOTE_SAMPLING_INPUT' and n['rows']==47 and n['all47_local_native_actual_IDs_rendering_decode_exact'] and n['all94_distinct_row_seeds'] and n['input_tokens']==289420 and n['max_input_tokens']==9935
 old=json.load(open(S/'a30/infer-plan.json'));p=copy.deepcopy(old);sources=copy.deepcopy(old['payload_sources'])
 for name in ['run.py','raw_store.py','sampling.py','infer_driver.py']:sources[name]=ref(T/name)
 for name in ['held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json']:sources[name]=decision['input_refs'][name]
 sources['native-CPU-result.json']=ref(D/'native-F0-cpu-v1/terminal/actual/result.json')
 for v in sources.values():assert ref(v['path'])==v
 p.update(schema='grounded-sample94-inference-v1',expected_rows=94,patients_per_seed=47,authorization='Root positiveF1 fixedF0 complete94 native sampling3600+TERM30; noGPUprobe, noselective retries, nootherarm',assets={k:dict(path=ROOT+'/'+Path(v['path']).name,**identity(sources[Path(v['path']).name])) for k,v in old['assets'].items()},payload_sources=sources,files={ROOT+'/'+k:identity(v) for k,v in sources.items()},output_dir=ROOT+'/output',argv=[old['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'],operations_source=ref(T/'remote.py'),collection_source=ref(T/'collect.py'),binding_source=ref(__file__))
 p['input_sources'].update(native_CPU_collection=ref(D/'native-F0-cpu-v1/collection.json'),original_F1_sample_inference=ref(S/'a30/infer-plan.json'),F1_result=ref(D/'quality-F1-v1/result.json'),F0_decision=ref(T/'decision.json'),F0_CPU_plan=ref(T/'cpu-plan.json'),F0_quality_template=ref(S/'postprocess_F0/plan-template.json'))
 p['budget_basis']={**decision['cost_proxy'],'cap_seconds':3600,'grace_seconds':30,'root_approved_before_GPUstart':True}
 p['reuse_packages']=[];p['reuse_actual_artifacts']={}
 assert p['generation']==old['generation'] and p['seeds']==old['seeds'] and p['presence_penalty']==0 and len(sources)==19
 save(T/'infer-plan.json',p);print(json.dumps(ref(T/'infer-plan.json')))
if __name__=='__main__':main()
