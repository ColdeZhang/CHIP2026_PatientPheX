"""Freeze original F1 inputs, one sampling recipe and the real native CPU gate."""
import gzip, hashlib, json, os
from pathlib import Path

S=Path(__file__).resolve().parent;T=S/'runtime'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
R=Path('/workspace/patientphex-runs/round2-20261023')
B=R/'research/grounded-generation-v1';D=R/'research/grounded-sampled-consensus-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';REMOTE=P+'/grounded-sampled-consensus-v1/native-cpu-v1'

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 d=p.parent.resolve()
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent

def main():
 original=json.loads((G/'inference/infer-plan.json').read_text())
 orig=original['payload_sources'];rows=[json.loads(x) for x in Path(orig['held-prompts.jsonl']['path']).read_text().splitlines()]
 audit=json.loads(Path(orig['held-audit.json']['path']).read_text());ids=json.loads(gzip.decompress(Path(orig['held-actual-ids.json.gz']['path']).read_bytes()))
 assert len(rows)==len(audit)==len(ids)==39 and len({r['pmc_id'] for r in rows})==16
 assert len({(r['pmc_id'],r['target_patient_id']) for r in rows})==39 and all('answer' not in r for r in rows)
 assert [r['row_id'] for r in rows]==[r['row_id'] for r in audit]==[r['row_id'] for r in ids]
 for r,a,i in zip(rows,audit,ids):
  assert len(i['input_ids'])==a['input_tokens'] and hashlib.sha256(json.dumps(i['input_ids']).encode()).hexdigest()==a['input_ids_sha256']
 assert sum(a['input_tokens'] for a in audit)==203588 and max(a['input_tokens'] for a in audit)==10080
 previous=json.loads((G/'f0/cpu-plan.json').read_text());oldroot=previous['output_dir'].rsplit('/',1)[0]
 def replace(v):
  if isinstance(v,str):return v.replace(oldroot,REMOTE)
  if isinstance(v,list):return [replace(x) for x in v]
  if isinstance(v,dict):return {replace(k):replace(x) for k,x in v.items()}
  return v
 p=replace(previous);p.update(schema='grounded-F1-quote-sampling-native39-CPU-v1',held_rows=39,operations_source=ref(T/'cpu_remote.py'),reuse_packages=[])
 inputs={'native_cpu.py':ref(T/'native_cpu.py'),'cpu_driver.py':ref(T/'cpu_driver.py'),'sampling.py':ref(T/'sampling.py'),'held.input.jsonl':orig['held-prompts.jsonl'],'held-audit.json':orig['held-audit.json'],'held-local-ids.json.gz':orig['held-actual-ids.json.gz'],'model-source-manifest.json':orig['model-source-manifest.json']}
 for v in inputs.values():assert ref(v['path'])==v
 p['payload_sources']=inputs;p['files']={REMOTE+'/'+k:ident(v) for k,v in inputs.items()}
 cache=json.loads((B/'F0-copy-v1/copy-complete.json').read_text());p['checkpoint_directory']=cache['ack']['destination'];p['adapter_files']=cache['ack']['files']
 prior=json.loads((B/'infer-F0-v1/terminal/output/run-manifest.json').read_text());p['eos_ids']=prior['eos_ids']
 # Original Qwen3.5 tokenizer/config identity, same padding rule as successful runner.
 p['pad_token_id']=248044
 p['input_sources']=dict(original_F1_plan=ref(G/'inference/infer-plan.json'),original_F1_native_collection=ref(B/'native-cpu-v2/collection.json'),same_adapter_copy=ref(B/'F0-copy-v1/copy-complete.json'),old_review=ref(R/'validation/grounded-F1-final-review/review.json'),decision=ref(S/'decision.json'))
 p['quality_policy']='Only B4 UNION(normalized sample1 INTERSECT normalized sample2); no single-seed v2; zero addition means exact base/zero v2; strict total Score gain'
 save(T/'cpu-plan.json',p)
 baseline=R/'research/four-source-fusion-20261024/F1-v1';f=json.loads((baseline/'freeze.json').read_text());assert f['final_relations']==524 and f['prediction']['sha256']=='5ad5cfe33ebea607c0012ee04420b31ff54cad06a52305091c54bb7d7ed7207e'
 c=json.loads((B/'infer-F0-v1/terminal/output/complete.json').read_text());raw=[json.loads(x) for x in (B/'infer-F0-v1/terminal/output/responses.jsonl').read_text().splitlines()];tokens=sum(r['output_tokens'] for r in raw)
 # Same adapter/native A800 F0 is a better runtime proxy than A100 F1 or full64.
 proxy=2*10513/tokens*c['seconds'];conservative=1.5*proxy+90
 assert conservative<1800
 save(D/'preparation.json',dict(schema='grounded-sampled-F1-readiness-v1',source=ref(__file__),cpu_plan=ref(T/'cpu-plan.json'),decision=ref(S/'decision.json'),inputs=inputs,rows39_twice=78,documents16=True,patients39=True,input_tokens_each_seed=203588,max_input_tokens=10080,max_new_tokens=1600,full_position_upper_bound=11680,original_full_input_unchanged=True,all78_fresh=True,seeds=[20261024,20261025],cost=dict(previous_same_d52_A800_F0_native_seconds=c['seconds'],previous_output_tokens=tokens,original_F1_greedy_output_tokens=10513,two_stream_token_proxy=proxy,conservative_1point5_plus90=conservative,cap_seconds=1800,grace_seconds=30,not_a_guarantee='Sampling may generate more or reach1600. Original shared absolute1800 remains fixed, incomplete means no quality/no selective retries.'),baseline=f['prediction'],baseline_freeze=ref(baseline/'freeze.json'),baseline_result=ref(baseline/'result.json'),adapter_cache=ref(B/'F0-copy-v1/copy-complete.json'),no_gold_body_opened=True,no_NN_or_fit=True))
 print(json.dumps(dict(preparation=ref(D/'preparation.json'),cpu_plan=ref(T/'cpu-plan.json'),seconds_proxy=proxy,conservative_seconds=conservative)))

if __name__=='__main__':main()
