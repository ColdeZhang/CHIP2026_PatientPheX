#!/usr/bin/env python3
"""Freeze, deploy and CPU-check one fresh weighted124 run; never launch GPU."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,sys,shlex

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True);a=p.parse_args();b=a.base.resolve();w=b/'json-value-weighted-cpu-preparation';out=w/'formal2ep';out.mkdir();src=Path(__file__).resolve().parent
 old=json.loads((w/'a100-probe-attempt2/run-plan.json').read_text());remote='case-retrieval/qwen35-preparation/json-value-weighted-formal2ep';prep=remote+'/preparation'
 mapping={n:src/n for n in ['train_weighted.py','loss.py','launch_formal.py','prepare_formal.py']}
 mapping.update({n:b/n for n in ['train.jsonl','excluded.input.jsonl','preparation-manifest.json','qwen35_smoke.py','qwen35_relation_cost.py','qwen35_runner.py']})
 mapping['weights.json']=w/'weights.json';mapping['model-source-manifest.json']=b.parent/'model-source-manifest.json'
 for n,path in mapping.items():shutil.copy2(path,out/n)
 policy=dict(hypothesis='Within identical old124 JSON responses, weight2 list-value tokens including empty lists can improve candidate patient selection',train='fresh strictF2+F3 base+r16,124rows,2epochs,32calls,2warmup; no initialization from prior adapter/probe',loss='Exact reviewed af262 row weighted mean then original actual-group mean8/4; all assistant JSON+EOS remain',selection='Only final complete epoch002/32; partial state is never inference-ready',F1=dict(prompts=ref(b/'meta.prompts.jsonl'),baseline=ref(b/'public-new-relations/baseline.pred.jsonl'),baseline_metrics=ref(b/'public-new-relations/baseline.metrics.json'),new_arm='Only same old80 prompts/587CID/1557pair; original runner/parser/greedy1600/no thinking; original entities/None assembly; fullfreeze before unique v2; no public62',inference_cap_seconds=600,grace_seconds=30),quality_gate='Report full v2 and delta to old SFT32; no new threshold/weight/epoch search, F0/dev or automatic migration',hardware_difference='Old SFT32A800 torch2.7.1cu126; new A100torch2.6cu124; samebase15/TF5.20.2/PEFT.21.2. Probe selected/full gradient matched predeclared2% tolerance, not bitwise.',probe_review=ref(b.parents[2]/'validation/json-value-weighted-probe-review/attempt2-review.json'))
 save(out/'comparison-policy.json',policy)
 sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r;r.M=old['mount_id']
 exists=r.call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(remote)+').exists()'),cwd='.',timeout_seconds=20));assert exists['exit_code']==0
 for n in mapping:r.upload(str(out/n),prep+'/'+n)
 r.upload(str(out/'comparison-policy.json'),prep+'/comparison-policy.json')
 model=old['argv'][old['argv'].index('--model')+1];python=old['argv'][4]
 args=[python,prep+'/train_weighted.py','--train',prep+'/train.jsonl','--excluded-input',prep+'/excluded.input.jsonl','--data-manifest',prep+'/preparation-manifest.json','--weights',prep+'/weights.json','--model',model,'--model-source-manifest',prep+'/model-source-manifest.json']
 cpuenv={**old['env'],'CUDA_VISIBLE_DEVICES':''};cpuargv=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','90']+args+['--mode','cpu','--output-dir',remote+'/cpu-contract']
 cpucmd='env '+shlex.join([k+'='+v for k,v in cpuenv.items()])+' '+shlex.join(cpuargv)
 actual=r.call('run_command',dict(command=cpucmd,cwd='.',timeout_seconds=105));save(out/'actual-cpu-command.json',dict(argv=cpuargv,env=cpuenv,result=actual));assert actual['exit_code']==0,actual
 cpu=out/'cpu-contract';cpu.mkdir()
 for n in ['cpu_complete.json','run_manifest.json','weighted-input-contract.json','token-mask-audit.json','meta-coverage.json','causal-loss-probe.json']:
  x=r.call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(remote+'/cpu-contract/'+n)+');print(p.read_text() if p.exists() else "MISSING")'),cwd='.',timeout_seconds=20))
  if x['stdout'].strip()!='MISSING':(cpu/n).write_text(x['stdout'].rstrip()+'\n')
 c=json.loads((cpu/'cpu_complete.json').read_text());assert c['source_sha256']==sha(out/'train_weighted.py') and c['rows']==124 and not c['cuda_initialized']
 refs={n:ref(out/n) for n in mapping};refs['comparison-policy.json']=ref(out/'comparison-policy.json');refs['actual_cpu']=ref(out/'actual-cpu-command.json');refs['cpu_complete']=ref(cpu/'cpu_complete.json')
 argv=['/usr/bin/timeout','--signal=TERM','--kill-after=30s','2100']+args+['--mode','train','--gpu-uuid',old['gpu_uuid'],'--output-dir',remote+'/run']
 plan=dict(schema='weighted124-fresh2ep32-A100-v1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mount_id=old['mount_id'],gpu_uuid=old['gpu_uuid'],cwd='.',remote_root=remote+'/run',launcher_sha256=sha(out/'launch_formal.py'),trainer_sha256=sha(out/'train_weighted.py'),references=refs,remote_sources_sha256={prep+'/'+n:sha(out/n) for n in list(mapping)+['comparison-policy.json']},minimum_free_disk_bytes=680000000,cap_seconds=2100,grace_seconds=30,soft_save_seconds=2040,predicted_next_group_guard='actual group tokens*.0023+8; leave60s before2100 for recoverable final/partial',storage='Final-only adapter+Adam/RNG approx520MB on persistent A100; no deleting/moving old artifacts',env=old['env'],argv=argv,phase='train',collect=['run_manifest.json','weighted-input-contract.json','token-mask-audit.json','model-coverage.json','base-loading-info.json','first-nonzero-update.json','updates.jsonl','training_terminal.json'],cost=dict(longest_actual_step_seconds=12.180502124,longest_tokens=5523,total_tokens_two_epochs=805188,token_linear_seconds=1775.772975616,estimate_not_measurement=True,load_hash_save_not_in_projection=True),comparison_policy_sha256=sha(out/'comparison-policy.json'))
 save(out/'run-plan.json',plan);print(json.dumps(dict(plan=ref(out/'run-plan.json'),cpu=c),indent=2))
if __name__=='__main__':main()
