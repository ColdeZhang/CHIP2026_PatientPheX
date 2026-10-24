#!/usr/bin/env python3
"""Bind complete weighted124 epoch2 to exactly original80 F1 requests; no GPU/labels."""
from pathlib import Path
import argparse,json,hashlib,shutil,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=a.directory.resolve();b=d.parent;train=d/'formal2ep';out=d/'F1-old80';assert not out.exists()
 trplan=read(train/'run-plan.json');terminal=read(train/'training_terminal.json');complete=read(train/'training_complete.json');cp=read(train/'epoch-002/checkpoint.json');actual=read(train/'checkpoint-actual-hashes.json');manifest=read(train/'run_manifest.json');policy=read(train/'comparison-policy.json')
 assert terminal['status']=='complete_epoch002_32' and terminal['complete_sha256']==sha(train/'training_complete.json') and terminal['checkpoint_sha256']==sha(train/'epoch-002/checkpoint.json')
 assert complete['epochs']==2 and complete['global_step']==32 and complete['elapsed_seconds']<2100 and not complete['smoke_only']
 assert cp['completed_epochs']==2 and cp['global_step']==32 and not cp['partial_not_for_inference'] and cp['final_fixed_epoch_only']
 assert cp['files']==actual['files'] and cp['file_sizes']==actual['file_sizes'] and cp['fingerprint']==manifest['fingerprint']
 assert manifest['examples']==124 and manifest['total_tokens']==402594 and manifest['supervised_tokens']==7170 and manifest['warmup_steps']==2
 fp=manifest['fingerprint'];assert fp['source_sha256']==complete['source_sha256']==trplan['trainer_sha256']==sha(train/'train_weighted.py')
 for field,name in [('train_sha256','train.jsonl'),('excluded_input_sha256','excluded.input.jsonl'),('data_preparation_sha256','preparation-manifest.json'),('weights_sha256','weights.json'),('loss_source_sha256','loss.py')]:assert fp[field]==trplan['references'][name]['sha256']==sha(train/name)
 assert sha(train/'comparison-policy.json')==trplan['comparison_policy_sha256']==trplan['references']['comparison-policy.json']['sha256']
 prior=read(b/'public-new-relations/run-plan.json');out.mkdir();shutil.copy2(b/'meta.prompts.jsonl',out/'prompts.jsonl');assert ref(out/'prompts.jsonl')['sha256']==policy['F1']['prompts']['sha256']=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
 ops=Path(__file__).resolve().parent;source=out/'source';source.mkdir()
 for n in ['prepare_f1.py','evaluate_f1.py','infer80.py']:shutil.copy2(ops/n,source/n)
 r={k:prior['references'][k] for k in ['input','old_entities','old_raw','old_manifest','old_raw_freeze','model_source','tree_helper','fusion_helper','parser','parser_base','hpo','evaluator','validator']}
 r.update(training_run_plan=ref(train/'run-plan.json'),prompts=ref(out/'prompts.jsonl'),checkpoint=ref(train/'epoch-002/checkpoint.json'),actual_checkpoint_hashes=ref(train/'checkpoint-actual-hashes.json'),training_complete=ref(train/'training_complete.json'),training_manifest=ref(train/'run_manifest.json'),training_terminal=ref(train/'training_terminal.json'),comparison_policy=ref(train/'comparison-policy.json'),old_baseline_pred=policy['F1']['baseline'],old_baseline_metrics=policy['F1']['baseline_metrics'],runner=ref(train/'qwen35_runner.py'))
 for x in r.values():assert ref(x['path'])==x
 oldraw=[json.loads(x) for x in Path(r['old_raw']['path']).read_text().splitlines()];prompts=[json.loads(x) for x in (out/'prompts.jsonl').read_text().splitlines()];assert len(prompts)==len(oldraw)==80
 audit=dict(prompts=80,concepts=587,patient_concept_pairs=1557,rows=[dict(row_id=x['row_id'],input_tokens=x['input_tokens']) for x in oldraw],total_tokens=sum(x['input_tokens'] for x in oldraw),max_tokens=max(x['input_tokens'] for x in oldraw),source='Reuse actual original same80 prompt/template/tokenizer counts; original manifest/input SHA frozen',old_raw=r['old_raw'],old_manifest=r['old_manifest'],truncated=0)
 assert audit['max_tokens']<14000;save(out/'token-audit.json',audit);r['token_audit']=ref(out/'token-audit.json')
 adapter={Path(k).name:v for k,v in cp['files'].items() if k in ['adapter/adapter_config.json','adapter/adapter_model.safetensors']};assert len(adapter)==2
 remote=trplan['remote_root'].rsplit('/',1)[0]+'/F1-old80';prep=trplan['remote_root'].rsplit('/',1)[0]+'/preparation';adapterpath=actual['path']+'/adapter';python=trplan['argv'][4];model=trplan['argv'][trplan['argv'].index('--model')+1]
 argv=['/usr/bin/timeout','--signal=TERM','--kill-after=30s','600',python,'-u',prep+'/qwen35_runner.py','--model',model,'--source-manifest',prep+'/model-source-manifest.json','--prompts',remote+'/prompts.jsonl','--output',remote+'/responses.jsonl','--adapter',adapterpath,'--max-input-tokens','14000','--max-new-tokens','1600']
 expected={prep+'/qwen35_runner.py':r['runner']['sha256'],prep+'/model-source-manifest.json':r['model_source']['sha256'],remote+'/prompts.jsonl':sha(out/'prompts.jsonl'),**{adapterpath+'/'+k:v for k,v in adapter.items()}}
 plan=dict(schema='weighted124-original80-F1-only-v1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),references=r,control_sha256=sha(source/'evaluate_f1.py'),driver_sha256=sha(source/'infer80.py'),runner_sha256=r['runner']['sha256'],adapter_files=adapter,remote_root=remote,argv=argv,remote_sources_sha256=expected,mount_id=trplan['mount_id'],gpu_uuid=trplan['gpu_uuid'],cwd='.',env=trplan['env'],row_documents={x['row_id']:str(x['pmc_id']) for x in prompts},cap_seconds=600,grace_seconds=30,gold_path=prior['gold_path'],historical_F1_gold_sha256=prior['historical_F1_gold_sha256'],python=prior['python'],policy='One original80 fixed JSON/greedy/parser complete arm; cached old32 baseline; no public new concepts or support filter; complete prediction freeze before unique v2/format')
 save(out/'run-plan.json',plan);save(out/'baseline-prepared.json',dict(status='cached_original80_SFT32_reference_no_new_quality',baseline=r['old_baseline_pred'],metrics=r['old_baseline_metrics']));print(json.dumps(ref(out/'run-plan.json')))
if __name__=='__main__':main()
