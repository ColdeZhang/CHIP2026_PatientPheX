#!/usr/bin/env python3
"""Bind only actual complete public212 epoch2 to the original62 F1 prompts."""
from pathlib import Path
import argparse,json,hashlib,shutil,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--source-directory',type=Path,required=True);a=p.parse_args();d=a.directory.resolve();b=d.parent;train=d/'formal2ep';out=d/'F1-new62';assert not out.exists()
 terminal=read(train/'training_terminal.json');complete=read(train/'training_complete.json');cp=read(train/'epoch-002/checkpoint.json');actual=read(train/'checkpoint-actual-hashes.json');manifest=read(train/'run_manifest.json');policy=read(d/'comparison-policy.json')
 assert terminal['status']=='complete_epoch002_54' and terminal['complete_sha256']==sha(train/'training_complete.json') and terminal['checkpoint_sha256']==sha(train/'epoch-002/checkpoint.json')
 assert complete['epochs']==2 and complete['global_step']==54 and complete['elapsed_seconds']<2400 and not complete['smoke_only']
 assert cp['completed_epochs']==2 and cp['global_step']==54 and not cp['partial_not_for_inference'] and cp['final_fixed_epoch_only']
 assert cp['files']==actual['files'] and cp['file_sizes']==actual['file_sizes'] and cp['fingerprint']==manifest['fingerprint']
 assert manifest['examples']==212 and manifest['total_tokens']==651512 and manifest['warmup_steps']==3 and complete['source_sha256']==manifest['fingerprint']['source_sha256']
 trplan=read(train/'run-plan.json');assert sha(train/'run-plan.json')=='910fc8a37769c36ee76bbab7fdeb72d73de6dc35e6aa86040eb2bf641808da99'
 fp=manifest['fingerprint'];assert fp['source_sha256']==complete['source_sha256']==trplan['trainer_sha256']
 for field,name in [('train_sha256','train.jsonl'),('excluded_input_sha256','excluded.input.jsonl'),('public212_preparation_sha256','preparation-manifest.json'),('input_only_freeze_sha256','input-only-freeze.json')]:
  assert fp[field]==trplan['references'][name]['sha256']
 assert sha(d/'comparison-policy.json')==trplan['references']['comparison_policy']['sha256']=='7e48e32139b51afd39246cace0215576275223189db45b8f9ceedafc2914d064'
 prior=read(b/'public-new-relations/run-plan.json');out.mkdir();shutil.copy2(b/'public-new-relations/prompts.jsonl',out/'prompts.jsonl');shutil.copy2(b/'public-new-relations/token-audit.json',out/'token-audit.json')
 ops=a.source_directory.resolve();source=out/'source';source.mkdir();shutil.copy2(ops/'evaluate_f1.py',source/'evaluate_f1.py');shutil.copy2(__file__,source/'prepare_f1.py')
 driver=ops.parent/'public_relation_append/run_once.py';assert sha(driver)=='047b82723d1ddc97ffc4d9faaf491af172263a7bc0c0615d79afb1d0d1ba9782';shutil.copy2(driver,source/'run_once.py')
 plan=prior.copy();r={k:v for k,v in prior['references'].items() if k not in ['checkpoint','actual_checkpoint_hashes','training_complete']}
 r.update(training_run_plan=ref(train/'run-plan.json'),prompts=ref(out/'prompts.jsonl'),token_audit=ref(out/'token-audit.json'),checkpoint=ref(train/'epoch-002/checkpoint.json'),actual_checkpoint_hashes=ref(train/'checkpoint-actual-hashes.json'),training_complete=ref(train/'training_complete.json'),training_manifest=ref(train/'run_manifest.json'),training_terminal=ref(train/'training_terminal.json'),comparison_policy=ref(d/'comparison-policy.json'),old_baseline_pred=ref(b/'public-new-relations/baseline.pred.jsonl'),old_supported_pred=ref(b/'public-new-relations/support-cut03-control/supported.pred.jsonl'),old_supported_metrics=ref(b/'public-new-relations/support-cut03-control/metrics.json'))
 oldsupport=read(b/'public-new-relations/support-cut03-control/prediction-freeze.json');r['support_helper']=oldsupport['helper'];r['support_prediction']=oldsupport['support']
 for x in r.values():assert ref(x['path'])==x
 adapter={Path(k).name:v for k,v in cp['files'].items() if k in ['adapter/adapter_config.json','adapter/adapter_model.safetensors']};assert len(adapter)==2
 root='sft-meta1-fusion/public-training-candidate-preparation';remote=root+'/F1-new62';adapterpath=root+'/formal2ep/epoch-002/adapter'
 argv=list(prior['argv']);argv[argv.index('--adapter')+1]=adapterpath;argv[argv.index('--output')+1]=remote+'/responses.jsonl'
 expected={k:v for k,v in prior['remote_sources_sha256'].items() if '/formal2ep/epoch-002/adapter/' not in k}
 expected.update({adapterpath+'/'+k:v for k,v in adapter.items()})
 plan.update(schema='public212-fresh-training-F1-new62-supported-only-v1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),references=r,control_sha256=sha(source/'evaluate_f1.py'),driver_sha256=sha(source/'run_once.py'),adapter_files=adapter,remote_root=remote,argv=argv,remote_sources_sha256=expected,fixed_control=dict(prior['fixed_control'],training=True,training_rows=212,training_epochs=2,training_calls=54,output_policy='Only support-filtered new relations; keep old80/383 and all1189entities'),boundaries=['Unique new supported arm; raw unfiltered not scored','Same fixed original62 prompts/decoder/policy, fresh212 epoch2','Both old cached/new full prediction refs freeze before unique newF1 quality','No originaldev/F0/gate/threshold search; pooled64 support calibration boundary remains'])
 with (out/'run-plan.json').open('x') as f:f.write(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
 with (out/'baseline-prepared.json').open('x') as f:f.write(json.dumps(dict(status='old_baseline_and_supported_cache_exact_no_new_quality',old_baseline=r['old_baseline_pred'],old_supported=r['old_supported_pred'],cached_metrics=r['old_supported_metrics'],plan=ref(out/'run-plan.json')),indent=2)+'\n')
 print(json.dumps(dict(plan=ref(out/'run-plan.json'),new_adapter=adapter,scope='Prepared only, no launch/quality'),indent=2))
if __name__=='__main__':main()
