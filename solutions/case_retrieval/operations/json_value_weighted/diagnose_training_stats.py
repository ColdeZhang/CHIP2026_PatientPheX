#!/usr/bin/env python3
"""Only saved training traces/checkpoints; no model load, gold or new inference."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def trace(p):
 out=[]
 for line in Path(p).read_text().splitlines():
  try:x=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(x,dict) and {'step','loss','learning_rate'}<=set(x):out.append(x)
 assert len({x['step'] for x in out})==len(out)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--qwen-root',type=Path,required=True);a=p.parse_args();q=a.qwen_root;b=q/'sft-meta1-fusion';w=b/'json-value-weighted-cpu-preparation/formal2ep';out=b/'json-value-weighted-cpu-preparation/training-stats-diagnostic';out.mkdir()
 specs=[('old32_JSON_CE',b/'formal2ep',b/'train.stdout.log',b/'epoch1-progress.json','progress'),('full64_JSON_CE',q/'full64-relation-cpu-preparation/formal2ep',q/'full64-relation-cpu-preparation/formal.stdout.log',q/'full64-relation-cpu-preparation/epoch1-observation.json','actual'),('value2_32_weighted_CE',w,w/'updates.jsonl',w/'epoch-001-summary.json',None)]
 arms={};sources={'source':ref(__file__),'existing_full64_fixed12_training_fit':ref(q/'full64-objective-diagnostic/fixed12-run/fit-result.json')} 
 for name,d,log,first,key in specs:
  manifest=read(d/'run_manifest.json');complete=read(d/'training_complete.json');cp=read(d/'epoch-002/checkpoint.json');step=int(complete['global_step']);warm=int(manifest['warmup_steps']);per=int(manifest['steps_per_epoch']);assert complete['epochs']==cp['completed_epochs']==2 and cp['global_step']==step==2*per
  v=trace(log);assert all(math.isfinite(x['loss']) and math.isfinite(x['learning_rate']) for x in v)
  lr=[1e-4*((j-1)/warm if j-1<warm else (step-(j-1))/(step-warm)) for j in range(1,step+1)]
  assert all(abs(x['learning_rate']-lr[x['step']-1])<1e-14 for x in v)
  e1=read(first);e1=e1[key] if key else e1;e2=read(d/('epoch-002-summary.json' if key is None else 'progress.json'));assert e1['global_step']==per and e2['global_step']==step
  firstupdate=read(d/'first-nonzero-update.json');assert firstupdate['step']==2 and firstupdate['learning_rate']==lr[1]
  epochs={str(e):dict(optimizer_calls=per,nonzero_learning_rate_calls=per-1 if e==1 else per,mean_actual_group_loss=[e1,e2][e-1]['mean_accumulation_group_loss'],recorded_step_losses=[x for x in v if x['epoch']==e]) for e in [1,2]}
  arms[name]=dict(examples=manifest['examples'],tokens=manifest['total_tokens'],supervised_tokens=manifest['supervised_tokens'],epochs=epochs,optimizer_calls=step,nonzero_lr_calls=step-1,warmup=warm,first_recorded_step=v[0]['step'],last_recorded_step=v[-1]['step'],first_recorded_loss=v[0]['loss'],last_recorded_loss=v[-1]['loss'],logged_loss_min=min(x['loss'] for x in v),logged_loss_max=max(x['loss'] for x in v),full_per_step_trace=(len(v)==step),logged_records=len(v),recorded_loss_increases=sum(y['loss']>x['loss'] for x,y in zip(v,v[1:])),lr_schedule_reconstructed_from_frozen_config_and_matching_saved_records=dict(first=lr[0],maximum=max(lr),last_optimizer_call=lr[-1],after_final_scheduler=0.,all_saved_lr_match=True),first_nonzero_update_actual=firstupdate,elapsed_seconds=complete['elapsed_seconds'],no_recorded_nonfinite=True,complete_finite_loss_gradient_guard_was_active=True)
  sources[name]={k:ref(vv) for k,vv in dict(manifest=d/'run_manifest.json',complete=d/'training_complete.json',checkpoint=d/'epoch-002/checkpoint.json',trace=log,epoch1=first,epoch2=d/('epoch-002-summary.json' if key is None else 'progress.json'),first_update=d/'first-nonzero-update.json').items()}
 result=dict(status='saved_training_trajectory_readonly_complete',sources=sources,arms=arms,interpretation=['Old32/full64 epoch averages decline, but they are online losses under changing weights and different shuffled groups, not fixed-final evaluation losses.','Sparse old traces start atstep5/end30 or65, not true beginning/end losses; only weighted32 has allstep records.','Warmup begins atzero and final scheduler reacheszero; 32/68 are calls,31/67 have nonzero learningrate. Reconstructed rates are distinguished from saved records.','No same-data original JSON recipe completed quality control with more than2epochs is present in this route; different binary/NER/identity tasks do not supply that control.','Full64 existing input-selected12 training generation fitF1 .883721/exact7of12 supports learning, but is a small training diagnostic, not full training loss or generalization.','Value2 weightedloss units differ from unweightedCE; neither its absolute size nor online decline demonstrates better generalization.','Current evidence does not establish obvious underfitting. Further epochs would require a new fixed schedule/control, not blindly continuing a schedule already atzero.'],no_new_checkpoint_evaluation=True,no_gold_or_quality_inputs=True)
 (out/'diagnose.py').write_bytes(Path(__file__).read_bytes());(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({n:{k:v for k,v in x.items() if k not in ['epochs','first_nonzero_update_actual']} for n,x in arms.items()},indent=2))
if __name__=='__main__':main()
