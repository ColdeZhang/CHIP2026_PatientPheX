"""Fresh hard-negative training via unchanged native loop; existing mention decoder."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'reuse/mention'))
import gpu_entry as original
from common import checked,sha,save,stamp,gpu_gate,verify_assets,load_model,memory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['formal','infer'],required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(plan['run_root'])/a.mode;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch=None;stage='source'
 try:
  for name,value in plan['source_files'].items():assert sha(Path(__file__).parent/name)==value,name
  cpu=json.loads(checked(plan['native_cpu']).read_text());assert cpu['status']=='PASS_INCREMENTAL_HARD_NEGATIVE_NATIVE_INPUTS_ALL578' and cpu['train_rows']==2931 and cpu['held_rows']==1416;assert cpu['schedule']=={'epochs':2,'updates_per_epoch':367,'updates':734,'last_group_size':3,'warmup_steps':37};assert cpu['all_new_shapes_dominated_by_previous_actual_probe_frontier']
  assert plan['shape_inheritance']['old_mention_probe']['indexed_forward_backward']==14 and plan['shape_inheritance']['new_host_probe']['indexed_forward_backward']==13
  previous=checked(plan['same_host_verified_assets']) if a.mode=='formal' else Path(plan['run_root'])/'formal/verified-assets.json'
  gpu_gate(plan,out);assets=verify_assets(plan,full=False,previous=json.loads(previous.read_text()));save(out/'verified-assets.json',assets)
  adapter=None
  if a.mode=='infer':
   formal=json.loads((Path(plan['run_root'])/'formal/complete.json').read_text());assert formal['global_step']==734 and len(formal['acknowledgements'])==2
   for epoch in [1,2]:
    ack=json.loads((Path(plan['run_root'])/f'formal/epoch-{epoch:03d}-persistent-ack.json').read_text());checked(ack['checkpoint'])
    for item in ack['files'].values():checked(item)
   adapter=str(Path(plan['run_root'])/'formal/epoch-002/adapter')
  stage='model_load';torch,np,model,named,manifest=load_model(plan,out,adapter);stage=a.mode
  result=original.native.train(plan,out,cpu,torch,np,model,named) if a.mode=='formal' else original.infer(plan,out,cpu,torch,model)
  result.update(at_utc=stamp(),elapsed_seconds=time.monotonic()-started,phase=a.mode);save(out/'complete.json',result)
 except BaseException:save(out/'failure.json',{'stage':stage,'seconds':time.monotonic()-started,'traceback':traceback.format_exc(),'memory':memory(torch) if torch is not None else None});raise

if __name__=='__main__':main()
