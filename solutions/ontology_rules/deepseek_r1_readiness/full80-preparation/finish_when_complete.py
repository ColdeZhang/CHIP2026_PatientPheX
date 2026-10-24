"""One local completion controller: observe existing monitor, collect, bind, fixed two evaluations.
Never launches/stops inference, retries a generation/evaluation, or changes frozen policies.
"""
from pathlib import Path
import json,hashlib,os,time,datetime,subprocess,sys,traceback
F=Path(__file__).parent
NAMES=['plan.json','infer_full80.py','launch_full80.py','full80_contract.py','normalized_final.py','strict_final.py','sparse_final.py','evaluate_full80.py','evaluation-template.json','collect_terminal.py','poll_original.py','bind_completed_evaluation.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 p=F/name
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 refs={n:sha(F/n) for n in NAMES};assert refs['plan.json']=='4a88bd33caf5ccdeb9ef3d27646437ad58719b5eadd23aee37024e18856b8f06' and refs['evaluate_full80.py']=='4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522'
 assert not (F/'quality').exists() and not (F/'evaluation-plan.json').exists()
 save('finish-controller-intent.json',{'at':now(),'pid':os.getpid(),'sources':refs,'only_existing_inference':True,'no_GPU_start_stop_or_retry':True,'policy':'await original complete80; each fixed evaluation exactly once; failures stop'})
 final_deadline=datetime.datetime.fromisoformat('2026-10-24T07:17:36.254031+00:00').timestamp()
 try:
  while True:
   if (F/'STOP-AUTO-FINISH').exists():raise RuntimeError('Explicit stop marker; leave remote job unchanged')
   if time.time()>final_deadline:raise TimeoutError('Completion observation window ended; no inference extension')
   p=F/'monitor-state.json'
   if p.exists():
    try:state=json.loads(p.read_text())
    except json.JSONDecodeError:time.sleep(1);continue
    if state['terminal']:break
   time.sleep(30)
  for n,h in refs.items():assert sha(F/n)==h
  steps=[('collect',[sys.executable,str(F/'collect_terminal.py')],300),('bind',[sys.executable,str(F/'bind_completed_evaluation.py')],60),('evaluate',[sys.executable,str(F/'evaluate_full80.py'),'--plan',str(F/'evaluation-plan.json'),'--responses',str(F/'recovered/terminal/responses.raw.jsonl.gz'),'--inference-result',str(F/'recovered/terminal/result.json')],300)]
  for name,cmd,budget in steps:
   if (F/'STOP-AUTO-FINISH').exists():raise RuntimeError('Explicit stop before '+name)
   for n,h in refs.items():assert sha(F/n)==h
   save('finish-'+name+'-intent.json',{'at':now(),'command':cmd,'timeout_seconds':budget,'no_automatic_retry':True})
   start=time.monotonic()
   with (F/('finish-'+name+'.stdout.log')).open('x') as out,(F/('finish-'+name+'.stderr.log')).open('x') as err:r=subprocess.run(cmd,cwd=F,stdout=out,stderr=err,timeout=budget,check=False,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
   receipt={'at':now(),'returncode':r.returncode,'seconds':time.monotonic()-start,'stdout_sha256':sha(F/('finish-'+name+'.stdout.log')),'stderr_sha256':sha(F/('finish-'+name+'.stderr.log'))};save('finish-'+name+'-complete.json',receipt)
   assert r.returncode==0,'Fixed step failed; no automatic repeat: '+name
  summary=F/'quality/summary.json';assert summary.exists();save('finish-controller-complete.json',{'at':now(),'status':'COMPLETE_FIXED_TWO_QUALITY_ENDPOINTS','summary':{'path':str(summary),'sha256':sha(summary),'bytes':summary.stat().st_size}});print(json.dumps(json.loads(summary.read_text())['endpoints']),flush=True)
 except BaseException as e:save('finish-controller-failed.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc(),'remote_inference_untouched':True,'no_automatic_retry':True});raise
if __name__=='__main__':main()
