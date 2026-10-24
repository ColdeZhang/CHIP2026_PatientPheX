"""Own children only; training and normalization share one absolute deadline."""
from pathlib import Path
import argparse,datetime,json,os,signal,subprocess,sys,time,shutil
from common import save,ref

def inventory():
 gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
 return {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpus':gpu,'apps':apps}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['probe','formal'],required=True);ap.add_argument('--source',required=True);ap.add_argument('--encoded',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());assert json.loads(Path(plan['cpu_complete']['remote_path']).read_text())['status']=='PASS_COMPLETE_PURE_F0_NATIVE_GEOMETRY';O=Path(a.output);O.mkdir(parents=True,exist_ok=False)
 if a.mode=='formal':
  n=plan['normalization'];assert ref(Path(n['bank'])/'public-bank.npy')['sha256']==n['public_bank_sha256']
 before=inventory();before['free_disk_bytes']=shutil.disk_usage(O).free;save(O/'fresh-inventory.json',before);uuid=plan['gpu_uuid'];matching=[x for x in before['gpus'].splitlines() if uuid in x];assert len(matching)==1 and int(matching[0].split(',')[2].strip())<1000;assert not any(uuid in x for x in before['apps'].splitlines());assert before['free_disk_bytes']>5*1024**3
 train=[sys.executable,'-u',str(Path(a.source)/'infer.py'),'--plan',a.plan,'--encoded',a.encoded,'--output',str(O/'native')];commands=[('native',train)]
 if a.mode=='formal':
  normalize=[sys.executable,'-u',str(Path(a.source)/'normalize.py'),'--native',str(O/'native'),'--encoded',a.encoded,'--output',str(O/'normalization')]
  for key in ['input','train','hpo','bank','model']:normalize.extend(['--'+key,n[key]])
  commands.append(('normalization',normalize))
 seconds=plan[a.mode]['cap_seconds'];grace=plan[a.mode]['term_grace_seconds'];started=time.time();deadline=started+seconds;save(O/'started.json',{'started':started,'started_utc':datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),'deadline':deadline,'term_grace_seconds':grace,'budget_seconds':seconds,'driver_pid':os.getpid(),'commands':commands,'plan':ref(a.plan),'source':ref(__file__)})
 results=[];timed_out=False;rc=0
 for name,cmd in commands:
  if time.time()>=deadline:timed_out=True;rc=124;break
  stage_start=time.time()
  with (O/(name+'.stdout.log')).open('wb') as out,(O/(name+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);save(O/'child.json',{'stage':name,'pid':child.pid,'driver_pid':os.getpid(),'deadline':deadline});save(O/(name+'-child.json'),{'stage':name,'pid':child.pid,'stage_started':stage_start,'deadline':deadline})
   try:rc=child.wait(timeout=max(.001,deadline-time.time()))
   except subprocess.TimeoutExpired:
    timed_out=True;os.killpg(child.pid,signal.SIGTERM)
    try:rc=child.wait(timeout=grace)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
  results.append({'stage':name,'pid':child.pid,'returncode':rc,'timed_out':timed_out,'elapsed':time.time()-stage_start});save(O/'stages.json',results)
  if rc!=0 or timed_out:break
 after=inventory();save(O/'released-gpu.json',after);assert not any(uuid in line for line in after['apps'].splitlines()),'Target still has apps; release not established'
 save(O/'child-result.json',{'returncode':rc,'timed_out':timed_out,'elapsed':time.time()-started,'original_deadline':deadline,'started':started,'stages':results,'release':ref(O/'released-gpu.json'),'complete_exists':(O/'native/complete.json').exists(),'normalization_complete_exists':(O/'normalization/complete.json').exists()});raise SystemExit(rc if not timed_out else 124)
if __name__=='__main__':main()
