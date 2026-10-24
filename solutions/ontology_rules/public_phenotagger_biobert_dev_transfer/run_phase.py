"""Bound one prepared CPU phase and retain its real outcome; no retry or search."""
import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector';sys.path.insert(0,str(COMMON))
from prepare_pool import sha,read,sources

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);sources(p);root=Path(p['record_dir']);root.mkdir(parents=True,exist_ok=False)
 assert p['phase'] in ('prepare_train','fit_full64','predict_dev','finalize_dev')
 command=[p['python'],str(Path(__file__).with_name(p['phase']+'.py')),'--plan',str(Path(a.plan).resolve())]
 cap=p['CPU_cap_seconds'];assert 0<cap<=120
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 state={'phase':p['phase'],'started_at':now(),'source_sha256':sha(__file__),'plan_sha256':sha(a.plan),'command':command,'cap_seconds':cap,'TERM_grace_seconds':5,'status':'running'}
 def save():(root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 save();start=time.monotonic()
 with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
  proc=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True);state['pid']=proc.pid;save()
  try:rc=proc.wait(timeout=cap)
  except subprocess.TimeoutExpired:
   state['timeout']=True;os.killpg(proc.pid,signal.SIGTERM)
   try:rc=proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
 state.update(returncode=rc,seconds=time.monotonic()-start,finished_at=now(),status='complete' if rc==0 and not state.get('timeout') else 'failed_preserved');save();print(json.dumps(state));assert state['status']=='complete'
if __name__=='__main__':main()
