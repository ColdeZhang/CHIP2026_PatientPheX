"""Bounded, expiring, exclusive CPU/inference controller; never retries."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--python',required=True)
    ap.add_argument('--entry',required=True);ap.add_argument('--plan',required=True)
    ap.add_argument('--budget',type=int,required=True);ap.add_argument('--grace',type=int,required=True)
    ap.add_argument('--gpu',action='store_true');a=ap.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);expires=float(os.environ['LAUNCH_EXPIRES_UNIX'])
    def save(p,x):
        with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(root/'admission.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'expires_unix':expires,'at_unix':time.time()},f);f.flush();os.fsync(f.fileno())
    if time.time()>expires:
        save(root/'admission-terminal.json',{'status':'EXPIRED_BEFORE_CHILD','at_unix':time.time(),'started_child':False});return
    job=root/'job';job.mkdir(exist_ok=False);start=time.time()
    cmd=['timeout','--signal=TERM',f'--kill-after={a.grace}s',f'{a.budget}s',a.python,'-u',a.entry,'--plan',a.plan]
    save(job/'start.json',{'at_unix':start,'pid':os.getpid(),'command':cmd,'launch_expires_unix':expires,'budget_seconds':a.budget,'TERM_grace':a.grace})
    assert time.time()<=expires,'Expired before child spawn'
    with (job/'stdout').open('x') as out,(job/'stderr').open('x') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err);save(job/'child.json',{'pid':child.pid});rc=child.wait()
    result={'at_unix':time.time(),'seconds':time.time()-start,'child_exit_code':rc}
    if a.gpu:
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        result['apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    save(job/'ended.json',result)
    raise SystemExit(rc)

if __name__=='__main__':main()
