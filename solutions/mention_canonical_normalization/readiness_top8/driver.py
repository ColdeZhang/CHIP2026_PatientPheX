from pathlib import Path
import json,subprocess,time,os
D=Path(__file__).parent;p=json.loads((D/'plan.json').read_text());R=Path(p['remote_root']);J=R/'jobs';J.mkdir(exist_ok=False)
def save(n,v):
 with (J/n).open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
start=time.time();cmd=['timeout','--signal=TERM','--kill-after=30s','120s',p['assets']['python'],'-u',str(D/'query.py')];save('start.json',{'unix':start,'pid':os.getpid(),'command':cmd})
with (J/'stdout').open('x') as out,(J/'stderr').open('x') as err:
 child=subprocess.Popen(cmd,stdout=out,stderr=err);save('child.json',{'pid':child.pid});rc=child.wait()
save('ended.json',{'unix':time.time(),'seconds':time.time()-start,'child_exit_code':rc,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)})
raise SystemExit(rc)
