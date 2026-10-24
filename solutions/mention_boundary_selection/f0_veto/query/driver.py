"""Single-owner, expiring admission; one original 180+30 query job, no retry."""
from pathlib import Path
import json, subprocess, time, os, sys, hashlib

D=Path(__file__).parent
p=json.loads((D/'plan.json').read_text())
R=Path(p['remote_root'])
R.mkdir(parents=True,exist_ok=True)
expires=float(os.environ['LAUNCH_EXPIRES_UNIX'])

def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

fd=os.open(R/'admission.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:
    json.dump({'pid':os.getpid(),'expires_unix':expires,'at_unix':time.time()},f)
    f.flush();os.fsync(f.fileno())
if time.time()>expires:
    save(R/'admission-terminal.json',{'status':'EXPIRED_BEFORE_START','at_unix':time.time(),'GPU_started':False})
    raise SystemExit(124)
J=R/'jobs';J.mkdir(exist_ok=False)
for name,h in p['entry_source_files'].items():
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h
start=time.time()
cmd=['timeout','--signal=TERM','--kill-after=30s','180s',p['assets']['python'],'-u',str(D/'query.py')]
save(J/'start.json',{'unix':start,'pid':os.getpid(),'command':cmd,'launch_expires_unix':expires,'seconds':180,'TERM_grace':30})
assert time.time()<=expires,'Expired before child spawn'
with (J/'stdout').open('x') as out,(J/'stderr').open('x') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err)
    save(J/'child.json',{'pid':child.pid})
    rc=child.wait()
save(J/'ended.json',{'unix':time.time(),'seconds':time.time()-start,'child_exit_code':rc,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True),
    'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)})
raise SystemExit(rc)
