"""Original bounded three-phase controller shape, fixed canonical-name budgets."""
from datetime import datetime,timezone,timedelta
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def ref(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(Path(p).parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    assert plan['authorized_phases']==['probe','formal','infer'] and plan['budgets']==dict(probe=240,formal=2400,infer=900,each_TERM_grace=30)
    for item in plan['source_refs'].values():assert ref(item['path'])==item
    root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=False)
    save(root/'driver-start.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),plan=ref(a.plan),source=ref(__file__),phase_budgets=plan['budgets']))
    stages=[]
    for phase in ['probe','formal','infer']:
        if phase=='formal':
            previous=json.loads((root/'probe/complete.json').read_text())
            if not previous['formal_budget_gate']:
                save(root/'closed-before-formal.json',dict(reason='Original measured cost exceeds2400',probe=ref(root/'probe/complete.json')));break
        seconds=plan['budgets'][phase];tick=time.monotonic();at=datetime.now(timezone.utc)
        env=dict(os.environ);env.update(plan['environment']);env['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid']
        cmd=[plan['python'],str(Path(__file__).parent/'qwen3_entry.py'),phase,'--plan',str(Path(a.plan).resolve())]
        with (root/f'{phase}.stdout.log').open('xb') as stdout,(root/f'{phase}.stderr.log').open('xb') as stderr:
            process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            item=dict(phase=phase,pid=process.pid,started_at_utc=at.isoformat(),term_at_utc=(at+timedelta(seconds=seconds)).isoformat(),kill_at_utc=(at+timedelta(seconds=seconds+30)).isoformat(),budget_seconds=seconds,command=cmd)
            save(root/f'{phase}-child-start.json',item)
            try:process.wait(timeout=max(.001,tick+seconds-time.monotonic()));timed_out=False
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=max(.001,tick+seconds+30-time.monotonic()))
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            for f in [stdout,stderr]:f.flush();os.fsync(f.fileno())
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
        item.update(return_code=process.returncode,timed_out=timed_out,seconds=time.monotonic()-tick,
            complete_file_exists=(root/phase/'complete.json').is_file(),released_gpu=dict(target_gpu=[s for s in gpu.splitlines() if plan['gpu_uuid'] in s],target_apps=[s for s in apps.splitlines() if plan['gpu_uuid'] in s]))
        save(root/f'{phase}-child-result.json',item);stages.append(item)
        if process.returncode or timed_out or not item['complete_file_exists'] or item['released_gpu']['target_apps']:break
    save(root/'driver-result.json',dict(status='COMPLETE_THREE_PHASES' if len(stages)==3 and all(x['return_code']==0 and not x['timed_out'] and x['complete_file_exists'] for x in stages) else 'STOPPED_ORIGINAL_NO_RETRY',stages=stages,no_quality_calls=True))

if __name__=='__main__':main()
