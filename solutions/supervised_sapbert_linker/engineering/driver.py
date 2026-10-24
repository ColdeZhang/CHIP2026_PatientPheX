"""Single exclusive launch, prebound launch expiry, then separate 300+30 clock."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from common import save,ref,fsync_dir


def inventory(root,uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    target=[x.strip() for line in rows.splitlines() if uuid in line for x in [line]][0]
    cols=[x.strip() for x in target.split(',')]
    return dict(observed_unix=time.time(),gpu=rows,apps=apps,target=cols,free_bytes=shutil.disk_usage(root).free,
        target_apps=[x for x in apps.splitlines() if uuid in x])


def main():
    root=Path(__file__).resolve().parent
    ticket=json.loads((root/'launch-ticket.json').read_text())
    cfg=json.loads((root/'execution-plan.json').read_text());job=root/'job';job.mkdir(exist_ok=True)
    if time.time()>ticket['launch_not_after_unix']:
        save(job/'expired-before-entry.json',dict(status='EXPIRED_NO_MODEL_START',ticket=ticket));return
    with open(job/'exclusive-launch.json','x') as f:
        json.dump(dict(pid=os.getpid(),start_unix=time.time(),ticket=ticket),f);f.flush();os.fsync(f.fileno())
    fsync_dir(job)
    assert ticket['plan']==ref(root/'execution-plan.json')
    for name,r in cfg['source_files'].items():assert ref(root/name)['sha256']==r['sha256']
    cpu=json.loads((root/'cpu-complete.json').read_text());assert cpu['status']=='PASS_ALL_NATIVE_IDS_NO_MODEL_OR_CUDA'
    fresh=inventory(root,cfg['gpu_uuid']);save(job/'fresh-gate.json',fresh)
    assert fresh['target'][0]=='0' and int(fresh['target'][2])==24576
    assert not fresh['target_apps'] and int(fresh['target'][3])<=32 and int(fresh['target'][4])>=23000
    assert fresh['free_bytes']>=3_000_000_000
    assert time.time()<=ticket['launch_not_after_unix'],'Launch expired before Popen; no model may start'
    started=time.time();deadline=started+300
    env={**os.environ,**cfg['environment'],'CUDA_VISIBLE_DEVICES':cfg['gpu_uuid']}
    with open(job/'stdout','wb') as stdout,open(job/'stderr','wb') as stderr:
        child=subprocess.Popen([cfg['python'],'-u',str(root/'probe.py')],cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        save(job/'started.json',dict(status='STARTED',driver_pid=os.getpid(),child_pid=child.pid,started_unix=started,
            term_deadline_unix=deadline,kill_deadline_unix=deadline+30,launch_ticket=ticket,plan=ref(root/'execution-plan.json')))
        timed_out=False
        try: rc=child.wait(timeout=max(.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    release=inventory(root,cfg['gpu_uuid'])
    save(job/'ended.json',dict(status='COMPLETE' if rc==0 else 'FAILED',returncode=rc,timed_out=timed_out,
        started_unix=started,ended_unix=time.time(),elapsed_seconds=time.time()-started,release=release,
        probe_complete=ref(root/'probe-output/complete.json') if (root/'probe-output/complete.json').exists() else None))
    sys.exit(rc if rc else 0)


if __name__=='__main__':main()
