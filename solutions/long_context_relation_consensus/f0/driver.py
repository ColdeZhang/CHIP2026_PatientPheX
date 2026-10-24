"""Exclusive H69 launch; single120s + TERM30 frozen-head F0 inference clock."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import save,ref,fsync_dir


def inventory(root,uuid):
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    target=[[x.strip() for x in line.split(',')] for line in gpu.splitlines() if uuid in line]
    assert len(target)==1
    return {'observed_unix':time.time(),'all_gpus':gpu,'all_apps':apps,'target':target[0],
            'target_apps':[x for x in apps.splitlines() if uuid in x],'diskfree':shutil.disk_usage(root).free}


def main():
    root=Path(__file__).resolve().parent;job=root/'job';job.mkdir(exist_ok=True)
    cfg=json.loads((root/'infer-plan.json').read_text());ticket=json.loads((root/'launch-ticket.json').read_text())
    if time.time()>ticket['launch_not_after_unix']:
        save(job/'expired-before-entry.json',{'status':'EXPIRED_NO_MODEL_OR_CHILD','ticket':ticket});return
    with (job/'exclusive-launch.json').open('x') as f:json.dump({'pid':os.getpid(),'ticket':ticket},f);f.flush();os.fsync(f.fileno())
    fsync_dir(job);assert ticket['plan']==ref(root/'infer-plan.json')
    for n,r in cfg['source_files'].items():assert ref(root/n)['sha256']==r['sha256']
    cpu=json.loads((root/'cpu-complete.json').read_text());assert cpu['status']=='PASS_ALL_F0_NATIVE_IDS_OWNERS_NO_CUDA'
    gate=inventory(root,cfg['gpu_uuid']);save(job/'fresh-gate.json',gate)
    assert gate['target'][0]=='0' and int(gate['target'][2])==24576
    assert not gate['target_apps'] and int(gate['target'][3])<=32 and int(gate['target'][4])>=23000
    assert gate['diskfree']>=4_000_000_000
    assert time.time()<=ticket['launch_not_after_unix'],'Expired before Popen'
    start=time.time();deadline=start+cfg['budget_seconds']
    with (job/'stdout').open('wb') as stdout,(job/'stderr').open('wb') as stderr:
        p=subprocess.Popen([cfg['python'],'-u',str(root/'infer.py')],cwd=root,
           env={**os.environ,**cfg['environment'],'CUDA_VISIBLE_DEVICES':cfg['gpu_uuid']},stdout=stdout,stderr=stderr,start_new_session=True)
        save(job/'started.json',{'status':'STARTED','driver_pid':os.getpid(),'child_pid':p.pid,'started_unix':start,
             'TERM_unix':deadline,'KILL_unix':deadline+30,'ticket':ticket,'plan':ref(root/'infer-plan.json')})
        timeout=False
        try:rc=p.wait(timeout=max(.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
    save(job/'ended.json',{'status':'COMPLETE' if rc==0 else 'FAILED','returncode':rc,'timed_out':timeout,
       'started_unix':start,'ended_unix':time.time(),'elapsed_seconds':time.time()-start,'release':inventory(root,cfg['gpu_uuid'])})
    sys.exit(rc)


if __name__=='__main__':main()
