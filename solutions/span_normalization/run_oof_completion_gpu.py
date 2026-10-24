"""Sequential bounded nested GP jobs; three fresh models and eight target passes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import write_json, ref, sha


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); a = ap.parse_args()
    p = json.loads(Path(a.plan).read_text()); out = Path(p['output']); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'driver-started.json').exists(), 'Existing intent; recover rather than duplicate'
    start = time.monotonic(); wall = time.time(); deadline = start + 900
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    line = [x for x in gpu.splitlines() if p['gpu_uuid'] in x]; assert len(line)==1 and int(line[0].split(',')[2])<=64
    assert not any(p['gpu_uuid'] in x for x in apps.splitlines()) and shutil.disk_usage(out).free >= 8*1024**3
    assert p['gpu_budget_seconds']==900 and p['term_grace_seconds']==30
    for r in p['sources']:
        assert sha(r['path'])==r['sha256']
    write_json(out/'driver-started.json', {'at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'plan':ref(a.plan),
        'deadline_TERM_unix':wall+900,'deadline_KILL_unix':wall+930,'fresh_gpu':gpu,'fresh_apps':apps,'free_bytes':shutil.disk_usage(out).free})
    steps=[]; status='COMPLETE'
    for job in p['jobs']:
        if job['train_new']:
            jobs=[('train-'+job['name'],[sys.executable,'-u',p['source_dir']+'/train_infer.py','--model',p['model'],
                   '--preparation',job['preparation'],'--output',job['model_output'],'--gpu-uuid',p['gpu_uuid']])]
        else: jobs=[]
        # Existing model SHA is frozen; new-model SHA is read only after complete training.
        jobs.append(('score-'+job['name'],None))
        for name, cmd in jobs:
            if cmd is None:
                if job['train_new'] or job.get('reuse_model_output'):
                    completed=json.loads((Path(job.get('reuse_model_output') or job['model_output'])/'complete.json').read_text())
                    assert completed['status']=='COMPLETE_TRAINING_AND_FULL_HELD_INFERENCE'
                    weights=completed['model']
                else: weights=job['existing_weights']
                cmd=[sys.executable,'-u',p['source_dir']+'/score_targets.py','--model',p['model'],'--weights',weights['path'],
                     '--weights-sha256',weights['sha256'],'--preparation',job['preparation'],'--targets',job['targets'],
                     '--output',job['score_output'],'--gpu-uuid',p['gpu_uuid']]
            before=time.monotonic(); timed=False
            with (out/(name+'.stdout')).open('wb') as so,(out/(name+'.stderr')).open('wb') as se:
                child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True)
                write_json(out/'child.json',{'name':name,'pid':child.pid,'command':cmd})
                try: rc=child.wait(timeout=max(.001,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed=True;os.killpg(child.pid,signal.SIGTERM)
                    try:rc=child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
            steps.append({'name':name,'returncode':rc,'timed_out':timed,'seconds':time.monotonic()-before,
                          'stdout':ref(out/(name+'.stdout')),'stderr':ref(out/(name+'.stderr'))})
            write_json(out/'progress.json',{'steps':steps})
            if rc or timed:status='FAILED';break
        if status!='COMPLETE':break
    write_json(out/'terminal.json',{'status':status,'steps':steps,'seconds':time.monotonic()-start,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status':status,'seconds':time.monotonic()-start}))
    if status!='COMPLETE':sys.exit(1)


if __name__=='__main__':main()
