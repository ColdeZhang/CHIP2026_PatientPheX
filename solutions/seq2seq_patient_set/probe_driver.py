"""One bounded probe launch; records timeout result and fresh release evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def put(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def main(a):
    root=Path(a.root); case=root/a.namespace; app=case/'application'; output=case/'driver'
    output.mkdir(exist_ok=False); now=time.time()
    assert now<a.admit_until<=now+300
    assert now<a.term_deadline<=now+600
    setup=json.loads((root/'cpu-setup/complete.json').read_text())
    assert setup['status']=='COMPLETE_NATIVE_CPU_RUNTIME_AND_SINGLE_BASE_ASSETS'
    assert setup['runtime']['CUDA_initialized'] is False
    command=['timeout','--signal=TERM','--kill-after=30s',str(a.term_deadline-time.time())+'s',sys.executable,'-u',str(app/'probe.py'),
             '--plan',str(app/'probe-plan.json'),'--supply',str(root/'probe-v1/supply/manifest.json'),
             '--model-dir',str(root/'assets-base'),'--output',str(case/'output'),
             '--gpu-uuid',a.gpu_uuid,'--admit-until',str(a.admit_until),
             '--authorized-probe-only','four-shapes-two-adam-no-formal']
    env={**os.environ,**setup['environment'],'CUDA_VISIBLE_DEVICES':''}
    launched=time.time()
    put(output/'started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command,started_unix=launched,term_deadline_unix=a.term_deadline,
        kill_deadline_unix=a.term_deadline+30,pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with (output/'stdout').open('x') as stdout,(output/'stderr').open('x') as stderr:
        completed=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
    release=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    put(output/'terminal.json',dict(exit_code=completed.returncode,seconds=time.time()-launched,
        finished_unix=time.time(),gpu_apps=dict(rc=release.returncode,stdout=release.stdout,stderr=release.stderr),
        gpu_memory=dict(rc=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),
        selected_uuid=a.gpu_uuid,selected_uuid_has_no_compute_app=release.returncode==0 and a.gpu_uuid not in release.stdout,
        probe_complete=(case/'output/complete.json').exists(),probe_failure=(case/'output/failure.json').exists()))
    raise SystemExit(completed.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--namespace',choices=['probe-v1','probe-v2'],required=True);p.add_argument('--term-deadline',type=float,required=True)
    p.add_argument('--admit-until',type=float,required=True);main(p.parse_args())
