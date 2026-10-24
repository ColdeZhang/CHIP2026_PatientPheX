"""Fresh TTL plus explicit bounded phase; records child exit and actual GPU release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ref,save


def main(a):
    p=Path(a.plan);plan=json.loads(p.read_text());b=json.loads(Path(a.binding).read_text());now=time.time()
    assert b['authorized_by_root'] and now<b['admit_until']<=now+300
    assert b['phase'] in ('train','infer') and b['grace_seconds']==30
    if b['phase']=='train':assert b['budget_seconds']==1800
    assert all(ref(p)[k]==b['plan'][k] for k in ('bytes','sha256'))
    cpu=json.loads(Path(b['target_CPU_receipt']['path']).read_text())
    assert all(ref(b['target_CPU_receipt']['path'])[k]==b['target_CPU_receipt'][k] for k in ('bytes','sha256'))
    assert cpu['status']=='PASS_TARGET_NATIVE_BASE_SP126_AND_TINY_FULL_STATE_CPU'
    assert all(cpu['source'][k]==plan['cpu_contract'][k] for k in ('bytes','sha256'))
    assert cpu['counts']=={'train':87,'held_input':39} and cpu['native_base_parameters']==247587456
    assert all(cpu['supply']['manifest.json'][k]==plan['supply'][k] for k in ('bytes','sha256'))
    for name,e in plan['model'].items():assert all(cpu['assets'][name][k]==e[k] for k in ('bytes','sha256'))
    out=Path(b['driver_output']);out.mkdir(parents=True,exist_ok=False)
    command=['timeout','--signal=TERM','--kill-after=30s',str(b['budget_seconds'])+'s',sys.executable,'-u',
             str(Path(__file__).parent/(b['phase']+'.py')),'--plan',str(p),'--binding',a.binding]
    launched=time.time();assert launched<b['admit_until']
    save(out/'started.json',{'started_unix':launched,'term_deadline_unix':launched+b['budget_seconds'],
         'kill_deadline_unix':launched+b['budget_seconds']+30,'command':command,'pid':os.getpid(),
         'binding':ref(a.binding),'plan':ref(p),'source':ref(__file__)})
    with (out/'stdout').open('x') as stdout,(out/'stderr').open('x') as stderr:
        child=subprocess.run(command,stdout=stdout,stderr=stderr,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        for f in (stdout,stderr):f.flush();os.fsync(f.fileno())
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    memory=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    save(out/'terminal.json',{'exit_code':child.returncode,'seconds':time.time()-launched,'finished_unix':time.time(),
        'expected_uuid':b['gpu_uuid'],'selected_uuid_has_no_compute_app':apps.returncode==0 and b['gpu_uuid'] not in apps.stdout,
        'gpu_apps':{'rc':apps.returncode,'stdout':apps.stdout,'stderr':apps.stderr},
        'gpu_memory':{'rc':memory.returncode,'stdout':memory.stdout,'stderr':memory.stderr},
        'phase_complete':(Path(b['output'])/'complete.json').exists(),'phase_failure':(Path(b['output'])/'failure.json').exists()})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--binding',required=True);main(p.parse_args())
