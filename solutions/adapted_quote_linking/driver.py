"""One admitted necessary-query child, fresh exact GPU0, 180+TERM30, no retries."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time
from common import checked, read, ref, save
from adapted_semantic import admitted


def resources():
    def query(arg):return subprocess.check_output(['nvidia-smi',arg,'--format=csv,noheader,nounits'],text=True,timeout=10)
    return dict(at_utc=datetime.now(timezone.utc).isoformat(),gpu=query('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),
                processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'),disk_free_bytes=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize)


def check_admission(plan):
    try:admitted(plan)
    except AssertionError:
        save('admission-rejected.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),deadline=plan['launch_valid_until_unix'],
            original_plan=ref('held-query-plan.json'),no_new_child_or_model_loaded=True))
        raise


def main():
    os.umask(0o077);p=read('held-query-plan.json');check_admission(p)
    assert p['schema']=='adapted-Grounded-necessary-query-execution-v1' and p['query_count']==254
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and os.environ['PATIENTPHEX_DURABILITY_ROOT']==p['storage_parent']
    for path,item in p['files'].items():
        actual=ref(path);assert {k:actual[k] for k in ['bytes','sha256']}==item
    assert not any(Path(n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'])
    assert not Path(p['output_dir']).exists()
    check_admission(p);gate=resources();save('fresh-resource-gate.json',gate)
    target=[x.split(',') for x in gate['gpu'].splitlines() if p['gpu_uuid'] in x]
    assert len(target)==1 and target[0][0].strip()=='0' and int(target[0][2])<=10 and int(target[0][3])>=6000
    assert p['gpu_uuid'] not in gate['processes'] and gate['disk_free_bytes']>=2*1024**3
    check_admission(p);begin=time.monotonic();start=time.time();cap=p['budget_seconds'];grace=p['grace_seconds'];timed_out=False
    save('driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),plan_sha256=ref('held-query-plan.json')['sha256'],
        launch_valid_until_unix=p['launch_valid_until_unix'],deadline_TERM_unix=start+cap,deadline_KILL_unix=start+cap+grace,
        budget_seconds=cap,grace_seconds=grace,command=p['argv'],no_automatic_retry=True))
    with Path('stdout.log').open('xb') as out,Path('stderr.log').open('xb') as err:
        check_admission(p);child=subprocess.Popen(p['argv'],stdout=out,stderr=err,start_new_session=True)
        save('child-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),child_pid=child.pid))
        try:rc=child.wait(timeout=max(.001,begin+cap-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=max(.001,begin+cap+grace-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        for f in [out,err]:f.flush();os.fsync(f.fileno())
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=rc,child_pid=child.pid,timed_out=timed_out,
        elapsed_seconds=time.monotonic()-begin,budget_seconds=cap,grace_seconds=grace,no_automatic_retry=True)
    if rc==0 and not timed_out:
        try:
            c=read(Path(p['output_dir'])/'complete.json')
            assert c['status']=='COMPLETE_ADAPTED_FIXED254_PUBLIC_QUERIES' and c['query_count']==254 and c['public_bank_rows']==41099
            assert c['plan']['sha256']==ref('held-query-plan.json')['sha256'] and c['source']['sha256']==p['source_files']['query.py']['sha256']
            assert c['no_new_fit'] and c['no_bank_reencoding'] and c['task_adapted'] and not c['no_task_fit']
            for key in ['scores','vectors','counts','encoder']:
                q=Path(c[key]['path']);q.resolve().relative_to(Path(p['output_dir']).resolve());checked(c[key])
            report.update(complete_necessary_query_members_verified=True,complete_sha256=ref(Path(p['output_dir'])/'complete.json')['sha256'],no_quality_performed=True)
        except Exception as error:report['endpoint_error']=dict(type=type(error).__name__,message=str(error))
    try:report['gpu_release']=resources();report['target_has_no_process']=p['gpu_uuid'] not in report['gpu_release']['processes']
    except Exception as error:report['release_unknown']=type(error).__name__
    save('wrapper-complete.json',report);print(__import__('json').dumps(report),flush=True)
    raise SystemExit(rc or (1 if timed_out or 'endpoint_error' in report else 0))


if __name__=='__main__':main()
