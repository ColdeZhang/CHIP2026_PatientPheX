"""Unique CPU or fresh GPU inference with one absolute deadline, no retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from raw_store import publish_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def resources():
    def query(arg):
        return subprocess.check_output(['nvidia-smi',arg,'--format=csv,noheader,nounits'],text=True,timeout=10)
    fs = os.statvfs('.')
    return dict(at_utc=datetime.now(timezone.utc).isoformat(),gpu=query('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),
                processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'),disk_free_bytes=fs.f_bavail*fs.f_frsize)


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
    os.umask(0o077);p=json.loads(Path(args.plan).read_text());root=Path(p['storage_parent'])
    cap=p['budget_seconds'];grace=p['grace_seconds'];gpu=p['phase']=='infer'
    assert (cap,grace)==((2400,30) if gpu else (120,5))
    assert p['execution_authorized'] and p['expected_rows']==47 and p['all_rows_fresh']
    assert os.environ['PATIENTPHEX_DURABILITY_ROOT']==str(root)
    for name,item in p['files'].items():
        assert Path(name).stat().st_size==item['bytes'] and sha(name)==item['sha256']
    assert not Path(p['output_dir']).exists()
    assert all(not Path(n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'])
    if gpu:
        assert os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid']
        # Parent-owned release receipt is a prerequisite; fresh hardware checks
        # still apply and do not infer release from an old F1 job.
        assert p['predecessor_release']['confirmed_by_owner'] is True
        gate=resources();publish_json('fresh-resource-gate.json',gate,root)
        target=[s.split(',') for s in gate['gpu'].splitlines() if p['gpu_uuid'] in s]
        assert len(target)==1 and int(target[0][2])<=10 and int(target[0][3])>=23000
        assert p['gpu_uuid'] not in gate['processes'] and gate['disk_free_bytes']>=512*1024**2
    else:
        assert os.environ['CUDA_VISIBLE_DEVICES']==''
    start=time.monotonic();utc=time.time();timed_out=False
    publish_json('driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),
        plan_sha256=sha(args.plan),command=p['argv'],deadline_TERM_unix=utc+cap,deadline_KILL_unix=utc+cap+grace,
        budget_seconds=cap,grace_seconds=grace,no_automatic_retry=True),root)
    with Path('stdout.log').open('xb') as out,Path('stderr.log').open('xb') as err:
        child=subprocess.Popen(p['argv'],stdout=out,stderr=err,start_new_session=True)
        publish_json('child-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),child_pid=child.pid),root)
        try:
            rc=child.wait(timeout=max(.001,start+cap-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:
                rc=child.wait(timeout=max(.001,start+cap+grace-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        for f in [out,err]:
            f.flush();os.fsync(f.fileno())
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=rc,child_pid=child.pid,timed_out=timed_out,
                elapsed_seconds=time.monotonic()-start,budget_seconds=cap,grace_seconds=grace,no_automatic_retry=True)
    if rc==0 and not timed_out:
        try:
            result=Path(p['output_dir'])/('complete.json' if gpu else 'result.json')
            v=json.loads(result.read_text())
            assert v['status']==p['expected_status'] and v['rows']==47 and v['documents']==16
            assert v['source']['sha256']==p['assets']['runner']['sha256'] and v['plan']['sha256']==sha(args.plan)
            refs=([v['responses']]+v['raw_refs']+v['generation_refs']) if gpu else [v['runtime'],v['observations']]
            for item in refs:
                q=Path(item['path']);q.resolve().relative_to(Path(p['output_dir']).resolve())
                assert q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
            if gpu:
                assert v['no_regeneration_or_answer_repair'] and 0<=v['strict_rows']<=v['EOS_rows']<=47 and 0<=v['cap_rows']<=47
                assert len(v['raw_refs'])==len(v['generation_refs'])==47
            report.update(result_sha256=sha(result),complete47_actual_members_verified=True,no_quality_performed=True)
        except Exception as error:
            report['endpoint_error']=dict(type=type(error).__name__,message=str(error))
    if gpu:
        try:
            report['gpu_release']=resources();report['target_has_no_process']=p['gpu_uuid'] not in report['gpu_release']['processes']
        except Exception as error:
            report['release_unknown']=type(error).__name__
    publish_json('wrapper-complete.json',report,root);print(json.dumps(report),flush=True)
    raise SystemExit(rc or (1 if timed_out or 'endpoint_error' in report else 0))


if __name__=='__main__':
    main()
