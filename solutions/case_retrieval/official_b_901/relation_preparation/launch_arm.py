"""One bounded old-Qwen B arm; preserve terminal/partial evidence on every exit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd=os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan', type=Path, required=True);a=ap.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    assert plan['policy']=='fixed_old1a703b_official901_B_arm_v1'
    root=Path(plan['project_root']).resolve()
    assert out.resolve().is_relative_to(root)
    def fsync_parent_chain():
        cursor=out.resolve()
        while True:
            fd=os.open(cursor,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            if cursor==root:break
            cursor=cursor.parent
    fsync_parent_chain()
    assert plan['arm'] in ('normal','reverse','public-new')
    assert 60 <= plan['cap_seconds'] <= 10800 and plan['term_grace_seconds']==30
    assert not (out/'launch-started.json').exists() and not (out/'responses.jsonl').exists()
    for name, expected in plan['local_files'].items():
        p=out/name;assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256']
    assert digest(__file__)==plan['launcher_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid']
    audit=json.loads((out/'native-audit.json').read_text())
    assert audit['status']=='PASS_NATIVE_INPUT_BUDGET' and audit['count']==plan['rows']
    assert audit['maximum_input_tokens']==plan['maximum_input_tokens']<=14000
    assert audit['prompts']['sha256']==digest(out/'prompts.jsonl')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    matched=[line.split(', ') for line in gpu.splitlines() if line.startswith(plan['gpu_uuid']+',')]
    assert len(matched)==1 and int(matched[0][1])<=32
    assert not any(line.startswith(plan['gpu_uuid']+',') for line in processes.splitlines())
    assert shutil.disk_usage(out).free>=plan['minimum_disk_free_bytes']
    started=time.time();save(out/'launch-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=digest(a.plan), launcher_sha256=digest(__file__), gpu=gpu, processes=processes,
        free_bytes=shutil.disk_usage(out).free, absolute_deadline_epoch=started+plan['cap_seconds'],
        term_grace_seconds=30, no_new_fit=True, pid=os.getpid()))
    command=[plan['python'],'-u',str(out/'qwen35_runner_raw_first.py'),'--model',plan['model'],
             '--adapter',plan['adapter'],'--source-manifest',str(out/'model-source-manifest.json'),
             '--prompts',str(out/'prompts.jsonl'),'--native-audit',str(out/'native-audit.json'),
             '--output',str(out/'responses.jsonl'),'--max-input-tokens','14000','--max-new-tokens','1600']
    timeout=False
    with (out/'stdout.log').open('xb') as stdout, (out/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        save(out/'child-process.json',dict(pid=child.pid,command=command,started_epoch=started))
        try:rc=child.wait(timeout=max(0.001,started+plan['cap_seconds']-time.time()))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
    members={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and not p.is_symlink() and p.name!='terminal.json':
            members[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    release_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    release_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    save(out/'terminal.json',dict(exit_code=rc,timeout_triggered=timeout,arm=plan['arm'],
        wall_seconds=time.time()-started,finished_utc=datetime.now(timezone.utc).isoformat(),files=members,
        gpu_after=release_gpu,processes_after=release_processes,
        scope='Old frozen1a703b officialB inference only; no training/quality/upload. All completed raw IDs and responses preserved on failure.'))
    fsync_parent_chain()
    print(json.dumps(dict(exit_code=rc,timeout_triggered=timeout,arm=plan['arm'],wall_seconds=time.time()-started)),flush=True)
    raise SystemExit(rc if not timeout else 124)


if __name__=='__main__':main()
