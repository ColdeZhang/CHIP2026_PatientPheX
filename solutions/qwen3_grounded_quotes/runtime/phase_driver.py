"""One phase, admission TTL and independent actual runtime clock; no retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--ticket', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    backend = Path(plan['backend_source_dir'])
    sys.path.insert(0, str(backend))
    import common
    ticket_path = Path(args.ticket).resolve()
    ticket = json.loads(ticket_path.read_text())
    assert ticket['plan'] == common.ref(plan_path)
    assert ticket['mode'] in plan['authorized_phases'] and plan['GPU_authorized'] is True
    expires = ticket['admission_expires_unix']
    assert time.time() < expires
    mode = ticket['mode']
    seconds = plan['budgets'][mode]
    assert isinstance(seconds, int) and seconds > 0 and plan['grace_seconds'] == 30
    root = Path(ticket['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    for item in plan['source_refs'].values():
        common.checked(item)
    started = time.time()
    deadline = started + seconds
    assert started < expires < deadline
    common.save(root / 'driver-started.json', dict(pid=os.getpid(), at_utc=common.stamp(),
        started_unix=started, admission_expires_unix=expires, deadline_unix=deadline,
        kill_unix=deadline + 30, plan=common.ref(plan_path), ticket=common.ref(ticket_path),
        no_automatic_retry=True))
    common.sync_dir(root)
    assert time.time() < expires
    env = dict(os.environ)
    env.update(plan['environment'])
    env['CUDA_VISIBLE_DEVICES'] = plan['gpu_uuid']
    cmd = ['timeout', '--signal=TERM', '--kill-after=30s', str(max(1, deadline-time.time())) + 's',
        plan['python'], str(Path(__file__).parent / 'qwen3_entry.py'), mode,
        '--plan', str(plan_path), '--output', str(root / 'output'),
        '--admission-expires-unix', str(expires), '--absolute-deadline-unix', str(deadline)]
    with (root / 'stdout.log').open('xb') as out, (root / 'stderr.log').open('xb') as err:
        child = subprocess.Popen(cmd, cwd=Path(__file__).parent, env=env, stdout=out, stderr=err)
        common.save(root / 'child-started.json', dict(pid=child.pid, at_utc=common.stamp()))
        rc = child.wait()
        for handle in [out, err]:
            handle.flush()
            os.fsync(handle.fileno())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    result = dict(returncode=rc, seconds=time.time()-started, at_utc=common.stamp(),
        actual_deadline_unix=deadline, child=child.pid,
        complete_file_exists=(root/'output/complete.json').is_file(),
        target_gpu=[x for x in gpu.splitlines() if plan['gpu_uuid'] in x],
        target_apps=[x for x in apps.splitlines() if plan['gpu_uuid'] in x])
    common.save(root / 'driver-complete.json', result)
    common.sync_dir(root)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
