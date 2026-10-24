"""Source preparation: one fresh two-epoch run; plan budget is frozen before launch."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    parent = path.resolve().parent
    while True:
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if parent.parent == parent: break
        parent = parent.parent


def resources():
    def query(argument):
        return subprocess.check_output(['nvidia-smi', argument, '--format=csv,noheader,nounits'],
                                       text=True, timeout=10)
    return dict(at_utc=datetime.now(timezone.utc).isoformat(),
        gpu=query('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),
        processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'),
        disk_free_bytes=os.statvfs('.').f_bavail * os.statvfs('.').f_frsize)


def main():
    os.umask(0o077)
    plan = json.loads(Path('formal-plan.json').read_text())
    assert plan['phase'] == 'train' and plan['formal_training_authorized'] is True
    cap = plan['budget_seconds']
    assert type(cap) is int and cap == 2400 and plan['grace_seconds'] == 30
    assert plan['epochs'] == 2 and plan['optimizer_calls'] == 32 and plan['warmup_steps'] == 2
    assert plan['rows'] == 124 and plan['alpha'] == .5
    assert not any(Path(n).exists() for n in ['driver-started.json', 'child-started.json',
        'wrapper-complete.json', 'stdout.log', 'stderr.log'])
    for name, item in plan['files'].items():
        path = Path(name)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    output = Path(plan['output_dir'])
    assert not output.exists(), 'Fresh only; never auto-resume a previous epoch or probe'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    assert os.environ['PATIENTPHEX_DURABILITY_ROOT'] == plan['storage_parent']
    assert '--resume' not in plan['argv'] and '--stop-after-epoch' not in plan['argv']
    fresh = resources(); save('fresh-resource-gate.json', fresh)
    target = [line.split(',') for line in fresh['gpu'].splitlines() if plan['gpu_uuid'] in line]
    assert len(target) == 1 and int(target[0][2]) <= 10 and int(target[0][3]) >= 39000
    assert plan['gpu_uuid'] not in fresh['processes'] and fresh['disk_free_bytes'] >= 4 * 1024**3
    start = time.monotonic(); utc = time.time()
    save('driver-started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        driver_pid=os.getpid(), plan_sha256=sha('formal-plan.json'), command=plan['argv'],
        deadline_TERM_unix=utc+cap, deadline_KILL_unix=utc+cap+30,
        budget_seconds=cap, grace_seconds=30, no_automatic_retry=True))
    timed_out = False
    with Path('stdout.log').open('xb') as out, Path('stderr.log').open('xb') as err:
        child = subprocess.Popen(plan['argv'], stdout=out, stderr=err, start_new_session=True)
        save('child-started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            driver_pid=os.getpid(), child_pid=child.pid))
        try: rc = child.wait(timeout=max(.001, start+cap-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(child.pid, signal.SIGTERM)
            try: rc = child.wait(timeout=max(.001, start+cap+30-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); rc = child.wait()
        for stream in [out, err]: stream.flush(); os.fsync(stream.fileno())
    report = dict(at_utc=datetime.now(timezone.utc).isoformat(), returncode=rc,
        child_pid=child.pid, timed_out=timed_out, elapsed_training_seconds=time.monotonic()-start,
        budget_seconds=cap, grace_seconds=30, no_automatic_retry=True)
    endpoint_error = None
    if rc == 0 and not timed_out:
        try:
            complete_path = output/'training_complete.json'
            complete = json.loads(complete_path.read_text())
            assert complete['epochs'] == 2 and complete['global_step'] == 32 and not complete['smoke_only']
            assert complete['source_sha256'] == plan['trainer_sha256']
            durability = {}
            for epoch in (1, 2):
                checkpoint = output/('epoch-%03d' % epoch)
                receipt_path = output/('durability-epoch-%03d.json' % epoch)
                receipt = json.loads(receipt_path.read_text())
                assert receipt['completed_epochs'] == epoch and receipt['global_step'] == 16*epoch
                assert receipt['source_sha256'] == complete['source_sha256']
                assert receipt['file_directory_fsync'] and receipt['reopened_SHA_all_exact']
                assert receipt['persistent_root'] == plan['storage_parent']
                assert receipt['ancestor_directories_fsynced'][0] == str(output)
                assert receipt['ancestor_directories_fsynced'][-1] == plan['storage_parent']
                assert Path(receipt['checkpoint']) == checkpoint
                assert receipt['checkpoint_metadata_sha256'] == sha(checkpoint/'checkpoint.json')
                for name, item in receipt['files'].items():
                    path = checkpoint/name
                    assert not path.is_symlink() and path.stat().st_size == item['bytes']
                    assert sha(path) == item['sha256']
                durability[str(epoch)] = dict(path=str(receipt_path), sha256=sha(receipt_path), receipt=receipt)
            report.update(training_complete_sha256=sha(complete_path), durability=durability,
                all_epochs_persistent_and_readback_verified=True, no_pending_export=True,
                no_probe_state_or_old_adapter_used=True, no_inference_or_quality=True)
        except Exception as error:
            endpoint_error = dict(type=type(error).__name__, message=str(error))
            report['endpoint_error'] = endpoint_error
    try:
        release = resources(); report['gpu_release'] = release
        report['target_has_no_process'] = plan['gpu_uuid'] not in release['processes']
    except Exception as error: report['release_unknown'] = type(error).__name__
    report['total_controller_seconds'] = time.monotonic()-start
    save('wrapper-complete.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(rc or (1 if timed_out or endpoint_error else 0))


if __name__ == '__main__': main()
