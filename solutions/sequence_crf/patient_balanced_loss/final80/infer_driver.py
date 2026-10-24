"""One fixed fresh final80 inference, only after durable final training receipts."""
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
    plan = json.loads(Path('final80-plan.json').read_text())
    assert plan['phase'] == 'infer' and plan['inference_authorized'] is True
    cap = plan['budget_seconds']
    assert type(cap) is int and cap == 600 and plan['grace_seconds'] == 30
    assert plan['expected_rows'] == 80 and plan['fresh_only'] is True
    assert plan['strict_all80'] is True and plan['alpha'] == .5
    assert not any(Path(n).exists() for n in ['driver-started.json', 'child-started.json',
        'wrapper-complete.json', 'stdout.log', 'stderr.log'])
    for name, item in plan['files'].items():
        path = Path(name)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    output = Path(plan['output_dir'])
    assert not output.exists(), 'All80 fresh only; never reuse prior model output or partial responses'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    assert os.environ['PATIENTPHEX_DURABILITY_ROOT'] == plan['storage_parent']
    assert '--resume' not in plan['argv'] and '--stop-after-epoch' not in plan['argv']
    fresh = resources(); save('fresh-resource-gate.json', fresh)
    target = [line.split(',') for line in fresh['gpu'].splitlines() if plan['gpu_uuid'] in line]
    assert len(target) == 1 and int(target[0][2]) <= 10 and int(target[0][3]) >= 39000
    assert plan['gpu_uuid'] not in fresh['processes'] and fresh['disk_free_bytes'] >= 4 * 1024**3
    start = time.monotonic(); utc = time.time()
    save('driver-started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        driver_pid=os.getpid(), plan_sha256=sha('final80-plan.json'), command=plan['argv'],
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
        child_pid=child.pid, timed_out=timed_out, elapsed_inference_seconds=time.monotonic()-start,
        budget_seconds=cap, grace_seconds=30, no_automatic_retry=True)
    endpoint_error = None
    if rc == 0 and not timed_out:
        try:
            response=output/'responses.jsonl'; manifest_path=output/'responses.manifest.json'
            rows=[json.loads(line) for line in response.read_text().splitlines()]
            manifest=json.loads(manifest_path.read_text())
            assert len(rows)==len({row['row_id'] for row in rows})==80
            assert len({str(row['pmc_id']) for row in rows})==16
            assert manifest['expected_rows']==manifest['new_rows']==manifest['complete_decisions']==80
            assert manifest['strict_all80_EOS_and_PID_CID_schema'] and manifest['all_rows_fresh']
            assert manifest['cached_rows']==manifest['fallback_rows']==0 and manifest['no_cache_or_resume']
            assert manifest['generation_IDs_saved_before_decode'] and manifest['decoded_raw_saved_before_strict_parser']
            assert manifest['partial_or_abstain_policy'] is False and manifest['model_answer_repair'] is False
            assert manifest['output_sha256']==sha(response)
            for key in ['raw_index','generation_index','decisions']:
                reference=manifest[key]; path=Path(reference['path'])
                assert path.stat().st_size==reference['bytes'] and sha(path)==reference['sha256']
                assert len(json.loads(path.read_text()))==80
            completed=dict(rows=80,documents=16,no_quality=True,all_rows_fresh=True,cached_rows=0,
                strict_all80_EOS_and_PID_CID_schema=True,no_baseline_fallback=True,
                responses_sha256=sha(response),manifest_sha256=sha(manifest_path),
                inference_plan_sha256=sha('final80-plan.json'),original_child_complete_without_retry=True)
            save(output/'complete.json',completed)
            report.update(inference_complete_sha256=sha(output/'complete.json'),
                all80_fresh_strict_complete=True,no_quality=True)
        except Exception as error:
            endpoint_error = dict(type=type(error).__name__, message=str(error))
            report['endpoint_error'] = endpoint_error
    try:
        release = resources(); report['gpu_release'] = release
        report['target_has_no_process'] = plan['gpu_uuid'] not in release['processes']
    except Exception as error: report['release_unknown'] = type(error).__name__
    report['complete_exists'] = (output/'complete.json').is_file()
    report['total_controller_seconds'] = time.monotonic()-start
    save('wrapper-complete.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(rc or (1 if timed_out or endpoint_error else 0))


if __name__ == '__main__': main()
