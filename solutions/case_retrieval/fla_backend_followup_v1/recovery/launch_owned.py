"""One absolute600+TERM30 A-only numerical owner; no training/retry/quality."""
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
from evidence import ref, save as atomic, fsync_chain

def check_sources(plan, directory):
    for name, expected in plan['sources'].items():
        actual=ref(directory/name)
        assert all(actual[k]==expected[k] for k in ('bytes','sha256'))
    assert ref(__file__)['sha256']==plan['launcher_sha256']


def snapshot():
    gpu = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    return dict(at_utc=datetime.now(timezone.utc).isoformat(), gpu=gpu, compute_processes=processes)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan_path = args.plan.resolve(); directory = plan_path.parent
    plan = json.loads(plan_path.read_text())
    assert plan['execution_ready'] and plan['CPU_result'] is not None
    assert plan['budget_seconds'] == 600 and plan['TERM_grace_seconds'] == 30 and plan['soft_stop_seconds'] == 570
    assert not (directory / 'launch-started.json').exists() and not (directory / 'terminal.json').exists()
    assert not (directory / 'actual').exists()
    check_sources(plan, directory)
    original_deadline = datetime.fromisoformat(plan['original_absolute_deadline_utc'])
    remaining_seconds = (original_deadline-datetime.now(timezone.utc)).total_seconds()
    assert 0 < remaining_seconds <= 600
    deadline = started + remaining_seconds
    state = dict(status='STARTING_UNIQUE_FLA_A_NUMERICAL_DIAGNOSTIC', started_utc=datetime.now(timezone.utc).isoformat(),
                 plan=ref(plan_path), source=ref(__file__), cap_seconds=600, term_grace_seconds=30,
                 soft_stop_seconds=570, scope='A_ONLY_NO_AUTOMATIC_B_C', quality_calls=0,
                 timeout_triggered=False, absolute_monotonic_deadline=deadline,
                 original_absolute_deadline_utc=plan['original_absolute_deadline_utc'],
                 remaining_budget_at_launch_seconds=remaining_seconds,
                 recovery_scope='H39 metadata serialization only; original absolute deadline retained; prior0forward preserved')
    atomic(directory / 'launch-started.json', state)
    fsync_chain(directory, Path(plan['project_root']))
    child = None; succeeded = False
    try:
        gate = snapshot(); atomic(directory / 'fresh-GPU1.json', gate)
        fields = [part.strip() for part in gate['gpu'].split(',')]
        assert len(fields) == 6 and fields[0] == '1' and fields[1] == plan['gpu_uuid'] and fields[2] == 'NVIDIA A30'
        assert int(fields[3]) <= 100 and int(fields[4]) >= 23000 and int(fields[5]) == 24576
        assert not any(line.split(',')[0].strip() == plan['gpu_uuid'] for line in gate['compute_processes'].splitlines())
        assert shutil.disk_usage(directory).free >= 2 * 1024 ** 3
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        assert environment.get('PYTORCH_ALLOC_CONF') is None and environment.get('PYTORCH_CUDA_ALLOC_CONF') is None
        command = [sys.executable, '-u', str(directory / 'probe_A.py'), '--plan', str(plan_path),
                   '--output', str(directory/'actual'), '--absolute-deadline-monotonic', str(deadline), '--run-authorized']
        state['command'] = command
        assert time.monotonic() < deadline
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                     cwd=directory, start_new_session=True)
            state['child_pid'] = child.pid
            atomic(directory / 'child.json', dict(pid=child.pid, owned_process_group=os.getpgid(child.pid),
                                                  started_utc=datetime.now(timezone.utc).isoformat()))
            try:
                code = child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                state['timeout_triggered'] = True; os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=max(0, deadline + 30 - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
            stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
        state['exit_code'] = code
        if code == 0 and not state['timeout_triggered']:
            result = json.loads((directory / 'actual/result.json').read_text())
            assert result['status'] == 'COMPLETE_A_RAW_DISTRIBUTION_DIAGNOSTIC_NO_ROLLOUT'
            assert result['quality_calls'] == 0 and result['training_steps'] == 0 and result['gold_read'] is False
            succeeded = True
        state['status'] = 'COMPLETE_FLA_A_NUMERICAL_PROCESS' if succeeded else 'FAILED_FLA_A_NUMERICAL_PROCESS_PRESERVED'
    except BaseException as error:
        state.update(status='FAILED_WRAPPER_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=max(0, min(30, deadline + 30 - time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        try:
            state['release'] = snapshot()
            state['target_UUID_has_no_compute_process'] = not any(
                line.split(',')[0].strip() == plan['gpu_uuid'] for line in state['release']['compute_processes'].splitlines())
        except BaseException as error:
            state['release_query_error_type'] = type(error).__name__
        state['files'] = {str(path.relative_to(directory)): {key: value for key, value in ref(path).items() if key != 'path'}
                          for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'terminal.json'}
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        atomic(directory / 'terminal.json', state)
        fsync_chain(directory, Path(plan['project_root']))
        print(json.dumps({key: state.get(key) for key in ['status', 'exit_code', 'wall_seconds',
                                                        'timeout_triggered', 'target_UUID_has_no_compute_process']}), flush=True)
    return 0 if succeeded else 1


if __name__ == '__main__':
    raise SystemExit(main())
