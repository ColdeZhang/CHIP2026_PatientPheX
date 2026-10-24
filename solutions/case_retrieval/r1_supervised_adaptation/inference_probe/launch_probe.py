"""One240+TERM30 GPU1 probe owner; no training, retry or full80 continuation."""
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
from probe_contract import check_sources, ref
from raw_first import atomic


def snapshot():
    gpu = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=index,uuid,name,memory.used,memory.free,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    return dict(at_utc=datetime.now(timezone.utc).isoformat(), gpu=gpu, compute_processes=processes)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan_path = args.plan.resolve(); directory = plan_path.parent
    plan = json.loads(plan_path.read_text())
    assert plan['execution_ready'] and plan['adapter_binding'] is not None
    assert plan['budget_seconds'] == 240 and plan['grace_seconds'] == 30
    assert not (directory / 'launch-started.json').exists() and not (directory / 'terminal.json').exists()
    assert not (directory / 'actual').exists()
    check_sources(plan, directory)
    state = dict(status='starting_unique_final_adapter_longest_probe', started_utc=datetime.now(timezone.utc).isoformat(),
                 plan=ref(plan_path), source=ref(__file__), budget_seconds=240, grace_seconds=30,
                 original_index=73, full80_automatically_authorized=False, quality_calls=0)
    atomic(directory / 'launch-started.json', state)
    child = None; started = time.monotonic()
    try:
        gate = snapshot(); atomic(directory / 'fresh-GPU1.json', gate)
        fields = [part.strip() for part in gate['gpu'].split(',')]
        assert len(fields) == 6 and fields[0] == '1' and fields[1] == plan['gpu_uuid'] and fields[2] == 'NVIDIA A30'
        assert int(fields[3]) <= 100 and int(fields[4]) >= 23000 and int(fields[5]) == 24576
        assert not any(line.split(',')[0].strip() == plan['gpu_uuid'] for line in gate['compute_processes'].splitlines())
        assert shutil.disk_usage(directory).free >= 1024 ** 3
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        assert environment.get('PYTORCH_ALLOC_CONF') is None and environment.get('PYTORCH_CUDA_ALLOC_CONF') is None
        command = [sys.executable, '-u', str(directory / 'infer_one.py'), '--plan', str(plan_path)]
        state['command'] = command; state['timeout_triggered'] = False
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                     cwd=directory, start_new_session=True)
            state['child_pid'] = child.pid
            atomic(directory / 'child.json', dict(pid=child.pid, owned_process_group=os.getpgid(child.pid),
                                                  started_utc=datetime.now(timezone.utc).isoformat()))
            try:
                code = child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                state['timeout_triggered'] = True; os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        state.update(exit_code=code, status='complete_owned_probe_process')
    except BaseException as error:
        state.update(status='failed_wrapper_preserved_no_retry', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        try:
            release = snapshot(); state['release'] = release
            state['target_UUID_has_no_compute_process'] = not any(
                line.split(',')[0].strip() == plan['gpu_uuid'] for line in release['compute_processes'].splitlines())
        except BaseException as error:
            state['release_query_error_type'] = type(error).__name__
        state['files'] = {str(path.relative_to(directory)): {key: value for key, value in ref(path).items() if key != 'path'}
                          for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'terminal.json'}
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        atomic(directory / 'terminal.json', state)
        print(json.dumps({key: state.get(key) for key in ['status', 'exit_code', 'wall_seconds',
                                                        'timeout_triggered', 'target_UUID_has_no_compute_process']}), flush=True)


if __name__ == '__main__':
    main()
