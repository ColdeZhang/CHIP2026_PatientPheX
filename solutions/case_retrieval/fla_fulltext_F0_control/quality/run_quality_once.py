"""One CPU60 / wall65+TERM5 paired quality owner; preserve every failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from common import ACTUAL_NAMES, QUALITY_COMPLETE, reference_gate


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))


def main():
    started = time.monotonic(); deadline = started + 65
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); directory = args.output_dir.resolve()
    directory.mkdir(exist_ok=False)
    descriptor = os.open(str(directory.parent), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    state = dict(status='STARTING_ONE_FIXED_FULLTEXT_FLA_F0_PAIR', at_utc=datetime.now(timezone.utc).isoformat(),
                 source=ref(args.source), plan=ref(args.plan), runner=ref(__file__),
                 CPU_hard_seconds=60, external_wall_seconds=65, TERM_grace_seconds=5,
                 timeout_triggered=False, retries=0)
    save(directory / 'intent.json', state)
    child = None; quality_directory = None; complete = False
    try:
        plan = json.loads(args.plan.read_text()); quality_directory = Path(plan['output_dir'])
        assert plan['execution_ready'] and plan['cpu_seconds'] == 60 and plan['grace_seconds'] == 5
        assert plan['wall_seconds'] == 65 and plan['wall_grace_seconds'] == 5
        assert plan['source_sha256'] == state['source']['sha256']
        assert plan['quality_runner_reference'] == state['runner']
        assert not quality_directory.exists() and quality_directory.resolve() != directory
        reference_gate(plan,args.source,True)
        assert not (Path(plan['endpoint_references']['terminal']['path']).parent/'evaluation-started-fulltext-fla-F0-v1.json').exists()
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-u', str(args.source.resolve()), '--plan', str(args.plan.resolve())]
        state['command'] = command
        assert time.monotonic() < deadline
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                     start_new_session=True, preexec_fn=child_limits)
            state['child_pid'] = child.pid
            save(directory / 'child.json', dict(pid=child.pid, process_group=os.getpgid(child.pid)))
            try:
                code = child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                state['timeout_triggered'] = True; os.killpg(child.pid, signal.SIGTERM)
                try: code = child.wait(timeout=max(0, deadline + 5 - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
            stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
        state['exit_code'] = code
        if code == 0 and not state['timeout_triggered']:
            summary = json.loads((quality_directory / 'summary.json').read_text())
            assert summary['status'] == QUALITY_COMPLETE
            assert summary['new_v2_calls'] == 2 and summary['old_v2_calls'] == 0
            complete = True
        state['status'] = 'COMPLETE_PAIRED_QUALITY' if complete else 'FAILED_PAIRED_QUALITY_PRESERVED'
    except BaseException as error:
        state.update(status='FAILED_WRAPPER_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                     error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=max(0, min(5, deadline + 5 - time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        if child is not None:
            state['exit_code'] = child.returncode
        state['quality_files'] = {} if quality_directory is None or not quality_directory.exists() else {
            str(path.relative_to(quality_directory)): ref(path)
            for path in sorted(quality_directory.rglob('*')) if path.is_file()}
        state['execution_files'] = {path.name: ref(path) for path in sorted(directory.iterdir())
                                    if path.is_file() and path.name != 'terminal.json'}
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started)
        save(directory / 'terminal.json', state)
        print(json.dumps({key: state.get(key) for key in ('status', 'exit_code', 'timeout_triggered', 'wall_seconds')}))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
