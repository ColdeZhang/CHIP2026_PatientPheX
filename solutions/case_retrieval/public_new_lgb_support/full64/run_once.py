"""Bounded one-shot originaldev controller; no partial-run retry or re-evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(__file__) == plan['sources']['controller']['sha256']
    source = Path(plan['sources']['control']['path'])
    assert sha(source) == plan['sources']['control']['sha256']
    assert plan['cap_seconds'] == 30 and plan['TERM_grace_seconds'] == 1
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(source), '--plan', str(args.plan.resolve()), '--output', str(output)]
    state = dict(status='running', command=command, plan_sha256=sha(args.plan),
                 started_utc=datetime.now(timezone.utc).isoformat(), cap_seconds=30, TERM_grace_seconds=1)
    (output / 'execution-state.json').write_text(json.dumps(state, indent=2) + '\n')
    start = time.monotonic()
    with (output / 'stdout.log').open('x') as stdout, (output / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        state['pid'] = process.pid
        (output / 'execution-state.json').write_text(json.dumps(state, indent=2) + '\n')
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            state['timed_out'] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    state.update(returncode=code, seconds=time.monotonic() - start,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    state['status'] = 'complete' if code == 0 and not state.get('timed_out') else 'failed_preserved_no_retry'
    (output / 'execution-state.json').write_text(json.dumps(state, indent=2) + '\n')
    assert state['status'] == 'complete', state
    result = json.loads((output / 'result.json').read_text())
    assert result['new_evaluate_report_calls'] == 1 and result['old_evaluation_calls'] == 0
    print(json.dumps(dict(status=state['status'], keep=result['keep'], seconds=state['seconds'])))


if __name__ == '__main__':
    main()
