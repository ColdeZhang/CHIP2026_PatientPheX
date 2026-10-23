"""One bounded, read-only CPU boundary diagnostic; never runs a draft plan."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root = Path.cwd()
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan.get('source_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', plan['source_sha256']):
        raise ValueError('Draft plan: freeze the final analyzer SHA before running')
    assert not (root / 'run-state.json').exists()
    assert not (root / 'output').exists()
    assert not (root / 'stdout.log').exists() and not (root / 'stderr.log').exists()
    assert sha(__file__) == plan['runner_sha256']
    source = root / 'analyze_boundaries.py'
    assert sha(source) == plan['source_sha256']
    assert plan['source_name'] == source.name
    assert Path(plan['output_directory']) == root / 'output'
    runtime = plan['runtime']
    assert runtime['cwd'] == root.name == 'ontology-public-phenotagger-boundary-diagnostic'
    assert runtime['CPU_hardcap_seconds'] == 120 and runtime['TERM_grace_seconds'] == 5
    assert sys.executable == runtime['python']
    required_environment = {
        'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '',
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'USE_TORCH': '0', 'USE_TF': '0', 'USE_FLAX': '0',
    }
    for name, value in required_environment.items():
        assert runtime['environment'][name] == value, name
        assert os.environ.get(name) == value, name
    query_path = Path(plan['queries']['path'])
    assert query_path == root / 'diagnostic-queries.json'
    assert sha(query_path) == plan['queries']['sha256']
    assert query_path.stat().st_size == plan['queries']['bytes']
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '120s',
               sys.executable, source.name, '--plan', 'plan.json']
    state = dict(status='running', started_at_utc=now(), command=command,
                 source_sha256=plan['source_sha256'], runner_sha256=sha(__file__),
                 plan_sha256=sha(plan_path), queries_sha256=sha(query_path),
                 CPU_cap_seconds=120, TERM_grace_seconds=5)
    state_path = root / 'run-state.json'
    with state_path.open('x') as stream:
        stream.write(json.dumps(state, indent=2) + '\n')
    started = time.monotonic()
    try:
        with (root / 'stdout.log').open('x') as out, (root / 'stderr.log').open('x') as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=130)
        state['returncode'] = result.returncode
        assert result.returncode == 0, result.returncode
        state['summary_sha256'] = sha(root / 'output/summary.json')
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed_preserved'
        state['failure'] = repr(exc)
        raise
    finally:
        state['ended_at_utc'] = now()
        state['seconds'] = time.monotonic() - started
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
