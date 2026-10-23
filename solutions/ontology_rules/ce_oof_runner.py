#!/usr/bin/env python3
"""One bounded, sequential GPU campaign; never scores an evaluation label."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ce_oof_control import sha, save


def main():
    ap = argparse.ArgumentParser()
    for name in ('work', 'hpo', 'semantic-source', 'sap-model', 'biomed-model', 'outer-model'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    root = Path(a.work).resolve()
    started = time.monotonic()
    stages = []
    fixed = json.loads((root / 'source-fingerprint.json').read_text())
    for relative, expected in fixed.items():
        assert sha(root / relative) == expected
    assert fixed['gpu_candidates_frozen.py'] == 'dc5cc0b85922f7dc45b0f4f9fd7d8a5828a0cc234020923b31ea67a2059c143f'
    assert fixed['context_calibrate.py'] == 'ce843f8244b8e8a63f72c98b78ac8dd1a32b08ab22c4c01cf32a26fecfabf9c7'
    assert sha(a.semantic_source) == 'b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a'
    def run(name, command):
        remaining = 720 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Total campaign cap reached; retain completed stages')
        before = time.monotonic()
        save(root / 'progress.json', {'active': name, 'completed': stages, 'elapsed_seconds': before - started, 'budget_seconds': 720})
        with (root / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        record = {'name': name, 'command': command, 'seconds': time.monotonic() - before, 'returncode': result.returncode}
        stages.append(record)
        save(root / 'progress.json', {'active': None, 'completed': stages, 'elapsed_seconds': time.monotonic() - started, 'budget_seconds': 720})
        assert result.returncode == 0, name
    try:
        for g in range(1, 4):
            run(f'generator{g}', [sys.executable, '-u', 'gpu_candidates_frozen.py',
                '--train', f'generator{g}/train16.jsonl', '--input', f'generator{g}/input32.jsonl',
                '--hpo', a.hpo, '--model-dir', a.sap_model, '--semantic-source', a.semantic_source,
                '--work-dir', f'generator{g}/scored', '--max-candidates', '6000'])
        run('prepare_examples', [sys.executable, '-u', 'ce_oof_control.py', 'examples', '--work', '.', '--hpo', a.hpo])
        for i in range(1, 4):
            run(f'fit{i}', [sys.executable, '-u', 'context_calibrate.py', 'fit',
                '--train-examples', f'fold{i}/examples/train.jsonl.gz', '--input-examples', f'fold{i}/examples/input.jsonl.gz',
                '--model', a.biomed_model, '--output-dir', f'fold{i}/trained', '--epochs', '3',
                '--batch-size', '32', '--max-length', '160', '--learning-rate', '0.00002'])
        run('outer32', [sys.executable, '-u', 'ce_oof_control.py', 'infer', '--model', a.outer_model,
            '--input', 'outer-missing.jsonl.gz', '--output', 'outer-missing-scores.jsonl.gz'])
        selected = [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json', '.gz', '.log'} and 'embeddings' not in p.parts]
        save(root / 'complete.json', {'source_sha256': sha(__file__), 'stages': stages,
            'seconds': time.monotonic() - started, 'budget_seconds': 720,
            'files': {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in selected},
            'models': {str((root / f'fold{i}/trained/model/model.safetensors').relative_to(root)):
                       sha(root / f'fold{i}/trained/model/model.safetensors') for i in range(1, 4)},
            'heldout_evaluation_run': False})
    except BaseException as exc:
        save(root / 'failure.json', {'exception': type(exc).__name__, 'message': str(exc),
            'stages': stages, 'seconds': time.monotonic() - started, 'budget_seconds': 720})
        raise


if __name__ == '__main__':
    main()
