"""Bind complete adapted-R1 inference to fixed paired quality; never read gold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(record):
    actual = ref(record['path'])
    assert actual['bytes'] == record['bytes'] and actual['sha256'] == record['sha256']
    return Path(record['path'])


def same(actual, expected):
    assert all(actual[key] == expected[key] for key in ('bytes', 'sha256'))


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--inference-dir', type=Path, required=True)
    parser.add_argument('--cached-row73', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.inference_dir.resolve()
    plan = json.loads(args.template.read_text())
    assert not plan['execution_ready'] and plan['endpoint_references'] is None
    assert plan['cpu_seconds'] == 60 and plan['grace_seconds'] == 5
    assert ref(args.source)['sha256'] == plan['source_sha256']
    assert not Path(plan['output_dir']).exists() and not args.output.exists()
    endpoints = {name: ref(root / relative) for name, relative in {
        'inference_plan': 'plan.json', 'inference_result': 'actual/result.json',
        'inference_source': 'infer_full80.py', 'launcher': 'launch_full80.py',
        'terminal': 'terminal.json', 'responses': 'actual/responses.jsonl.gz'}.items()}
    endpoints['cached_row73'] = ref(args.cached_row73)
    assert endpoints['cached_row73']['sha256'] == '00e2dcf183487482cf37e32f6da0fbb58c6d0c950afc20ad19f4d61d7d631f88'
    terminal = json.loads((root / 'terminal.json').read_text())
    result = json.loads((root / 'actual/result.json').read_text())
    infer = json.loads((root / 'plan.json').read_text())
    assert terminal['exit_code'] == 0 and terminal['timeout_triggered'] is False
    assert terminal['target_UUID_has_no_compute_process']
    assert result['status'] == 'COMPLETE_FIXED80_FINAL_ADAPTER_79FRESH_1CACHE_NO_QUALITY'
    assert (result['new_generations'], result['cached_generations'], result['completed_rows'], result['quality_calls']) == (79, 1, 80, 0)
    assert infer['cached_indices'] == result['cached_indices'] == [73]
    assert infer['new_indices'] == result['new_indices'] == [i for i in range(80) if i != 73]
    assert result['adapter_binding'] == infer['adapter_binding']
    for actual, expected in [(endpoints['inference_plan'], terminal['plan']),
                             (endpoints['launcher'], terminal['source']),
                             (endpoints['inference_plan'], result['plan']),
                             (endpoints['inference_source'], result['source']),
                             (endpoints['responses'], result['responses_gzip'])]:
        same(actual, expected)
    for name, expected in terminal['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        same(ref(root / relative), expected)
    for name, expected in infer['sources'].items():
        assert Path(name).name == name
        same(ref(root / name), expected)
    same(ref(root / 'actual/row73.json'), endpoints['cached_row73'])
    assert len(result['row_sources']) == 80
    assert [row['original_index'] for row in result['row_sources']] == list(range(80))
    # Do not open or hash gold. Only source/config/input references are bound here.
    for record in [plan['quality_contract'], plan['quality_addendum'], plan['legacy_helpers'], *plan['references'].values()]:
        assert Path(record['path']).resolve() != Path(plan['gold_path']).resolve()
        checked(record)
    runner = Path(__file__).resolve().with_name('run_quality_once.py')
    plan.update(execution_ready=True, endpoint_references=endpoints,
        bound_utc=datetime.now(timezone.utc).isoformat(), template_reference=ref(args.template),
        binder_reference=ref(__file__), quality_runner_reference=ref(runner),
        binding_scope='Complete terminal and all original file SHA; no gold/model/parser/score execution.',
        wall_seconds=65, wall_grace_seconds=5,
        budget_interpretation='Child RLIMIT_CPU hard60; external absolute wall65 then TERM5. No reset after checks, no retry.')
    save(args.output, plan)
    print(json.dumps(ref(args.output)))


if __name__ == '__main__':
    main()
