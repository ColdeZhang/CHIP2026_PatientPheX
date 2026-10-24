"""Small local quality evidence helpers; no model, parser or gold imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ENDPOINTS = ('primary-fulltext-fla-alone', 'diagnostic-fulltext-normal-union-fla')
COMPLETE = 'COMPLETE_FIXED58_SFT48_FULLTEXT_FLA_NO_QUALITY'
QUALITY_COMPLETE = 'COMPLETE_FIXED_FULLTEXT_FLA_F0_DEVELOPMENT_PAIR'
GOLD_SHA = '630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48'
LEGACY_SHA = '662aeccc6efef1f93e7a9cde76710ca9714f4eaabcf75e93b4eb5c1984991ce3'
LITERAL_SHA = '1c70b9c4a8fb61be607618564409a1ea3480f9e099900c4d9b9aaf4b6ded2bf0'
ACTUAL_NAMES = {
    'inference_plan':'plan.json', 'inference_source':'infer.py', 'launcher':'launch.py',
    'terminal':'terminal.json', 'summary':'actual/summary.json', 'complete':'actual/complete.json',
    'responses':'actual/responses.jsonl.gz', 'canonical':'actual/canonical.jsonl.gz',
    'record_manifest':'actual/record-manifest.json', 'assets':'actual/actual-input-and-assets.json',
    'generation':'actual/generation-config.json', 'cache_use':'actual/cache-use.json',
    'backend':'actual/actual-FLA-bindings-and-dependencies.json',
    'longest_gate':'actual/longest-engineering-gate.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''): h.update(data)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def same(actual, expected):
    assert all(actual[k] == expected[k] for k in ('bytes', 'sha256'))


def checked(record):
    assert set(record) == {'path','bytes','sha256'}
    path = Path(record['path']); assert path.is_absolute()
    same(ref(path), record); return path


def unique(items):
    result = {}
    for key, value in items:
        assert key not in result
        result[key] = value
    return result


def read(path): return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def rows(path):
    return [json.loads(line, object_pairs_hook=unique) for line in Path(path).read_text().splitlines() if line.strip()]


def jsha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    fsync_directory(path.parent)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded
    spec.loader.exec_module(loaded); return loaded


def reference_gate(plan, source, ready):
    assert plan['execution_ready'] is ready and plan['schema'] == 'fixed-fulltext-FLA-F0-paired-quality-v1'
    assert plan['endpoints'] == list(ENDPOINTS) and plan['new_v2_calls'] == 2 and plan['old_v2_calls'] == 0
    assert plan['historical_F0_gold_sha256'] == GOLD_SHA
    assert str(plan['gold_path']).endswith('/round2-inner4/fold-0/validation16.gold.jsonl')
    assert plan['source_sha256'] == sha(source)
    assert plan['cpu_seconds'] == 60 and plan['grace_seconds'] == 5
    assert plan['wall_seconds'] == 65 and plan['wall_grace_seconds'] == 5
    directory = Path(source).resolve().parent
    for name, expected in plan['sources'].items():
        assert Path(name).name == name; same(ref(directory/name),expected)
    records = [plan['quality_contract'], plan['legacy_helpers'], plan['literal_helpers'], *plan['references'].values()]
    assert 'cached_probe_references' not in plan
    if ready:
        assert isinstance(plan['expected_inference_sources'],dict) and plan['expected_inference_sources']
        assert isinstance(plan['expected_inference_plan_sha256'],str) and len(plan['expected_inference_plan_sha256']) == 64
    if ready:
        assert set(plan['endpoint_references']) == set(ACTUAL_NAMES)
        records += list(plan['endpoint_references'].values()) + [plan['published_inference_plan_reference']]
    else: assert plan['endpoint_references'] is None
    for record in records:
        assert Path(record['path']).resolve() != Path(plan['gold_path']).resolve() and record['sha256'] != GOLD_SHA
        checked(record)
    assert plan['legacy_helpers']['sha256'] == LEGACY_SHA and plan['literal_helpers']['sha256'] == LITERAL_SHA
    assert Path(plan['literal_helpers']['path']) == Path(plan['legacy_helpers']['path']).with_name('literal_helpers.py')
    return records
