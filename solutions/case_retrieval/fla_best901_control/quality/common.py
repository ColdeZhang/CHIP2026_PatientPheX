"""Pure evidence utilities for one complete FLA901 quality endpoint."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SCHEMA = 'fixed-originaldev901-four-FLA-one-quality-v1'
QUALITY_COMPLETE = 'COMPLETE_ORIGINAL901_ALL_FLA_SINGLE_DEVELOPMENT_QUALITY'
GOLD_SHA = 'c1f0f24a58b62bb898eb672550dc58247ddb9a0e8768c6f899812cfc327cc050'
NATIVE_SOURCE_SHA = '1b414126210d4d6992b58fd6ce3f1840c42a5dc258e389843e51966ebe789a71'
FLOWS = ('normal', 'reverse', 'public-new', 'patient')
MARKER = 'evaluation-started-fla-best901-v1.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve(); return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def same(actual, expected):
    assert all(actual[k] == expected[k] for k in ('bytes', 'sha256'))


def checked(record):
    assert set(record) == {'path', 'bytes', 'sha256'}
    p = Path(record['path']); assert p.is_absolute(); same(ref(p), record); return p


def unique(items):
    result = {}
    for k, v in items:
        assert k not in result
        result[k] = v
    return result


def read(path): return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def rows(path):
    return [json.loads(s, object_pairs_hook=unique) for s in Path(path).read_bytes().splitlines() if s.strip()]


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save(path, value):
    p = Path(path)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    fsync_directory(p.parent)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def reference_gate(plan, source, ready):
    assert plan['schema'] == SCHEMA and plan['execution_ready'] is ready
    assert plan['new_v2_calls'] == 1 and plan['old_v2_calls'] == 0
    assert plan['gold_sha256'] == GOLD_SHA
    assert str(plan['gold_path']).endswith('/day1/dev16.gold.jsonl')
    assert plan['source_sha256'] == sha(source)
    assert (plan['cpu_seconds'], plan['wall_seconds'], plan['grace_seconds']) == (60, 65, 5)
    for name, expected in plan['sources'].items():
        assert Path(name).name == name
        same(ref(Path(source).resolve().parent / name), expected)
    records = [plan['quality_contract'], plan['native_assembly_source'],
               plan['native_replay'], plan['expected_gpu_plan'],
               *plan['fixed_references'].values(), *plan['pure_sources'].values()]
    assert plan['native_assembly_source']['sha256'] == NATIVE_SOURCE_SHA
    if ready:
        assert set(plan['actual']) == {'root', 'plan', 'terminal', 'files'}
        records += [plan['actual']['plan'], plan['actual']['terminal'], *plan['actual']['files'].values()]
    else: assert plan['actual'] is None
    for record in records:
        assert Path(record['path']).resolve() != Path(plan['gold_path']).resolve()
        assert record['sha256'] != GOLD_SHA
        checked(record)
    contract = read(plan['quality_contract']['path'])
    assert plan['fixed_references'] == contract['fixed_references']
    assert plan['pure_sources'] == contract['pure_sources']
    assert plan['expected_gpu_plan'] == contract['expected_gpu_plan']
    if ready:
        required = {k: plan['fixed_references'][k] for k in
                    ('input','old_candidates','old_HGB_support','task1_template')}
        for arm in ('normal','reverse','public','patient'):
            stem = 'public_new' if arm == 'public' else arm
            flow = 'public-new' if arm == 'public' else arm
            required[arm+'_prompts'] = plan['fixed_references'][stem+'_prompts']
            required[arm+'_responses'] = plan['actual']['files']['actual/'+flow+'/responses.jsonl']
        assert plan['assembly_references'] == required
    else: assert plan['assembly_references'] is None
    return records
