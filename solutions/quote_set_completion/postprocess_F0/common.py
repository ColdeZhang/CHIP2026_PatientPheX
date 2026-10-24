"""Pinned existing interpretation/normalization; local metadata IO only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
RUN = Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1')
POLICY_SHA = 'b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
SOURCE_PINS = {
    'quality/metadata_recovery/support.py': 'a70eca6b5c0fbf6a2c5474ebd4ed3976b020109e1c4b43cf9bd81a716f3072b4',
    'quality/metadata_recovery/assemble.py': '4e3153d399e56611a5059b0f4c74687068bb95fa3b3652628c66e3a43ccb1ee6',
    'inference/decode.py': '8e98f9ef2c1e123e9d071dc83f525a92ba90f26fffcca89532bcd78ac2b61305',
    'schema.py': '460dd7cce3208a7aca521f73e6a675959eb649b562312a877eeac0f2af36edbf',
    'normalization/strict.py': '31d380f582a6ab83e0c73bddbac983c5cf038b250dd2b29355c14f11d8eaebf5',
    'normalization/semantic.py': 'e0e015a9b37593f474cb2d053842db74b1992685e972dda3c616b3f3952c9b82',
    'normalization/public_bank.py': 'd17acf74a151d8214d9d54af73cedaaadedeb77b956d606fb501e87eb8649061',
    'normalization/policy_v1.json': POLICY_SHA,
}
GOLD = {'path': '/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl', 'bytes': 507921, 'sha256': '630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48'}
ORIGINAL_DEV_GOLD = '/workspace/patientphex-data/day1/dev16.gold.jsonl'
BASE_SHA = '0c37ee543876f957bbb04049d0be51a905565d6219559ebab5d66abd43bc3ac3'
BASE_SCORE = .7281956289113009


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item
    return Path(item['path'])


def read(path): return json.loads(Path(path).read_text())
def jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def guard():
    state = {'F0_quality_enabled': False}
    blocked = {os.path.realpath(GOLD['path']), os.path.realpath(ORIGINAL_DEV_GOLD)}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if path in blocked and not (path == os.path.realpath(GOLD['path']) and state['F0_quality_enabled']):
                raise PermissionError('Held gold cannot be opened/hashed before the fixed complete freeze')
    sys.addaudithook(audit)
    return state


def modules():
    for relative, sha in SOURCE_PINS.items(): assert ref(OLD / relative)['sha256'] == sha
    sys.path.insert(0, str(OLD.parents[1]))
    sys.path.insert(0, str(OLD / 'normalization'))
    H = load('support', OLD / 'quality/metadata_recovery/support.py')
    A = load('_fixed_grounded_old_assemble', OLD / 'quality/metadata_recovery/assemble.py')
    D = load('_fixed_grounded_old_decode', OLD / 'inference/decode.py')
    schema = sys.modules['solutions.grounded_generation.schema']
    assert ref(schema.__file__)['sha256'] == SOURCE_PINS['schema.py']
    strict = load('strict', OLD / 'normalization/strict.py')
    semantic = load('semantic', OLD / 'normalization/semantic.py')
    return H, A, D, schema, strict, semantic


def save(path, value):
    return publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def publish(path, data):
    path = Path(path); pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path); pending.unlink()
    parent = path.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent
    return ref(path)


def source_contract():
    return dict(reused={name: ref(OLD / name) for name in SOURCE_PINS},
                new={p.name: ref(p) for p in sorted(HERE.glob('*.py'))})


def check_sources(contract):
    for group in ('reused', 'new'):
        for item in contract[group].values(): checked(item)
    assert {k: v['sha256'] for k, v in contract['reused'].items()} == SOURCE_PINS


def load_plan(path, stage):
    plan = read(path)
    assert plan['schema'] == 'quote-completion-postprocess-v1' and plan['stage'] == stage
    assert plan['gold_metadata_only'] == GOLD
    assert plan['state'] == 'BOUND_COMPLETE_ACTUAL_ARTIFACTS'
    check_sources(plan['sources'])
    return plan


def all_patient_keys(inputs):
    keys = {(str(d['pmc_id']), p['patient_id']) for d in inputs for p in d['patient']}
    assert len(inputs) == 16 and len(keys) == 47
    return keys


