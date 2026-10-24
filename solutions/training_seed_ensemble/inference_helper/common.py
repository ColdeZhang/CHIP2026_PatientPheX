"""Metadata-only immutable file helpers; no model, gold or evaluator imports."""
import hashlib
import json
import os
from pathlib import Path

S = Path(__file__).resolve().parent
ROUTE = S.parent
G = Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
B = Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1')
OLD_B = B.parent / 'grounded-generation-v1'
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REMOTE = PROJECT + '/training-seed-ensemble-v1/infer-F1-v1'


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    parent = path.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def verify_sources():
    manifest = read(S / 'source-manifest.json')
    for item in manifest['original_sources'].values(): checked(item)
    for item in manifest['helper_sources'].values(): checked(item)
    for item in manifest['fixed_protocol'].values(): checked(item)
    return manifest
