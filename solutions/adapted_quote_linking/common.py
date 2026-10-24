"""Small exclusive durable records; raw experiment artifacts stay outside Git."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def ref(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, raw):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    p = p.parent.resolve()
    while True:
        fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        if p.parent == p: break
        p = p.parent


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def deny_gold():
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if '.gold.' in p: raise PermissionError('Held gold prohibited in preparation/query')
    sys.addaudithook(audit)


def module(name, item):
    p = checked(item); spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def pairs(rows, evaluator):
    return {(str(d['pmc_id']), pid, hid) for d in rows
            for pid, values in evaluator.associations(d).items() for hid in values if hid != 'NO'}


def primary(base, new_grounded, sampled, qwen, coded):
    # Old G is deliberately absent from this function's inputs.
    return base | (new_grounded & (sampled | qwen | coded))
