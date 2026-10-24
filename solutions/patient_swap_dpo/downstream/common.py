"""Unchanged selected helpers from original Qwen3 common.py; provenance in source-manifest."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, value):
    def convert(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item, key=repr)
        raise TypeError(type(item).__name__)
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=convert)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def update(path, value):
    temporary = Path(str(path) + '.next')
    save(temporary, value)
    os.replace(temporary, path)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def verify_assets(plan, *, full, previous=None):
    result = {}
    for name, expected in plan['model_files'].items():
        path = Path(plan['model_dir']) / name
        state = path.stat()
        assert state.st_size == expected['size'], name
        item = {'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns, 'inode': state.st_ino}
        if full:
            item['sha256'] = sha(path)
            assert item['sha256'] == expected['sha256'], name
        else:
            assert previous is not None
            assert all(item[key] == previous[name][key] for key in item), name
            item['sha256'] = previous[name]['sha256']
            assert item['sha256'] == expected['sha256']
        result[name] = item
    return result


def memory(torch):
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
            'free_bytes': torch.cuda.mem_get_info()[0]}
