"""Small immutable evidence utilities; no model imports."""
import json
import os
from pathlib import Path
import legacy_runner as legacy


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=legacy.digest(path))


def checked(value):
    path = Path(value['path']); actual = ref(path)
    assert all(actual[k] == value[k] for k in ('bytes', 'sha256'))
    return path


def fsync_chain(directory, root):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    assert directory == root or root in directory.parents
    while True:
        descriptor = os.open(str(directory), os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if directory == root: break
        directory = directory.parent


def save(path, value, replace=False):
    path = Path(path); assert replace or not path.exists()
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=legacy.json_default, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    assert replace or not path.exists()
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def check_sources(plan, directory):
    for name, expected in plan['sources'].items():
        assert Path(name).name == name
        actual = ref(Path(directory) / name)
        assert all(actual[k] == expected[k] for k in ('bytes', 'sha256'))
