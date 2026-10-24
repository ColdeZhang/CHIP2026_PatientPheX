"""Durable JSON and canonical digest helpers; no probe entrypoint."""
from pathlib import Path
import hashlib
import json
import os
import time

import legacy_runner as legacy


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=legacy.json_default)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

