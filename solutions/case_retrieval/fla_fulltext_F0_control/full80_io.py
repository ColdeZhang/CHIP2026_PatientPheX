"""Original deterministic JSONL/gzip persistence; no cache loading."""
import gzip
import json
import os
from pathlib import Path
from common import ref


def jsonl_with_gzip(path, values):
    path = Path(path)
    data = ''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in values).encode()
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    archive = path.with_suffix(path.suffix+'.gz')
    with archive.open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as zipped: zipped.write(data)
        handle.flush(); os.fsync(handle.fileno())
    assert gzip.decompress(archive.read_bytes()) == data
    descriptor=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return {'plain':ref(path),'gzip':ref(archive)}
