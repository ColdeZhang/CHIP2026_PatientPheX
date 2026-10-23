"""Read only: inventory existing embedding headers and small model metadata."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    start = time.monotonic()
    root = Path(a.root)
    paths = ['frozen-fold%d-sapbert' % f for f in range(4)]
    paths += ['nested-outer0-inner%d' % f for f in [1, 2, 3]]
    paths += ['nested-outer%d/inner%d' % (outer, inner)
              for outer in [1, 2, 3] for inner in range(4) if outer != inner]
    output = []
    for name in paths:
        directory = root / 'ontology-round2' / name
        manifest = directory / 'manifest.json'
        files = []
        for path in sorted((directory / 'embeddings').glob('*.npy')):
            with path.open('rb') as stream:
                assert stream.read(6) == b'\x93NUMPY'
                version = tuple(stream.read(2))
                n = struct.unpack('<H' if version == (1, 0) else '<I', stream.read(2 if version == (1, 0) else 4))[0]
                header = ast.literal_eval(stream.read(n).decode('latin1'))
            files.append(dict(name=path.name, bytes=path.stat().st_size, header=header))
        output.append(dict(directory=str(directory), exists=directory.exists(),
                           manifest_sha256=sha(manifest) if manifest.exists() else None,
                           files=files, bytes=sum(f['bytes'] for f in files)))
    small = {}
    for name in ['semantic-neural/run.py', 'semantic-neural/model/config.json',
                 'semantic-neural/model/manifest.json', 'semantic-neural/model/tokenizer_config.json',
                 'semantic-neural/model/special_tokens_map.json', 'semantic-neural/model/vocab.txt']:
        path = root / name
        small[name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        if name.endswith('manifest.json') or name.endswith('/config.json'):
            small[name]['content'] = json.loads(path.read_text())
    bank = root / 'ontology-round2/frozen-fold0-sapbert/embeddings/19ab05a4a614bbc869be5c1d2943e08962c4e82f968026bcb41aa259e617d242.npy'
    print(json.dumps(dict(source_sha256=sha(__file__), roots=output, small_files=small,
                         selected_bank=dict(path=str(bank), sha256=sha(bank), bytes=bank.stat().st_size),
                         seconds=time.monotonic()-start, models_loaded=False,
                         embeddings_computed=False, gold_read=False)))


if __name__ == '__main__':
    main()
