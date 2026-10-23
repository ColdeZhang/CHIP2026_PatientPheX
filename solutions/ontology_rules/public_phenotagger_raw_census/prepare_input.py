"""Project frozen old pools to keys/memberships only; no gold or scores used."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    started = time.monotonic()
    assert sha(a.old_manifest) == '87264eb85a1801e249ac6f052faeec5ddeb03fa3044ecb014ca3b9c8d6d220bf'
    parent = json.loads(Path(a.old_manifest).read_text())
    root = Path(a.output)
    root.mkdir(exist_ok=False, parents=True)
    memberships, pools = {}, []
    for fold in parent['folds']:
        for role in ('train', 'input'):
            spec = fold[role]
            assert sha(spec['path']) == spec['sha256']
            bit = 1 << len(pools)
            ids, n = [], 0
            with gzip.open(spec['path'], 'rt') as stream:
                for line in stream:
                    document = json.loads(line)
                    did = str(document['pmc_id'])
                    ids.append(did)
                    keys = [(did, c['offset'], c['length'], c['identifier']) for c in document['candidates']]
                    assert len(keys) == len(set(keys))
                    n += len(keys)
                    for key in keys:
                        assert not memberships.get(key, 0) & bit
                        memberships[key] = memberships.get(key, 0) | bit
            assert ids == fold['train_ids' if role == 'train' else 'input_ids']
            assert n == fold['old_train_rows' if role == 'train' else 'old_input_rows']
            pools.append(dict(index=len(pools), fold=fold['fold'], role=role, ids=ids, old_rows=n,
                              nested_input_ids=fold['nested_input_ids'] if role == 'train' else None,
                              source=spec))
    data = b''.join((json.dumps([*key, mask], separators=(',', ':')) + '\n').encode()
                    for key, mask in sorted(memberships.items()))
    dest = root / 'old-key-memberships.jsonl.gz'
    dest.write_bytes(gzip.compress(data, mtime=0))
    report = dict(status='label-free key membership projection', source_sha256=sha(__file__),
                  old_manifest=dict(path=a.old_manifest, sha256=sha(a.old_manifest)), pools=pools,
                  projection=dict(path=str(dest), sha256=sha(dest), bytes=dest.stat().st_size,
                                  rows=len(memberships), decompressed_sha256=hashlib.sha256(data).hexdigest()),
                  old_row_sum=sum(p['old_rows'] for p in pools),
                  fields_read=['pmc_id', 'candidates.offset', 'candidates.length', 'candidates.identifier'],
                  no_labels_or_scores_used=True, seconds=time.monotonic() - started)
    (root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('projection', 'old_row_sum', 'seconds')}))


if __name__ == '__main__':
    main()
