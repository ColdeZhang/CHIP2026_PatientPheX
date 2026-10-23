#!/usr/bin/env python3
"""Union two attributed relation predictions while keeping primary entities."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {row['pmc_id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def main():
    ap = argparse.ArgumentParser()
    for name in ['primary', 'secondary', 'output']:
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); primary = read(a.primary); secondary = read(a.secondary)
    assert set(primary) == set(secondary)
    output = []; added = 0
    for did, pred in primary.items():
        first = {r['patient_id']: set(r['phenotype']) for r in pred['association']}
        second = {r['patient_id']: set(r['phenotype']) for r in secondary[did]['association']}
        assert set(first) == set(second)
        added += sum(len(second[pid] - values) for pid, values in first.items())
        output.append(dict(pred, association=[{'patient_id': pid, 'phenotype': sorted(values | second[pid])}
                                               for pid, values in first.items()]))
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
    manifest = {'config': vars(a), 'source_sha256': sha(__file__),
                'primary_sha256': sha(a.primary), 'secondary_sha256': sha(a.secondary),
                'entity_outputs_unchanged': True, 'added_relations': added, 'output_sha256': sha(dest)}
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__': main()
