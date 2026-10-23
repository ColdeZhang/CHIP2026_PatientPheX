#!/usr/bin/env python3
"""One fixed union of two relation sets on identical documents/entities/patients.

No gold, score, confidence, threshold, or fallback selection is accepted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {str(row['pmc_id']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate document')
    return rows, result


def relations(row):
    result = {str(a['patient_id']): set(a['phenotype']) for a in row['association']}
    if len(result) != len(row['association']):
        raise ValueError('Duplicate patient')
    if any(not isinstance(value, str) for values in result.values() for value in values):
        raise ValueError('Non-string relation')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'rules', 'secondary', 'output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or out.with_suffix('.freeze.json').exists():
        raise ValueError('Use a new output path')
    inputs, imap = read(a.input)
    rules, rmap = read(a.rules)
    _, smap = read(a.secondary)
    if set(imap) != set(rmap) or set(imap) != set(smap):
        raise ValueError('Document coverage differs')
    outputs = []
    counts = dict(documents=0, patients=0, entities=0, rules=0, secondary=0, intersection=0, added=0, union=0)
    for original in rules:
        did = str(original['pmc_id']); other = smap[did]; inp = imap[did]
        if 'entities' in inp or 'association' in inp:
            raise ValueError('Annotated inference input forbidden')
        if original['entities'] != other['entities']:
            raise ValueError('Entity lists differ')
        r, s = relations(original), relations(other)
        patients = {str(patient['patient_id']) for patient in inp['patient']}
        if set(r) != patients or set(s) != patients:
            raise ValueError('Patient coverage differs')
        result = copy.deepcopy(original)
        for association in result['association']:
            pid = str(association['patient_id'])
            association['phenotype'] = sorted(r[pid] | s[pid])
            counts['rules'] += len(r[pid]); counts['secondary'] += len(s[pid])
            counts['intersection'] += len(r[pid] & s[pid]); counts['added'] += len(s[pid] - r[pid])
            counts['union'] += len(r[pid] | s[pid])
        counts['documents'] += 1; counts['patients'] += len(patients); counts['entities'] += len(original['entities'])
        if result['entities'] != original['entities']:
            raise AssertionError('Entity pool changed')
        outputs.append(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in outputs))
    manifest = {'policy': 'Unconditional set union; all original rule links retained; identical entity/patient pools.',
                'source_sha256': sha(__file__), 'configuration': vars(a),
                'inputs': {k: sha(getattr(a, k)) for k in ['input', 'rules', 'secondary']},
                'output_sha256': sha(out), 'counts_without_gold': counts,
                'heldout_gold_read': False, 'evaluation_performed': False,
                'development_control_not_blind_test': True}
    out.with_suffix('.freeze.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
