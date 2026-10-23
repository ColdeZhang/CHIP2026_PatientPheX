#!/usr/bin/env python3
"""Keep candidate entities but restore the complete frozen baseline associations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'candidate', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    read = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
    bases = read(args.base)
    candidates = read(args.candidate)
    base_map = {d['pmc_id']: d for d in bases}
    candidate_map = {d['pmc_id']: d for d in candidates}
    assert len(base_map) == len(bases) == len(candidate_map) == len(candidates)
    assert set(base_map) == set(candidate_map)
    outputs = []
    added = 0
    for base in bases:
        candidate = candidate_map[base['pmc_id']]
        assert all(e in candidate['entities'] for e in base['entities'])
        assert len({(e['offset'], e['length'], e['identifier']) for e in candidate['entities']}) == len(candidate['entities'])
        for key in set(base) | set(candidate):
            if key not in ('entities', 'association'):
                assert base.get(key) == candidate.get(key), key
        result = copy.deepcopy(base)
        result['entities'] = copy.deepcopy(candidate['entities'])
        assert result['association'] == base['association']
        added += len(result['entities']) - len(base['entities'])
        outputs.append(result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in outputs))
    manifest = {'source_sha256': sha(__file__),
                'inputs': {name: {'path': str(Path(getattr(args, name)).resolve()),
                                  'sha256': sha(getattr(args, name))}
                           for name in ('base', 'candidate')},
                'documents': len(outputs), 'added_entities': added,
                'all_base_entities_preserved': True, 'all_base_associations_exact': True,
                'gold_used': False, 'output_sha256': sha(out)}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
