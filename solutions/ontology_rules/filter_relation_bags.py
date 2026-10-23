#!/usr/bin/env python3
"""Fixed .5 doc-HPO deletion only; preserve entities, assignments and NIL."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import run as ontology


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ['base', 'hpo', 'scores', 'output']:
        ap.add_argument('--'+k, required=True)
    args = ap.parse_args(); _, allowed, _, _ = ontology.load_hpo(args.hpo)
    base = ontology.read_jsonl(args.base)
    expected = {(str(d['pmc_id']), h) for d in base for e in d['entities']
                for h in str(e['identifier']).split(';') if h in allowed}
    scores = {}
    for row in ontology.read_jsonl(args.scores):
        key = (str(row['pmc_id']), row['concept'])
        assert key not in scores and math.isfinite(row['probability']) and 0 <= row['probability'] <= 1
        scores[key] = row['probability']
    assert set(scores) == expected
    output, removed = copy.deepcopy(base), []
    for doc in output:
        did = str(doc['pmc_id'])
        for association in doc['association']:
            keep = []
            for value in association['phenotype']:
                parts = str(value).split(';')
                # Preserve NIL/natural text/out-of-scope values exactly.
                if not all(p in allowed for p in parts):
                    keep.append(value); continue
                retain = [p for p in parts if scores.get((did, p), 1.) >= .5]
                if retain == parts:
                    keep.append(value)
                else:
                    keep.extend(retain)
                    removed.extend((did, association['patient_id'], p) for p in parts if p not in retain)
            association['phenotype'] = keep
    for before, after in zip(base, output):
        assert before['entities'] == after['entities']
        assert [r['patient_id'] for r in before['association']] == [r['patient_id'] for r in after['association']]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if removed:
        out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output))
    else:
        out.write_bytes(Path(args.base).read_bytes())
    result = {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'ontology_source_sha256': sha(ontology.__file__),
              'inputs': {k: sha(getattr(args, k)) for k in ['base', 'hpo', 'scores']},
              'threshold': .5, 'scored_bags': len(scores), 'negative_bags': sum(v < .5 for v in scores.values()),
              'removed_relations': len(removed), 'entities_unchanged': True, 'new_relations': 0,
              'prediction_sha256': sha(out), 'gold_read': False}
    out.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
