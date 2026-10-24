"""Freeze two complete fixed F1 predictions before the only two v2 calls."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def relations(document):
    result = defaultdict(set)
    for row in document['association']:
        for value in row['phenotype']:
            parts = value.split(';') if re.fullmatch(r'HP:\d{7}(;HP:\d{7})+', value) else [value]
            result[str(row['patient_id'])].update(parts)
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--held-encoded', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--baseline-metrics', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--evaluation-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    assert sha(args.baseline) == 'f33a30f1f3c505022dbcf846a19e6496b3810627808e97cde595a9ed53d17866'
    assert sha(args.baseline_metrics) == '875292c3eb226b484ff86309cb92e5d28dca6f3857767481d5cb56cf9c68ee86'
    with gzip.open(args.held_encoded, 'rt') as stream:
        held = json.load(stream)
    assert len(held) == 16 and all('labels' not in doc for doc in held)
    expected = {tuple(key) for doc in held for key in doc['keys']}
    scores = read_jsonl(args.scores)
    assert len(scores) == len(expected) == 2874
    assert len({tuple(row['key']) for row in scores}) == len(scores)
    assert {tuple(row['key']) for row in scores} == expected
    positive = defaultdict(lambda: defaultdict(set))
    candidates = defaultdict(lambda: defaultdict(set))
    for row in scores:
        pmc, patient, concept = row['key']
        assert re.fullmatch(r'HP:\d{7}', concept)
        logit, probability = float(row['logit']), float(row['probability'])
        assert math.isfinite(logit) and 0 <= probability <= 1
        assert abs(probability - 1 / (1 + math.exp(-max(-80., min(80., logit))))) < 1e-8
        candidates[pmc][patient].add(concept)
        if logit > 0:
            positive[pmc][patient].add(concept)
    baseline = read_jsonl(args.baseline)
    assert {str(row['pmc_id']) for row in baseline} == {doc['pmc_id'] for doc in held}
    assert sum(len(doc['entities']) for doc in baseline) == 1189
    assert sum(len(doc['patient']) for doc in baseline) == 39
    predictions = {}
    for policy in ['union', 'candidate_replacement']:
        rows = []
        for original in baseline:
            pmc = str(original['pmc_id'])
            old = relations(original)
            changed = deepcopy(original)
            changed['association'] = []
            for patient in original['patient']:
                pid = str(patient['patient_id'])
                values = old[pid] | positive[pmc][pid] if policy == 'union' else (old[pid] - candidates[pmc][pid]) | positive[pmc][pid]
                changed['association'].append({'patient_id': patient['patient_id'], 'phenotype': sorted(values)})
            assert {k: v for k, v in changed.items() if k != 'association'} == {k: v for k, v in original.items() if k != 'association'}
            rows.append(changed)
        directory = output / policy
        directory.mkdir()
        path = directory / 'prediction.jsonl'
        with path.open('x') as stream:
            stream.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            stream.flush()
            os.fsync(stream.fileno())
        predictions[policy] = {'rows': rows, 'file': ref(path)}
    freeze = {'at_utc': datetime.now(timezone.utc).isoformat(), 'scores': ref(args.scores),
              'baseline': ref(args.baseline), 'baseline_metrics': ref(args.baseline_metrics),
              'predictions': {k: v['file'] for k, v in predictions.items()}, 'fixed_logit_threshold': 0,
              'documents': 16, 'patients': 39, 'entities': 1189, 'new_v2_calls_before_freeze': 0}
    write(output / 'freeze.json', freeze)
    evaluator_dir = Path(args.evaluation_dir)
    assert sha(evaluator_dir / 'evaluate.py') == 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    assert sha(evaluator_dir / 'validate_predictions.py') == 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f'
    module(evaluator_dir / 'validate_predictions.py', 'validate_predictions')
    evaluator = module(evaluator_dir / 'evaluate.py', 'fixed_public_evaluator')
    # First held gold access occurs after both complete prediction bodies and freeze.
    gold = evaluator.load_jsonl(args.gold)
    old_metrics = json.loads(Path(args.baseline_metrics).read_text())['metrics']
    results = {}
    for policy, entry in predictions.items():
        metrics = evaluator.evaluate_report(gold, entry['rows'], 'all', 'text')
        write(output / policy / 'metrics.json', metrics)
        results[policy] = {'prediction': entry['file'], 'metrics': metrics['metrics'],
                           'counts': metrics['counts'],
                           'delta': {key: metrics['metrics'][key] - old_metrics[key] for key in old_metrics}}
    improving = [name for name, row in results.items() if row['metrics']['Score'] > old_metrics['Score']]
    result = {'status': 'complete_fixed_training_internal_F1_control', 'baseline': old_metrics,
              'results': results, 'keep': bool(improving),
              'selected': max(improving, key=lambda name: results[name]['metrics']['Score']) if improving else None,
              'new_v2_calls': 2, 'old_v2_calls': 0, 'freeze': ref(output / 'freeze.json'),
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Original train64-internal F1; no original dev or official score. Fixed binary threshold, no sweep.'}
    write(output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
