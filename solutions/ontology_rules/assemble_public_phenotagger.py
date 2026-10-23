#!/usr/bin/env python3
"""Frozen native PhenoTagger additions; retain all baseline relationships."""
import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import run as baseline


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def assemble_document(doc, base, candidates, allowed):
    assert str(doc['pmc_id']) == str(base['pmc_id'])
    assert 'entities' not in doc and 'association' not in doc
    pred = copy.deepcopy(base)
    original = copy.deepcopy(base['entities'])
    counts = Counter()
    accepted = []
    for row in sorted(candidates, key=lambda r: (-r['score'], -r['length'], r['offset'], r['hpo_id'])):
        counts['native_candidates'] += 1
        assert str(row['pmc_id']) == str(doc['pmc_id'])
        passage = doc['full_text'][row['passage_index']]
        left, right = row['local_start'], row['local_end']
        assert 0 <= left < right <= len(passage['text'])
        assert passage['offset'] == row['passage_offset'] and passage['type'] == row['passage_type']
        assert row['offset'] == passage['offset'] + left
        assert row['length'] == right - left == len(row['text'])
        assert passage['text'][left:right] == row['text']
        assert math.isfinite(row['score']) and 0 <= row['score'] <= 1
        if row['hpo_id'] not in allowed or row['hpo_id'] == baseline.ROOT:
            counts['reject_disallowed'] += 1
            continue
        # The original native decoder already applied rounded-top1 > .95;
        # dictionary/abbreviation evidence is not given a new score threshold.
        if baseline.is_negated(passage['text'], left, right):
            counts['reject_negated'] += 1
            continue
        if any(row['offset'] < e['offset'] + e['length'] and e['offset'] < row['offset'] + row['length']
               for e in original):
            counts['reject_overlap_original'] += 1
            continue
        if any(row['offset'] < e['offset'] + e['length'] and e['offset'] < row['offset'] + row['length']
               for e in accepted):
            counts['reject_overlap_added'] += 1
            continue
        accepted.append({'identifier': row['hpo_id'], 'offset': row['offset'], 'length': row['length'],
                         'text': row['text'], 'type': 'Phenotype', 'note': None})
    pred['entities'].extend(accepted)
    pred['entities'].sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
    assert all(pred[k] == value for k, value in base.items() if k != 'entities')
    assert Counter(json.dumps(e, sort_keys=True) for e in pred['entities']) == (
        Counter(json.dumps(e, sort_keys=True) for e in original) +
        Counter(json.dumps(e, sort_keys=True) for e in accepted))
    counts['added_entities'] = len(accepted)
    counts['added_relations'] = 0
    return pred, counts


def main():
    p = argparse.ArgumentParser()
    for name in ['plan', 'entities', 'native-manifest', 'output-dir']:
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for entry in plan['local_inputs'].values():
        assert sha(entry['path']) == entry['sha256'], entry['path']
    assert sha(__file__) == plan['assembly_sources']['assembler']
    assert sha(baseline.__file__) == plan['assembly_sources']['baseline']
    native = json.loads(Path(args.native_manifest).read_text())
    native_plan = json.loads(Path(plan['local_inputs']['native_plan']['path']).read_text())
    assert native['status'] == 'complete_predictions_frozen_before_quality_evaluation'
    assert native['plan_sha256'] == plan['local_inputs']['native_plan']['sha256']
    assert native['input_sha256'] == plan['local_inputs']['input']['sha256']
    assert native['model_weight_sha256'] == native_plan['weight_sha256']
    assert native['source_sha256'] == native_plan['remote_source_sha256']['native-pipeline/run_native_entities.py']
    assert native['fixed_policy'] == native_plan['policy']
    assert native['public_model_training_on_task'] is False
    assert native['outputs']['canonical-entities.jsonl']['sha256'] == sha(args.entities)
    assert native['outputs']['canonical-entities.jsonl']['bytes'] == Path(args.entities).stat().st_size
    docs = read(plan['local_inputs']['input']['path'])
    base_rows = read(plan['local_inputs']['base']['path'])
    bases = {str(d['pmc_id']): d for d in base_rows}
    assert len(bases) == len(base_rows) == len(docs) == 16
    assert set(bases) == {str(d['pmc_id']) for d in docs}
    candidates = defaultdict(list)
    for row in read(args.entities):
        assert str(row['pmc_id']) in bases
        candidates[str(row['pmc_id'])].append(row)
    allowed = baseline.load_hpo(plan['local_inputs']['hpo']['path'])[1]
    output = []
    counts = Counter()
    for doc in docs:
        pred, current = assemble_document(doc, bases[str(doc['pmc_id'])], candidates[str(doc['pmc_id'])], allowed)
        output.append(pred)
        counts.update(current)
    assert counts['native_candidates'] == native['counts']['canonical_kept']
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'prediction.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
    manifest = {'gold_read': False, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'assembly_plan_sha256': sha(args.plan), 'sources': plan['assembly_sources'],
                'input_sha256': plan['local_inputs']['input']['sha256'],
                'base_sha256': plan['local_inputs']['base']['sha256'],
                'hpo_sha256': plan['local_inputs']['hpo']['sha256'],
                'native_manifest_sha256': sha(args.native_manifest), 'entities_sha256': sha(args.entities),
                'policy': plan['policy'], 'counts': dict(counts), 'documents': len(output),
                'all_base_entities_preserved': True, 'all_other_document_fields_equal': True,
                'prediction_sha256': sha(out / 'prediction.jsonl')}
    save(out / 'manifest.json', manifest)
    save(out / 'prediction-freeze.json', {'gold_read': False, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'prediction_sha256': manifest['prediction_sha256'], 'manifest_sha256': sha(out / 'manifest.json'),
        'base_sha256': manifest['base_sha256'], 'assembly_plan_sha256': sha(args.plan)})
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
