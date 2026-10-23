#!/usr/bin/env python3
"""Aggregate fixed-terminal error stages; never produces alternative predictions.

This diagnostic consumes held-out labels only after the fixed terminal prediction
and its evaluation have been frozen. Top3 presence is a capacity diagnostic, not
an oracle score or a proposal to change the threshold.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import run as baseline
import enhance
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import evaluate


def read(path):
    return json.loads(Path(path).read_text())


def overlapping(a, b):
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def wildcard_matched(truth, guesses):
    # Same exact-ID-first, then one remaining ID semantics as evaluator v2.
    return '-1' in truth and ('-1' in guesses or bool(guesses - truth))


def main():
    ap = argparse.ArgumentParser()
    for key in ['input', 'gold', 'base', 'pred', 'scored', 'training-data',
                'training-complete', 'prediction-manifest', 'metrics', 'hpo', 'output']:
        ap.add_argument('--' + key, required=True)
    args = ap.parse_args()
    terminal = read(args.training_complete)
    assert terminal['epochs'] == 100 and terminal['global_step'] == 45800
    scored_manifest = read(args.prediction_manifest)
    assert scored_manifest['terminal'] == terminal
    assert scored_manifest['output_sha256'] == baseline.sha256(args.scored)
    metrics = read(args.metrics)
    assert metrics['prediction_sha256'] == baseline.sha256(args.pred)
    assert metrics['gold_sha256'] == baseline.sha256(args.gold)
    assert metrics['evaluator_version'] == evaluate.VERSION
    assert metrics['configuration']['body_scope'] == 'all'
    training = Path(args.training_data); train_manifest = read(training / 'manifest.json')
    assert baseline.sha256(training / 'rows.jsonl.gz') == train_manifest['outputs']['rows.jsonl.gz']
    assert baseline.sha256(training / 'labels.json') == train_manifest['outputs']['labels.json']
    assert baseline.sha256(args.hpo) == train_manifest['hpo_sha256']
    labels = set(read(training / 'labels.json')) - {'HP:None'}
    allowed = baseline.load_hpo(args.hpo)[1] - {baseline.ROOT}
    docs = {str(d['pmc_id']): d for d in baseline.read_jsonl(args.input)}
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    assert set(docs).isdisjoint(map(str, train_manifest['train_ids']))
    bases = {str(d['pmc_id']): d for d in baseline.read_jsonl(args.base)}
    predictions = {str(d['pmc_id']): d for d in baseline.read_jsonl(args.pred)}
    golds = {str(d['pmc_id']): d for d in baseline.read_jsonl(args.gold)}
    assert set(docs) == set(bases) == set(predictions) == set(golds)
    alias = defaultdict(set); negative_surfaces = set()
    with gzip.open(training / 'rows.jsonl.gz', 'rt') as handle:
        for row in map(json.loads, handle):
            if row['label'] == 'HP:None': negative_surfaces.add(row['text'])
            else: alias[row['text']].add(row['label'])
    assert not (set(alias) & negative_surfaces)
    candidates = {}
    with gzip.open(args.scored, 'rt') as handle:
        for row in map(json.loads, handle):
            key = (str(row['pmc_id']), row['offset'], row['length'])
            assert key not in candidates
            _, passage, start, end = enhance.location(docs[key[0]], row['offset'], row['offset'] + row['length'])
            assert passage['text'][start:end] == row['text']
            assert row['top3'][0][0] == row['identifier'] and row['top3'][0][1] == row['probability']
            candidates[key] = row
    stages = Counter(); all_units = Counter(); wildcard = Counter()
    strata = defaultdict(Counter); errors = Counter(); raw = Counter()
    for did, doc in docs.items():
        truth = evaluate.positive_units(golds[did], doc, 'all')
        base = evaluate.positive_units(bases[did], doc, 'all')
        final = evaluate.positive_units(predictions[did], doc, 'all')
        for span, ids in base.items(): assert ids <= final.get(span, set())
        base_entities = [(e['offset'], e['length']) for e in bases[did]['entities']]
        new_spans = [span for span, ids in final.items() if ids - base.get(span, set())]
        for entity in golds[did]['entities']:
            if str(entity.get('note') or '').upper() != 'NO':
                raw['positive_records'] += 1
                raw['atomic_ids_before_deduplication'] += len(entity['identifier'].split(';'))
        raw['unique_positive_spans'] += len(truth)
        raw['multi_id_positive_spans'] += sum(len(ids) > 1 for ids in truth.values())
        for span, ids in truth.items():
            row = candidates.get((did, *span))
            if '-1' in ids:
                wildcard['gold_units'] += 1
                wildcard['physical_span_in_pool'] += row is not None
                wildcard['base_matched'] += wildcard_matched(ids, base.get(span, set()))
                wildcard['final_matched'] += wildcard_matched(ids, final.get(span, set()))
            for identifier in ids - {'-1'}:
                all_units['gold_linked_atomic_units'] += 1
                all_units['model_label_supported'] += identifier in labels
                all_units['physical_span_in_pool'] += row is not None
                if row:
                    correct = row['identifier'] == identifier
                    all_units['top1_correct'] += correct
                    all_units['top3_contains_correct'] += identifier in [i for i, _ in row['top3']]
                    all_units['top1_correct_and_p95'] += correct and row['probability'] >= .95
                if identifier in base.get(span, set()):
                    stages['already_covered_by_base'] += 1
                    continue
                if not row:
                    stage = 'missing_exact_span_in_pool'; seen = 'no_candidate_surface'
                else:
                    surface = row['model_text']
                    seen = ('seen_positive_correct_id' if identifier in alias[surface] else
                            'seen_positive_other_id_only' if alias[surface] else
                            'seen_weak_negative_only' if surface in negative_surfaces else 'unseen_surface')
                    if identifier not in labels or identifier not in allowed:
                        stage = 'target_outside_model_or_output_labels'
                    elif row['identifier'] != identifier:
                        stage = ('wrong_top1_correct_in_top3' if identifier in [i for i, _ in row['top3']]
                                 else 'wrong_top1_correct_not_in_top3')
                    elif row['probability'] < .95:
                        stage = 'correct_top1_below_fixed_p95'
                    elif row['overlength']:
                        stage = 'correct_confident_but_overlength'
                    else:
                        _, passage, start, end = enhance.location(doc, span[0], span[0] + span[1])
                        if baseline.is_negated(passage['text'], start, end):
                            stage = 'correct_confident_but_negation_filter'
                        elif any(overlapping(span, other) for other in base_entities):
                            stage = 'correct_confident_but_base_overlap'
                        elif identifier in final.get(span, set()):
                            stage = 'new_exact_unit_retained'
                        elif any(overlapping(span, other) for other in new_spans):
                            stage = 'correct_confident_but_new_span_competition'
                        else:
                            stage = 'unexplained_assembly_discrepancy'
                            errors[stage] += 1
                stages[stage] += 1; strata[seen][stage] += 1
        # Independently match final additions using exact IDs, then -1 wildcard.
        for span, final_ids in final.items():
            known = set(truth.get(span, set())); matched = final_ids & known
            rest = final_ids - matched
            if '-1' in known - matched and rest: matched.add(sorted(rest)[0])
            for identifier in final_ids - base.get(span, set()):
                row = candidates[(did, *span)]
                assert row['identifier'] == identifier
                key = 'added_tp' if identifier in matched else 'added_fp'
                errors[key] += 1
    assert sum(stages.values()) == all_units['gold_linked_atomic_units']
    assert errors['unexplained_assembly_discrepancy'] == 0
    result = {'policy': 'Fixed terminal100 aggregate diagnosis only; no alternate predictions/thresholds/epochs/scores',
              'source_sha256': baseline.sha256(__file__),
              'input_sha256': {k: baseline.sha256(getattr(args, k)) for k in
                  ['input', 'gold', 'base', 'pred', 'scored', 'training_complete', 'prediction_manifest', 'metrics', 'hpo']},
              'training_rows_sha256': train_manifest['outputs']['rows.jsonl.gz'],
              'raw_and_deduplicated_counts': dict(raw), 'all_linked_unit_diagnostics': dict(all_units),
              'mutually_exclusive_linked_stages': dict(stages),
              'baseline_missing_linked_by_training_surface': {k: dict(v) for k, v in sorted(strata.items())},
              'unlinked_gold_wildcard_separate': dict(wildcard), 'addition_match_check': dict(errors),
              'fixed_result_metrics': metrics['metrics'],
              'interpretation': ['Gold linked units deduplicate exact(doc,offset,length,splitID); -1 handled separately by evaluator exact-ID-first wildcard convention.',
                 'Top3 presence indicates candidate capacity only; it is not a model score, oracle prediction or threshold-selection result.',
                 'Seen surface means exact frozen normalized model_text present in the actual training rows, not a semantic synonym claim.',
                 'Stage causes describe this frozen pipeline; generic unseen language may involve normalization, boundary and concept-specific ambiguity.',
                 'No document IDs, individual error strings, rewritten predictions or new evaluation scores are emitted.']}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'stages': dict(stages), 'wildcard': dict(wildcard), 'addition_match_check': dict(errors)}))


if __name__ == '__main__': main()
