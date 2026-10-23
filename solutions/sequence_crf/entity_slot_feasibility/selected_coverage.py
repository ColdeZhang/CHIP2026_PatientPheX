#!/usr/bin/env python3
"""Input-only score coverage of already selected F1 entities; no decode/eval/fit."""
from pathlib import Path
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
import time


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in data.splitlines()]


def key(pmc, row):
    return str(pmc), row['offset'], row['length'], row['identifier']


def quantiles(values):
    values = sorted(values)
    assert values and all(math.isfinite(x) for x in values)
    # Nearest rank; these are descriptive summaries, not decoder cut selection.
    return {str(q): values[max(0, math.ceil(q * len(values)) - 1)]
            for q in [0, .25, .5, .75, .95, 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    started = time.monotonic()
    route = a.root / 'sequence_crf'
    formal = route / 'entity-slot-formal32-v1'
    gate = route / 'entity-slot-feasibility/gate-p01'
    public = route / 'public-bioformer-selector-control/fold1/fitted'
    nested = route / 'selector-nested-control/fold1'
    paths = dict(public_manifest=public / 'manifest.json',
                 public_scores=public / 'selector-scores.jsonl',
                 raw_cut3_pred=route / 'public-bioformer-cut-calibration/cut-3/fold1.jsonl',
                 retained_nested_pred=nested / 'nested.jsonl',
                 raw_cut3_cached_metrics=nested / 'baseline.metrics.json',
                 retained_nested_cached_metrics=nested / 'nested.metrics.json',
                 final_scores=formal / 'terminal-infer/inference/final.probabilities.jsonl',
                 gate_train_scores=gate / 'train.gate-scores.jsonl.gz',
                 gate_F1_scores=gate / 'heldout.gate-scores.jsonl.gz',
                 gate_census=gate / 'census.json',
                 strict_freeze=route / 'entity-slot-feasibility/input-only-freeze.json')
    meta, freeze = load(paths['public_manifest']), load(paths['strict_freeze'])
    public_ref = load(meta['arguments']['reference'])
    # Read document IDs only; neither entities nor association labels are used.
    public_train = {str(r['pmc_id']) for r in rows(public_ref['config']['train'])}
    train32, held16 = set(freeze['train_ids']), set(freeze['heldout_ids'])
    assert len(public_train) == 48 and len(train32) == 32 and len(held16) == 16
    assert train32 <= public_train and not public_train & held16
    paths['public_source_reference'] = Path(meta['arguments']['reference'])
    paths['public_training_ids_source'] = Path(public_ref['config']['train'])
    pool = {}
    for doc in rows(paths['public_scores']):
        for c in doc['candidates']:
            k = key(doc['pmc_id'], c)
            assert k not in pool
            pool[k] = c
    legacy = {key(r['pmc_id'], r): r for r in rows(paths['gate_F1_scores'])}
    legacy_spans = {k[:3] for k in legacy}
    scores = {}
    for r in rows(paths['final_scores']):
        assert len(r['keys']) == len(r['probabilities'])
        for k, p in zip(r['keys'], r['probabilities']):
            k = (str(r['pmc_id']), *k)
            assert k not in scores and math.isfinite(p) and 0 <= p <= 1
            scores[k] = p
    assert set(scores) == {k for k, r in legacy.items() if r['retained']}
    assert len(scores) == 1589
    public_flags = ['public_present', 'bioformer_present']
    old_flags = ['semantic_source', 'crf', 'is_base']
    selected = {}
    for arm in ['raw_cut3', 'retained_nested']:
        docs = rows(paths[arm + '_pred'])
        keys = [key(d['pmc_id'], e) for d in docs for e in d['entities']]
        assert len(keys) == len(set(keys)) and set(keys) <= set(pool)
        assert {str(d['pmc_id']) for d in docs} == held16
        counts = collections.Counter()
        by_doc = collections.defaultdict(collections.Counter)
        combinations = collections.Counter()
        for k in keys:
            counts['selected'] += 1
            by_doc[k[0]]['selected'] += 1
            if k in scores:
                status = 'covered_below_point5' if scores[k] < .5 else 'covered_at_or_above_point5'
                counts['covered'] += 1
                by_doc[k[0]]['covered'] += 1
            elif k in legacy:
                assert legacy[k]['gate_probability'] < .1 and not legacy[k]['retained']
                status = 'uncovered_legacy_key_below_gate'
            else:
                status = ('uncovered_legacy_same_span_other_id' if k[:3] in legacy_spans
                          else 'uncovered_legacy_span_absent')
            counts[status] += 1
            by_doc[k[0]][status] += 1
            if k not in scores:
                c = pool[k]
                flags = [f for f in public_flags + old_flags if c.get(f, False)]
                combinations['+'.join(flags) or 'no_recorded_flag'] += 1
                if any(c.get(f, False) for f in public_flags):
                    counts['uncovered_public_flag'] += 1
                if (any(c.get(f, False) for f in public_flags)
                        and not any(c.get(f, False) for f in old_flags)):
                    counts['uncovered_public_only_recorded_source'] += 1
                if k not in legacy and any(c.get(f, False) for f in public_flags):
                    counts['uncovered_public_flag_and_absent_legacy_key'] += 1
        metrics = load(paths[arm + '_cached_metrics'])['metrics']
        selected[arm] = dict(counts=dict(counts), coverage=counts['covered'] / len(keys),
                            uncovered_source_combinations=dict(combinations),
                            by_document={k: dict(v) for k, v in sorted(by_doc.items())},
                            cached_metrics_only=metrics,
                            association_objects=sum(len(d['association']) for d in docs))
    dist = {}
    for split in ['train', 'F1']:
        rs = rows(paths['gate_' + split + '_scores'])
        ps = [r['gate_probability'] for r in rs]
        retained = [r for r in rs if r['retained']]
        assert all(r['retained'] == (r['gate_probability'] >= .1) for r in rs)
        dist[split] = dict(candidates=len(rs), documents=len({r['pmc_id'] for r in rs}),
                           retained=len(retained), retained_fraction=len(retained) / len(rs),
                           all_gate_quantiles=quantiles(ps),
                           retained_gate_quantiles=quantiles([r['gate_probability'] for r in retained]),
                           retained_length_quantiles=quantiles([r['length'] for r in retained]),
                           retained_nil=sum(r['identifier'] == '-1' for r in retained))
    census = load(paths['gate_census'])
    output = dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source=ref(__file__), references={k: ref(p) for k, p in paths.items()},
                  scope='Existing prediction/score coverage and cached metric lookup only; no new predictions or evaluation.',
                  training_scope=dict(public_HGB=48, public_natural_features=len(meta['features']),
                                      entity_binary=32, extra_public_train_documents=len(public_train-train32),
                                      both_exclude_F1=True, same_training_resources=False,
                                      cut_point3_was_pooled64_selected=True),
                  selected=selected, input_only_distribution=dist,
                  cached_prompt_census={s: census[s] for s in ['train', 'heldout']},
                  cached_training_diagnostic=census['training_diagnostic'],
                  next_control_not_executed='One possible fixed .5 veto on already selected, exactly scored entities only; all uncovered entities and nonentities retained. No prediction produced.',
                  limitations=[
                      'Current family is often called97; this F1 fit naturally has95 features. Binary77 gate is a different32-document resource.',
                      'Public flags can overlap old sources; absent legacy key is not by itself proof of a public-only origin.',
                      'Train gate is resubstitution. Input distribution shifts do not establish F1 class prevalence, calibration failure or causal overfitting.',
                      'No F1 annotation file is opened. Cached already-evaluated metrics are only copied; training-ID file is parsed only for IDs.',
                      'Rawcut3 and retained-nesting predictions are both reported to avoid silently choosing a weaker baseline.'
                  ], no_new_quality=True, no_fit=True, no_GPU=True, no_cut_scan=True,
                  seconds=time.monotonic()-started)
    with a.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: output[k] for k in ['training_scope', 'selected', 'input_only_distribution', 'seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
