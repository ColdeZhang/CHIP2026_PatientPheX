"""Full-input mechanical raw-supply census plus separate frozen FN stage audit."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(spec):
    h, count = hashlib.sha256(), 0
    with Path(spec['path']).open('rb') as stream:
        for line in stream:
            h.update(line)
            count += 1
            yield json.loads(line)
    assert h.hexdigest() == spec['sha256'] and Path(spec['path']).stat().st_size == spec['bytes']
    if 'rows' in spec:
        assert count == spec['rows']


def main():
    started = time.monotonic()
    signal.alarm(115)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    plan = json.loads(Path('plan.json').read_text())
    assert sha(__file__) == plan['source_sha256']
    out = Path('output')
    out.mkdir(exist_ok=False)
    for spec in plan['resources'].values():
        assert sha(spec['path']) == spec['sha256']
    root = Path(plan['public_resource_root'])
    docs = {str(d['pmc_id']): d for d in map(json.loads, (root / 'train64-cache/train64.input.jsonl').read_text().splitlines())}
    assert len(docs) == 64 and all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs.values())
    labels = (root / 'lable.vocab').read_text().strip().splitlines()
    mapping = json.loads((root / 'original-column-to-current-HPO.json').read_text())['mapping']
    assert [r['original_label'] for r in mapping] == labels
    cmap = {r['original_label']: r['current_label'] for r in mapping}
    projection = json.loads(Path('projection-manifest.json').read_text())
    assert sha('old-key-memberships.jsonl.gz') == projection['projection']['sha256']
    data = gzip.decompress(Path('old-key-memberships.jsonl.gz').read_bytes())
    assert hashlib.sha256(data).hexdigest() == projection['projection']['decompressed_sha256']
    old = [set() for _ in projection['pools']]
    for line in data.splitlines():
        did, offset, length, ident, mask = json.loads(line)
        assert did in docs
        for index, keys in enumerate(old):
            if mask & (1 << index):
                keys.add((did, offset, length, ident))
    assert [len(s) for s in old] == [p['old_rows'] for p in projection['pools']]
    # All three policies are fixed before reading diagnostic queries. C reuses
    # the original threshold exactly; it is not a newly chosen gate.
    policy_names = ('raw_top1_valid', 'raw_top3_valid', 'raw_top1_original95_valid')
    policy = {name: set() for name in policy_names}
    per_model, all_spans = {}, set()
    scan_start = time.monotonic()
    for model, specs in plan['neural_raw'].items():
        local = {name: set() for name in policy_names}
        count = 0
        for spec in specs:
            for row in rows(spec):
                count += 1
                did = str(row['pmc_id'])
                passage = docs[did]['full_text'][row['passage_index']]
                assert passage['offset'] + row['local_start'] == row['offset']
                assert row['local_end'] - row['local_start'] == row['length']
                assert passage['text'][row['local_start']:row['local_end']] == row['text']
                span = (did, row['offset'], row['length'])
                all_spans.add(span)
                top = row['top3']
                assert len(top) == 3 and all(top[j]['probability'] >= top[j + 1]['probability'] for j in range(2))
                for rank, cell in enumerate(top, 1):
                    assert labels[cell['column']] == cell['hpo_id']
                    assert math.isfinite(cell['probability']) and 0 <= cell['probability'] <= 1
                    ident = cmap.get(cell['hpo_id'])
                    if not ident or cell['hpo_id'] == 'HP:None':
                        continue
                    key = (*span, ident)
                    local['raw_top3_valid'].add(key)
                    if rank == 1:
                        local['raw_top1_valid'].add(key)
                        if float(cell['native_probability_5dp']) > .95:
                            local['raw_top1_original95_valid'].add(key)
        assert count == 296984
        per_model[model] = {name: dict(keys=len(keys), spans=len({key[:3] for key in keys})) for name, keys in local.items()}
        for name, keys in local.items():
            policy[name].update(keys)
    raw_scan_seconds = time.monotonic() - scan_start
    assert policy['raw_top1_original95_valid'] <= policy['raw_top1_valid'] <= policy['raw_top3_valid']
    global_old = set.union(*old)
    old_spans = {key[:3] for key in global_old}
    census = {}
    for name, keys in policy.items():
        spans = {key[:3] for key in keys}
        folds = []
        for pool, base in zip(projection['pools'], old):
            ids = set(pool['ids'])
            supplied = {key for key in keys if key[0] in ids}
            new_keys = supplied - base
            base_spans = {key[:3] for key in base}
            supplied_spans = {key[:3] for key in supplied}
            nested = []
            for group in pool['nested_input_ids'] or []:
                g = set(group)
                bn = sum(key[0] in g for key in base)
                add = sum(key[0] in g for key in new_keys)
                nested.append(dict(documents=len(g), old_rows=bn, added_keys=add, total_rows=bn + add, ratio=(bn + add) / bn))
            folds.append(dict(fold=pool['fold'], role=pool['role'], old_rows=len(base),
                              supplied_keys=len(supplied), added_keys=len(new_keys), merged_rows=len(base) + len(new_keys),
                              row_ratio=(len(base) + len(new_keys)) / len(base),
                              added_spans=len(supplied_spans - base_spans), nested_groups=nested))
        total = sum(x['merged_rows'] for x in folds)
        census[name] = dict(raw_union_keys=len(keys), raw_union_spans=len(spans),
            global_old_keys=len(global_old), global_old_spans=len(old_spans),
            added_global_keys=len(keys - global_old), added_global_spans=len(spans - old_spans),
            merged_global_keys=len(keys | global_old), folded_old_rows=projection['old_row_sum'],
            folded_merged_rows=total, maximum_fold_train_ratio=max(x['row_ratio'] for x in folds if x['role'] == 'train'),
            maximum_nested_ratio=max(g['ratio'] for x in folds for g in x['nested_groups']),
            old50_2126second_fit_linear_proxy=50.21263894800359 * total / projection['old_row_sum'], folds=folds)
    census_seconds = time.monotonic() - started
    # Save the full-input census BEFORE loading the existing gold-indexed
    # diagnostic. Nothing from that diagnostic selects the policies or keys.
    preliminary = dict(status='input-only mechanical census complete; no candidate file or fit',
        source_sha256=sha(__file__), plan_sha256=sha('plan.json'), policies=census, per_model=per_model,
        all_raw_spans=len(all_spans), raw_scan_seconds=raw_scan_seconds, census_seconds=census_seconds,
        independent_of_query_gold=True)
    (out / 'census.json').write_text(json.dumps(preliminary, indent=2) + '\n')

    ds = plan['stage_diagnostic']
    for name in ('queries', 'audit'):
        assert sha(ds[name]['path']) == ds[name]['sha256']
    queries = json.loads(Path(ds['queries']['path']).read_text())['queries']
    audit = json.loads(Path(ds['audit']['path']).read_text())
    assert len(queries) == len(audit) == 384
    stages = {}
    for model, specs in ds['stage_files'].items():
        stage_map = {}
        for spec in specs:
            for row in rows(spec):
                key = (str(row['pmc_id']), row['passage_index'])
                assert key not in stage_map
                stage_map[key] = row
        stages[model] = stage_map
    records, totals, correct_totals = [], defaultdict(Counter), defaultdict(Counter)
    for q, item in zip(queries, audit):
        if not item['generated_coordinate_count']:
            continue
        for model in ('pubmed', 'bioformer'):
            assert len(item['raw_rows'][model]) == 1
            top = item['raw_rows'][model][0]['top3'][0]
            ident, prob = top['hpo_id'], float(top['native_probability_5dp'])
            key = (q['local_start'], q['local_end'], ident)
            stage = stages[model][q['pmc_id'], q['passage_index']]
            before = {(int(x[0]), int(x[1]), x[2]) for x in stage['before_abbreviation']}
            after = {(int(x[0]), int(x[1]), x[2]) for x in stage['after_abbreviation']}
            if ident == 'HP:None':
                why = 'top1_None'
            elif prob <= .95:
                why = 'original_rounded_probability_gate'
            elif key not in before:
                why = 'native_hybrid_overlap_selection'
            elif key not in after:
                why = 'abbreviation_stage_removed_or_changed'
            elif not cmap.get(ident):
                why = 'current_ID_mapping_invalid'
            else:
                raise AssertionError(('valid raw surviving all stages must be canonical', q['query_index'], model))
            correct_literal = q['gold_literal_id'] != '-1' and cmap.get(ident) == q['gold_literal_id']
            totals[model][why] += 1
            correct_totals[model][why] += int(correct_literal)
            records.append(dict(query_index=q['query_index'], model=model, stage=why,
                                top1_current=cmap.get(ident), top1_raw=ident, original_5dp=prob,
                                before_exact=key in before, after_exact=key in after,
                                correct_linked_literal=correct_literal))
    assert all(sum(totals[m].values()) == 217 for m in ('pubmed', 'bioformer'))
    stage_report = dict(status='conditional diagnostic of previous 217 atoms only; does not alter census',
                        atoms_per_model=217, totals={m: dict(v) for m, v in totals.items()},
                        literal_linked_correct_top1_by_stage={m: dict(v) for m, v in correct_totals.items()},
                        records=records, parent_audit_sha256=ds['audit']['sha256'])
    (out / 'stage-diagnostic.json').write_text(json.dumps(stage_report, indent=2) + '\n')
    result = dict(status='complete no-fit CPU census and separate prior-FN stage audit',
        source_sha256=sha(__file__), plan_sha256=sha('plan.json'), completed_at_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - started, census=preliminary,
        stage_totals=stage_report['totals'], stage_literal_correct=stage_report['literal_linked_correct_top1_by_stage'],
        outputs={n: dict(sha256=sha(out / n), bytes=(out / n).stat().st_size) for n in ('census.json', 'stage-diagnostic.json')},
        no_new_NN=True, no_GPU=True, no_fit=True, no_originaldev=True, no_candidate_or_prediction_written=True,
        limitations=plan['limitations'])
    assert result['seconds'] < 120
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], policies={n: {k: v for k, v in r.items() if k != 'folds'} for n, r in census.items()}, stage_totals=result['stage_totals'])))


if __name__ == '__main__':
    main()
