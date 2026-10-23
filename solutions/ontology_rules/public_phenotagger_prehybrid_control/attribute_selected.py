"""Posthoc counts of frozen .3 predictions; no fit, decode or new predictions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def checked(item):
    actual = ref(item['path'])
    assert actual['sha256'] == item['sha256'], item['path']
    if 'bytes' in item:
        assert actual['bytes'] == item['bytes']
    return Path(item['path'])


def rows(item):
    path = checked(item)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        result = [json.loads(line) for line in stream]
    assert len({r['pmc_id'] for r in result}) == len(result)
    return {r['pmc_id']: r for r in result}


def key(entity):
    return entity['offset'], entity['length'], entity['identifier']


def nonoverlap(entities):
    ordered = sorted(entities, key=lambda e: (e['offset'], e['length']))
    assert all(a['offset'] + a['length'] <= b['offset'] for a, b in zip(ordered, ordered[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    assert ref(__file__)['sha256'] == plan['source_sha256']
    out = Path(plan['output_dir'])
    assert not out.exists()
    evaluator_path = checked(plan['evaluator'])
    import sys
    sys.path.insert(0, str(evaluator_path.parent))
    spec = importlib.util.spec_from_file_location('evaluate', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    frozen = json.loads(checked(plan['complete_freeze']).read_text())
    old_manifest = json.loads(checked(plan['old_manifest']).read_text())
    prepared = json.loads(checked(plan['prepared_manifest']).read_text())
    results = json.loads(checked(plan['results']).read_text())
    gold = rows(plan['train64'])
    total = Counter(); folds = []; audit = []; seen = set()
    for frozen_fold, old_fold, new_fold in zip(frozen['folds'], old_manifest['folds'], prepared['folds']):
        index = frozen_fold['fold']
        assert index == old_fold['fold'] == new_fold['fold']
        old_pool, new_pool = rows(old_fold['input']), rows(new_fold['input'])
        before, after = rows(frozen_fold['baseline_prediction']), rows(frozen_fold['new_prediction'])
        assert old_pool.keys() == new_pool.keys() == before.keys() == after.keys()
        assert len(before) == 16 and not seen.intersection(before)
        seen.update(before)
        counts = Counter()
        for docid in before:
            truth = gold[docid]
            old_cs, new_cs = old_pool[docid]['candidates'], new_pool[docid]['candidates']
            assert new_cs[:len(old_cs)] == old_cs
            old_keys = {key(c) for c in old_cs}
            all_keys = {key(c) for c in new_cs}
            appended = all_keys - old_keys
            assert len(all_keys) == len(new_cs) and len(old_keys) == len(old_cs)
            counts['held_old_pool_keys'] += len(old_keys)
            counts['held_append_keys'] += len(appended)
            old_ents = {key(e): e for e in before[docid]['entities']}
            new_ents = {key(e): e for e in after[docid]['entities']}
            assert len(old_ents) == len(before[docid]['entities'])
            assert len(new_ents) == len(after[docid]['entities'])
            assert old_ents.keys() <= old_keys and new_ents.keys() <= all_keys
            assert {k:v for k,v in before[docid].items() if k != 'entities'} == {k:v for k,v in after[docid].items() if k != 'entities'}
            # Thus disjoint subsets below have no competing same-span NIL supply.
            nonoverlap(list(old_ents.values())); nonoverlap(list(new_ents.values()))
            groups = {
                'append_selected': new_ents.keys() & appended,
                'common_added': (new_ents.keys() - old_ents.keys()) & old_keys,
                'common_removed': old_ents.keys() - new_ents.keys(),
                'common_retained': old_ents.keys() & new_ents.keys(),
            }
            gu = evaluator.positive_units(truth, truth, 'all')
            for group, keys in groups.items():
                entities = old_ents if group == 'common_removed' else new_ents
                subset = dict(after[docid], entities=[entities[k] for k in sorted(keys)])
                tp, fp, _ = evaluator.mention_counts(gu, evaluator.positive_units(subset, truth, 'all'))
                counts[group + '_records'] += len(keys)
                counts[group + '_tp'] += tp
                counts[group + '_fp'] += fp
                if group != 'common_retained':
                    for k in sorted(keys):
                        e = entities[k]
                        one_tp, one_fp, _ = evaluator.mention_counts(gu, evaluator.positive_units(dict(subset, entities=[e]), truth, 'all'))
                        audit.append(dict(fold=index, pmc_id=docid, group=group, entity=e, tp=one_tp, fp=one_fp,
                                          same_span_in_old_pool=any(c['offset']==e['offset'] and c['length']==e['length'] for c in old_cs)))
            for name, pred in [('before', before[docid]), ('after', after[docid])]:
                tp, fp, _ = evaluator.mention_counts(gu, evaluator.positive_units(pred, truth, 'all'))
                counts[name+'_tp'] += tp; counts[name+'_fp'] += fp
                counts[name+'_records'] += len(pred['entities'])
        for metric in ['tp', 'fp', 'records']:
            assert counts['before_'+metric] == counts['common_retained_'+metric] + counts['common_removed_'+metric]
            assert counts['after_'+metric] == counts['common_retained_'+metric] + counts['common_added_'+metric] + counts['append_selected_'+metric]
            counts['net_'+metric] = counts['after_'+metric] - counts['before_'+metric]
            counts['common_net_'+metric] = counts['common_added_'+metric] - counts['common_removed_'+metric]
        folds.append(dict(fold=index, counts=dict(counts)))
        total.update(counts)
    assert len(seen) == 64
    for metric in ['tp','fp']:
        assert total['net_'+metric] == results['fixed03_comparison']['pooled']['deltas']['counts']['mention'][metric]
    assert total['after_records'] == results['anchor_cut03']['entity_records']
    assert total['before_records'] == results['comparison']['old_selected_pooled']['entity_records']
    out.mkdir()
    audit_path = out/'selected-changes.jsonl'
    with audit_path.open('x') as stream:
        for row in audit: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = dict(status='posthoc frozen .3 selected-key attribution complete', finished_at=datetime.now(timezone.utc).isoformat(),
                   source=ref(__file__), plan=ref(plan_path), folds=folds, total=dict(total), audit=ref(audit_path),
                   seconds=time.monotonic()-started, no_fit=True, no_decode=True, no_new_prediction=True, no_originaldev=True,
                   semantics='Counts are selected records and evaluator-v2 all/text mention atoms. Compound/NIL use unchanged v2. Disjoint nonoverlapping output groups add exactly to full mention TP/FP; document concepts are not attributed. Observational decomposition, not a causal counterfactual.')
    with (out/'summary.json').open('x') as stream: json.dump(summary, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
