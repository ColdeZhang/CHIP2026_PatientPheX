"""Aggregate the frozen four-fold union failure; no fitting, threshold or oracle."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def ck(c):
    return c['offset'], c['length'], c['identifier']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    root = Path(plan['output_dir'])
    output = root/'existing-result-diagnostic.json'
    assert not output.exists()
    freeze = json.loads((root/'complete-freeze.json').read_text())
    final = json.loads((root/'results.json').read_text())
    assert final['freeze_sha256'] == sha(root/'complete-freeze.json')
    assert freeze['plan_sha256'] == final['plan_sha256'] == sha(a.plan)
    sys.path.insert(0, plan['evaluation_source'])
    import evaluate
    gold = {str(d['pmc_id']): d for d in read(plan['train64_gold'])}
    assert sha(plan['train64_gold']) == plan['train64_gold_sha256']
    prepared = read(root/'prepared/candidates.jsonl')
    union = {str(d['pmc_id']): {ck(c): c for c in d['candidates']} for d in prepared}
    old_pub = {str(d['pmc_id']): {ck(c): c for c in d['candidates']}
               for d in read(plan['old_prepared']['path'])}
    inputs = {}
    totals, selection_changes = defaultdict(Counter), defaultdict(Counter)
    observed_new_public = set()
    fold_reports = []
    for spec, ff in zip(plan['folds'], freeze['folds']):
        old_manifest_path = Path(spec['old095_manifest']['path'])
        assert sha(old_manifest_path) == spec['old095_manifest']['sha256']
        old_manifest = json.loads(old_manifest_path.read_text())
        old_scores_path = old_manifest_path.parent/'selector-scores.jsonl'
        assert sha(old_scores_path) == old_manifest['outputs']['selector-scores.jsonl']
        assert sha(ff['scores']['path']) == ff['scores']['sha256']
        for role in ['old095_prediction', 'new_prediction']:
            assert sha(ff[role]['path']) == ff[role]['sha256']
        inputs[str(old_scores_path)] = sha(old_scores_path)
        inputs[ff['scores']['path']] = ff['scores']['sha256']
        old_scores = {str(d['pmc_id']): {ck(c): c for c in d['candidates']} for d in read(old_scores_path)}
        new_scores = {str(d['pmc_id']): {ck(c): c for c in d['candidates']} for d in read(ff['scores']['path'])}
        before = {str(d['pmc_id']): d for d in read(ff['old095_prediction']['path'])}
        after = {str(d['pmc_id']): d for d in read(ff['new_prediction']['path'])}
        assert before.keys() == after.keys() == old_scores.keys() == new_scores.keys()
        fold_count = defaultdict(Counter)
        for did in before:
            gt = gold[did]
            truth = {}
            for e in gt['entities']:
                if str(e.get('note', '')).upper() != 'NO':
                    truth.setdefault((e['offset'], e['length']), set()).update(e['identifier'].split(';'))
            def correct(k):
                ids = truth.get(k[:2], set())
                return bool(ids and ('-1' in ids or set(k[2].split(';')) <= ids))
            old, new = old_scores[did], new_scores[did]
            assert old.keys() <= new.keys()
            extra = set(union[did])-set(old_pub[did])
            truly_new = set(new)-set(old)
            assert truly_new <= extra
            accepted_old = {ck(e) for e in before[did]['entities']}
            accepted_new = {ck(e) for e in after[did]['entities']}
            assert accepted_old <= old.keys() and accepted_new <= new.keys()
            for k in extra:
                observed_new_public.add((did, *k))
                role = 'extra_public_evidence_on_existing_pool' if k in old else 'truly_new_pool_candidate'
                stats = fold_count[role]
                stats['candidates'] += 1
                stats['training_label_semantics_positive'] += correct(k)
                stats['selected_by_new'] += k in accepted_new
                stats['selected_by_new_positive'] += k in accepted_new and correct(k)
                stats['selected_by_old'] += k in accepted_old
            shared_old = set(old)-extra
            for role, keys in [('shared_old_candidates_without_new_public_key', shared_old),
                               ('all_common_old_pool', set(old))]:
                fold_count[role]['candidates'] += len(keys)
                fold_count[role]['selected_by_old'] += len(keys & accepted_old)
                fold_count[role]['selected_by_new'] += len(keys & accepted_new)
            gu = evaluate.positive_units(gt, gt, 'all')
            au = evaluate.positive_units(before[did], gt, 'all')
            bu = evaluate.positive_units(after[did], gt, 'all')
            for action, pred, left, right in [('added', after[did], bu, au),
                                             ('removed', before[did], au, bu)]:
                diff = {span: ids-right.get(span, set()) for span, ids in left.items()
                        if ids-right.get(span, set())}
                partition = defaultdict(lambda: defaultdict(set))
                for e in pred['entities']:
                    k = ck(e)
                    atoms = set(e['identifier'].split(';')) & diff.get(k[:2], set())
                    if not atoms:
                        continue
                    if k in truly_new:
                        role = 'truly_new_pool_candidate'
                    elif k in extra:
                        role = 'extra_public_evidence_on_existing_pool'
                    else:
                        role = 'shared_old_without_new_public_key'
                    partition[role][k[:2]].update(atoms)
                for role, units in partition.items():
                    tp, fp, _ = evaluate.mention_counts(gu, units)
                    selection_changes[action+'__'+role].update(tp=tp, fp=fp)
        for role, count in fold_count.items():
            totals[role].update(count)
        fold_reports.append({'fold': spec['fold'], 'groups': {k: dict(v) for k, v in fold_count.items()}})
    assert len(observed_new_public) == 2157
    for action in ['added', 'removed']:
        for outcome in ['tp', 'fp']:
            actual = sum(v[outcome] for role, v in selection_changes.items() if role.startswith(action+'__'))
            assert actual == final['pooled']['changes']['entity_'+action+'_'+outcome]
    report = {'source_sha256': sha(__file__), 'plan_sha256': sha(a.plan),
              'frozen_results_sha256': sha(root/'results.json'), 'freeze_sha256': sha(root/'complete-freeze.json'),
              'scope': 'Aggregate already frozen strict OOF held predictions and source pools only; no fitting, new decode, threshold search, oracle prediction or point patch.',
              'new_public_keys': 2157, 'groups': {k: dict(v) for k, v in totals.items()},
              'v2_atomic_entity_changes': {k: dict(v) for k, v in selection_changes.items()},
              'folds': fold_reports, 'probability_source_files': inputs,
              'label_semantics': 'Candidate positive labels use the unchanged HGB training exact span and original-ID/subset/NIL rule. Atomic added/removed TPFP use frozen v2; these two denominators are explicit.',
              'interpretation_limit': 'Shared-pool decision changes cannot alone separate refitting, added public features and greedy competition; no causal attribution to one mechanism without another controlled experiment.'}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'groups': report['groups'], 'v2_atomic_entity_changes': report['v2_atomic_entity_changes']}))


if __name__ == '__main__':
    main()
