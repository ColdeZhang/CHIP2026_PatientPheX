"""One frozen-pool error decomposition; no fitting, prediction, cutoff scan or v2 call."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from common import read_jsonl, write_json, write_jsonl, ref, sha


def units(rows, field='entities'):
    out = set()
    for doc in rows:
        for e in doc[field]:
            if str(e.get('note') or '').upper() == 'NO':
                continue
            for identifier in e['identifier'].split(';'):
                out.add((str(doc['pmc_id']), int(e['offset']), int(e['length']), identifier))
    return out


def main():
    ap = argparse.ArgumentParser()
    for name in ('pool', 'strong-manifest', 'gold', 'plan', 'quality', 'output'):
        ap.add_argument('--'+name, required=True)
    a = ap.parse_args(); out = Path(a.output)
    assert not (out/'supply.json').exists(), 'Completed analysis: reuse instead of replacing'
    plan = json.loads(Path(a.plan).read_text()); q = Path(a.quality)
    freeze = json.loads((q/'paired-freeze.json').read_text())
    assert sha(a.pool) == freeze['inputs']['pool']['sha256']
    manifest = json.loads(Path(a.strong_manifest).read_text())
    strong_ref = next(x['input'] for x in manifest['folds'] if x['fold'] == plan['held_fold'])
    assert sha(strong_ref['path']) == strong_ref['sha256']
    with gzip.open(strong_ref['path'], 'rt') as f:
        strong = [json.loads(x) for x in f]
    gold = read_jsonl(a.gold); gp = read_jsonl(a.pool)
    assert {str(x['pmc_id']) for x in gold} == {str(x['pmc_id']) for x in strong} == {str(x['pmc_id']) for x in gp}
    G = units(gold); S = units(strong, 'candidates')
    B = units(read_jsonl(plan['strong_baseline']['path']))
    P = units(read_jsonl(q/'primary.jsonl'))
    D = units(read_jsonl(q/'diagnostic.jsonl'))
    A = P - B
    boundaries = defaultdict(set)
    for k in G: boundaries[k[:3]].add(k[3])
    def category(k):
        if k in G:
            return 'exact_nil' if k[3] == '-1' else 'exact_hpo'
        if '-1' in boundaries.get(k[:3], set()): return 'nil_wildcard_only'
        if k[:3] in boundaries: return 'boundary_correct_wrong_hpo'
        return 'no_positive_gold_boundary'
    def summary(U):
        c = Counter(category(k) for k in U)
        return {'units': len(U), 'categories': dict(sorted(c.items())),
                'strict_hpo_precision': c['exact_hpo']/len(U) if U else None,
                'same_unit_in_strong_pool': len(U & S),
                'same_unit_in_strong_selected': len(U & B),
                'gold_boundary_correct_wrong_hpo_but_strong_has_correct_unit': sum(
                    category(k) == 'boundary_correct_wrong_hpo' and any((*k[:3], h) in S for h in boundaries[k[:3]] if h != '-1') for k in U)}
    GP = set(); rows = []
    for e in gp:
        for h in e['identifier'].split(';'):
            k = (str(e['pmc_id']), e['offset'], e['length'], h); GP.add(k)
            rows.append({**e, 'unit_identifier': h, 'error_category': category(k),
                         'in_strong_pool': k in S, 'in_strong_selected': k in B,
                         'in_primary_added': k in A, 'in_gp_only': k in D})
    selected = {k for k in GP if any(e['pmc_id'] == k[0] and e['offset'] == k[1] and e['length'] == k[2] and k[3] in e['identifier'].split(';') and e['gp_decision'] for e in gp)}
    base_metrics = json.loads(Path(plan['strong_metrics']['path']).read_text())
    primary_metrics = json.loads((q/'primary.metrics.json').read_text())
    delta = {group: {k: primary_metrics['counts'][group][k] - base_metrics['counts'][group][k] for k in ('tp','fp','fn')}
             for group in ('mention','document','relation')}
    result = {'status': 'COMPLETE_FROZEN_SUPPLY_ANALYSIS', 'source': ref(__file__),
              'sources': {'gp_pool': ref(a.pool), 'gold': ref(a.gold), 'strong_manifest': ref(a.strong_manifest),
                          'strong_pool': strong_ref, 'paired_freeze': ref(q/'paired-freeze.json')},
              'scope': 'All passages, positive gold entities only, split compound IDs; exact units use pmc/offset/length/HPO. NIL wildcard reported separately, not counted as exact HPO supply.',
              'gold_units': len(G), 'strong_pool_units': len(S), 'strong_selected_units': len(B),
              'gp_full_pool': summary(GP), 'gp_native_selected': summary(selected),
              'gp_pool_outside_strong_pool': summary(GP-S),
              'gp_selected_outside_strong_pool': summary(selected-S),
              'primary_added': summary(A),
              'primary_added_outside_strong_pool': summary(A-S),
              'gp_selected_in_strong_pool': summary(selected & S),
              'primary_exact_hpo_gain_outside_strong_selected': len({k for k in A & G if k[3] != '-1'}),
              'gp_raw_boundary_supply': {'unique_spans': len({k[:3] for k in GP}),
                                         'gold_boundary_spans': len({k[:3] for k in GP} & set(boundaries)),
                                         'gold_boundary_spans_absent_strong_pool': len(({k[:3] for k in GP} & set(boundaries))-{k[:3] for k in S})},
              'actual_metric_count_delta_reusing_existing_metrics': delta,
              'gold_read_for_diagnostic_only_after_existing_paired_freeze': True,
              'new_v2_calls': 0, 'fits': 0, 'new_predictions': 0, 'threshold_scan': False,
              'limits': 'This decomposition does not optimize a selector and cannot refute learned fusion. GP train32 and strong selector train48 differ. Original dev untouched.'}
    write_jsonl(out/'classified-pool.jsonl', rows)
    result['classified_pool'] = ref(out/'classified-pool.jsonl')
    write_json(out/'supply.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
