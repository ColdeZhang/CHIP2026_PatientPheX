"""Freeze both complete predictions before exactly two public-v2 evaluations."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
from common import read_jsonl, write_json, write_jsonl, ref, sha


def entity_key(e): return (int(e['offset']), int(e['length']), e['identifier'])


def compatible(candidate, existing):
    a, b = candidate['offset'], candidate['offset'] + candidate['length']
    for old in existing:
        c, d = old['offset'], old['offset'] + old['length']
        if max(a, c) >= min(b, d): continue
        # Exact-span alternative IDs and crossing boundaries are not appended.
        if a == c and b == d: return False
        contained = a <= c and d <= b or c <= a and b <= d
        if not contained or candidate['identifier'] == old['identifier']: return False
    return True


def main():
    ap = argparse.ArgumentParser()
    for name in ('plan', 'input', 'entities', 'pool', 'gold', 'evaluator', 'output', 'runtime-complete'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); started = time.monotonic(); out = Path(a.output)
    assert not (out/'paired-freeze.json').exists(), 'Already frozen; recover without repeating scores'
    plan = json.loads(Path(a.plan).read_text())
    for k in ('strong_baseline', 'strong_metrics'):
        assert sha(plan[k]['path']) == plan[k]['sha256']
    assert sha(a.evaluator) == 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    sys.path.insert(0, str(Path(a.evaluator).parent))
    spec = importlib.util.spec_from_file_location('gp_public_v2', a.evaluator)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    inputs = read_jsonl(a.input); baseline = read_jsonl(plan['strong_baseline']['path'])
    gp = {str(x['pmc_id']): x['entities'] for x in read_jsonl(a.entities)}
    pool = read_jsonl(a.pool); bydoc = {str(x['pmc_id']): [] for x in inputs}
    for r in pool:
        if r['gp_decision']: bydoc[r['pmc_id']].append(r)
    assert len(inputs) == 16 and set(gp) == set(bydoc) == {str(x['pmc_id']) for x in baseline}
    primary = []; diagnostic = []; added = []
    for old in baseline:
        pmc = str(old['pmc_id']); new = copy.deepcopy(old); alone = copy.deepcopy(old)
        new_keys = {entity_key(x) for x in new['entities']}
        for row in sorted(bydoc[pmc], key=lambda x: (-x['gp_logit'], -x['length'], x['offset'], x['identifier'])):
            if row['identifier'] == '-1': continue
            e = {k: row[k] for k in ('identifier', 'offset', 'length', 'text')}; e.update(type='Phenotype', note=None)
            if entity_key(e) in new_keys or not compatible(e, new['entities']): continue
            new['entities'].append(e); new_keys.add(entity_key(e)); added.append({'pmc_id': pmc, **e, 'gp_logit': row['gp_logit']})
        dedup = {entity_key(e): e for e in gp[pmc]}
        alone['entities'] = sorted(dedup.values(), key=entity_key)
        assert all(new[k] == old[k] == alone[k] for k in old if k != 'entities')
        assert new['entities'][:len(old['entities'])] == old['entities']
        primary.append(new); diagnostic.append(alone)
    for name, rows in [('primary', primary), ('diagnostic', diagnostic)]:
        report = ev.validate_predictions(inputs, rows); write_json(out/(name+'.format.json'), report)
        assert report['valid'], report['errors']
        write_jsonl(out/(name+'.jsonl'), rows)
    write_jsonl(out/'added-entities.jsonl', added)
    freeze = {'at_utc': datetime.now(timezone.utc).isoformat(), 'plan': ref(a.plan), 'runtime_complete': ref(a.runtime_complete),
              'inputs': {k: ref(getattr(a, k)) for k in ('input', 'entities', 'pool', 'evaluator')},
              'primary': ref(out/'primary.jsonl'), 'diagnostic': ref(out/'diagnostic.jsonl'),
              'gold_opened_before_freeze': False, 'baseline_reused_without_reevaluation': True}
    write_json(out/'paired-freeze.json', freeze)
    write_json(out/'evaluation-intent.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'calls': 2, 'old_calls': 0})
    gold = read_jsonl(a.gold)
    m1 = ev.evaluate(gold, primary); write_json(out/'primary.metrics.json', m1)
    m2 = ev.evaluate(gold, diagnostic); write_json(out/'diagnostic.metrics.json', m2)
    old = json.loads(Path(plan['strong_metrics']['path']).read_text())['metrics']
    result = {'status': 'COMPLETE_FIXED_TWO_ENDPOINTS', 'primary': m1['metrics'], 'diagnostic': m2['metrics'],
              'baseline': old, 'delta': {k: m1['metrics'][k]-old[k] for k in old},
              'retain_primary': m1['metrics']['Score'] > old['Score'], 'additional_entities': len(added),
              'primary_entities': sum(len(x['entities']) for x in primary), 'diagnostic_entities': sum(len(x['entities']) for x in diagnostic),
              'relation_objects_exact': True, 'new_v2_calls': 2, 'old_v2_calls': 0,
              'seconds': time.monotonic()-started, 'freeze': ref(out/'paired-freeze.json'),
              'interpretation': 'GP trained32 versus existing strong selector trained48; not a matched pure architecture causal comparison. F1 development only, not original dev. Fixed union failure would not alone refute OOF learned source integration.'}
    write_json(out/'result.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
