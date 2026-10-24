"""Read-only component census of a frozen four-fold selector, without optimization.

Only manifest references, input pools, probabilities and public HPO are opened.
The original decoder is replayed in memory to identify its anchors and feasible
selection. No new selection, model, gold, evaluator or prediction file is used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time


def ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def checked(spec):
    assert ref(spec['path']) == spec, spec['path']
    return Path(spec['path'])


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def components(nodes, edges):
    adjacency = [set() for _ in nodes]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen = set(range(len(nodes)))
    result = []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        group = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v in adjacency[u]:
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
        result.append(group)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second census cap')))
    signal.alarm(60)
    plan = read(args.plan)
    assert plan['read_only_no_gold_fit_score_prediction'] is True
    old = read(checked(plan['old_plan']))
    freeze = read(checked(plan['old_freeze']))
    prepared = read(checked(old['prepared_manifest']))
    assert freeze['plan_sha256'] == plan['old_plan']['sha256']
    for path, digest in old['sources'].items():
        assert ref(path)['sha256'] == digest
    sys.path.insert(0, old['nested_source'])
    import decode_nested_selector as decoder
    prior = decoder.prior
    _, allowed, _, _ = prior.base.parse_hpo(checked(old['hpo']))
    allowed.discard('HP:0000118')
    provenance, documents = [], []
    total = Counter()
    hist = Counter()
    overlap_hist = Counter()
    seen = set()
    for fold in freeze['folds']:
        f = fold['fold']
        scores = rows(checked(fold['scores']))
        input_ref = prepared['folds'][f]['input']
        inputs = rows(checked(input_ref))
        fit = read(checked(fold['new_manifest']))
        assert len(scores) == len(inputs) == 16
        provenance.append(dict(fold=f, scores=fold['scores'], input=input_ref,
                               fit=fold['new_manifest'], actual_features=fit['actual_feature_count']))
        for scored, row in zip(scores, inputs):
            assert set(scored) == {'pmc_id', 'candidates'}
            assert scored['pmc_id'] == row['pmc_id'] and row['pmc_id'] not in seen
            seen.add(row['pmc_id'])
            assert set(row['input']) == {'pmc_id', 'pmid', 'patient', 'full_text'}
            cs = scored['candidates']
            assert [{k: v for k, v in c.items() if k != 'selector_probability'} for c in cs] == row['candidates']
            assert all(not ({'label', 'labels', 'target', 'gold', 'y'} & set(c)) for c in cs)
            greedy = decoder.greedy(cs, .3)
            anchor_keys = {decoder.key(c) for c in greedy}
            selected, diagnostic = decoder.decode(cs, greedy, row['input'], allowed, .3)
            selected_keys = {decoder.key(c) for c in selected}
            counts = Counter(candidates=len(cs), original_greedy=len(greedy), current_nested=len(selected))
            eligible = []
            for c in cs:
                p = c['selector_probability']
                assert math.isfinite(p) and 0 <= p <= 1
                if p < .3:
                    continue
                counts['above_or_at_cut'] += 1
                counts['exact_at_cut'] += int(p == .3)
                k = decoder.key(c)
                if k not in anchor_keys:
                    ids = set(c['identifier'].split(';'))
                    if not ids or not ids <= allowed:
                        counts['nongreedy_invalid_hpo'] += 1
                        continue
                    _, passage = prior.base.passage_for(row['input'], c['offset'])
                    local = c['offset'] - int(passage.get('offset', 0))
                    assert passage['text'][local:local+c['length']] == c['text']
                    if prior.combine.ontology.is_negated(passage['text'], local, local+c['length']):
                        counts['nongreedy_negated'] += 1
                        continue
                eligible.append(c)
            # The original greedy anchors bypassed these unary filters. Preserve
            # that asymmetry; globally applying them would be an extra variable.
            assert selected_keys <= {decoder.key(c) for c in eligible}
            counts['eligible'] = len(eligible)
            conflicts, overlaps = [], []
            selected_indices = {i for i, c in enumerate(eligible) if decoder.key(c) in selected_keys}
            for i, a in enumerate(eligible):
                for j in range(i):
                    b = eligible[j]
                    if not prior.overlap(a, b):
                        continue
                    overlaps.append((j, i))
                    ids_shared = bool(set(a['identifier'].split(';')) & set(b['identifier'].split(';')))
                    proper = prior.proper_containment(a, b)
                    if proper and not ids_shared:
                        counts['compatible_nested_pairs'] += 1
                        continue
                    conflicts.append((j, i))
                    assert not {i, j} <= selected_indices
                    if (a['offset'], a['length']) == (b['offset'], b['length']):
                        counts['conflict_same_boundaries'] += 1
                    elif proper:
                        assert ids_shared
                        counts['conflict_containment_shared_id'] += 1
                    else:
                        counts['conflict_crossing'] += 1
            groups = components(eligible, conflicts)
            overlap_groups = components(eligible, overlaps)
            counts['conflict_edges'] = len(conflicts)
            counts['overlap_pairs'] = len(overlaps)
            counts['components'] = len(groups)
            counts['nontrivial_components'] = sum(len(g) > 1 for g in groups)
            sizes = Counter(map(len, groups))
            osizes = Counter(map(len, overlap_groups))
            # An upper bound on exhaustive subset feasibility, not a search.
            subset_bound = sum(2 ** len(g) for g in groups if len(g) > 1)
            counts['nontrivial_exhaustive_subset_bound'] = subset_bound
            hist.update(sizes)
            overlap_hist.update(osizes)
            total.update(counts)
            documents.append(dict(fold=f, pmc_id=row['pmc_id'], counts=dict(counts),
                                  conflict_component_histogram=dict(sorted(sizes.items())),
                                  max_conflict_component=max(sizes, default=0),
                                  max_overlap_component=max(osizes, default=0)))
    assert len(seen) == len(documents) == 64
    result = dict(status='read_only_component_census_complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.monotonic()-started, source=ref(__file__), plan=ref(args.plan),
                  inputs=provenance, counts=dict(total),
                  conflict_component_histogram=dict(sorted(hist.items())),
                  overlap_component_histogram=dict(sorted(overlap_hist.items())),
                  max_conflict_component=max(hist, default=0), max_overlap_component=max(overlap_hist, default=0),
                  documents=documents, gold_read=False, new_fit=False, gpu_used=False,
                  optimized=False, new_predictions_written=False, quality_evaluated=False,
                  eligibility='original greedy anchors OR above-cut nonanchors passing original valid-HPO and negation guards',
                  compatible='disjoint half-open intervals OR proper containment with disjoint atomic HPO IDs',
                  limitations=['No summed-utility optimization was run.', 'No selector calibration or quality claim.',
                               'Current anchors retain original unary-filter bypass; no changed eligibility policy.'])
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    signal.alarm(0)
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'counts', 'conflict_component_histogram', 'max_conflict_component', 'max_overlap_component')}))


if __name__ == '__main__':
    main()
