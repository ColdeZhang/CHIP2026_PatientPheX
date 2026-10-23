#!/usr/bin/env python3
"""Transfer the frozen proper-nesting policy to a frozen HGB candidate pool."""
from collections import Counter
import math
import append_nested_spans as prior
from decode_weighted_intervals import entity, greedy, key


def decode(candidates, reference, doc, allowed, cut=.3):
    assert cut == .3
    assert greedy(candidates, cut) == reference
    selected = list(reference)
    refkeys = {key(c) for c in reference}
    assert len(refkeys) == len(reference)
    assert len({key(c) for c in candidates}) == len(candidates)
    counts = Counter()
    collision_checks = []
    for c in sorted(candidates, key=lambda c: (-c['selector_probability'], -c['length'], c['offset'])):
        p = c['selector_probability']
        assert math.isfinite(p) and 0 <= p <= 1
        counts['all_candidates'] += 1
        if key(c) in refkeys:
            counts['already_selected'] += 1
            continue
        if p < cut:
            counts['below_cut'] += 1
            continue
        # Original greedy retained every above-cut candidate without an overlap.
        assert any(prior.overlap(c, e) for e in reference)
        counts['original_overlap_rejection'] += 1
        ids = set(c['identifier'].split(';'))
        if not ids or not ids <= allowed:
            counts['invalid_hpo_or_nil'] += 1
            continue
        _, passage = prior.base.passage_for(doc, c['offset'])
        local = c['offset'] - int(passage.get('offset', 0))
        assert passage['text'][local:local+c['length']] == c['text']
        if prior.combine.ontology.is_negated(passage['text'], local, local+c['length']):
            counts['negated'] += 1
            continue
        collisions = [e for e in selected if prior.overlap(c, e)]
        detail = dict(candidate_key=key(c), collisions=[
            dict(entity_key=key(e), proper_containment=prior.proper_containment(c, e),
                 disjoint_ids=not ids.intersection(e['identifier'].split(';')),
                 original_entity=key(e) in refkeys) for e in collisions])
        accepted = prior.may_append(c, selected)
        assert accepted == (bool(collisions) and all(x['proper_containment'] and x['disjoint_ids'] for x in detail['collisions']))
        collision_checks.append(dict(**detail, accepted=accepted))
        if not accepted:
            counts['not_distinct_proper_nesting'] += 1
            continue
        selected.append(entity(c))
        counts['added'] += 1
    selected.sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
    assert all(e in selected for e in reference)
    return selected, dict(counts, collision_checks=collision_checks)
