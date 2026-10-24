"""F0-only complete literal interface. Does not change the fitted F1 representation.

Actual grounded F0 includes four accepted NIL mentions as original text. Match
public-v2 association units: split only pure compound HPO IDs, preserve every
other literal. Complete sampler rows may explicitly abstain with no relations.
"""
from common import COMPOUND, json_lines, universe


def terms(value):
    assert isinstance(value, str) and value
    return value.split(';') if COMPOUND.fullmatch(value) else [value]


def keys(key, u):
    assert len(key) == 3 and all(isinstance(v, str) for v in key)
    pmc, pid, value = key
    assert pmc in u and pid in u[pmc]
    return {(pmc, pid, term) for term in terms(value)}


def literal_predictions(r, u):
    rows = json_lines(r)
    assert len(rows) == len(u) and {x['pmc_id'] for x in rows} == set(u)
    result = set()
    for doc in rows:
        seen = set()
        for a in doc['association']:
            pid = a['patient_id']
            assert pid in u[doc['pmc_id']] and pid not in seen
            seen.add(pid)
            assert len(a['phenotype']) == len(set(a['phenotype']))
            for value in a['phenotype']:
                result.update(keys((doc['pmc_id'], pid, value), u))
        assert seen == u[doc['pmc_id']]
    return result


def sampled_intersection(r, u, expected_rows):
    streams = {'S1': {}, 'S2': {}}
    sets = {'S1': set(), 'S2': set()}
    for row in json_lines(r):
        stream = row['stream']
        if stream not in streams:
            assert stream == 'G'
            continue
        assert row['row_id'] not in streams[stream] and row['pmc_id'] in u and row['fresh']
        if row['whole_row_abstention']:
            assert row['strict_complete'] is False and row['relations'] == []
            assert row['reasons'] or row['parser_error'] or row['actual_hit_token_limit'] or row['actual_has_think_marker']
        else:
            assert row['strict_complete'] and row['actual_natural_eos']
            assert not row['actual_hit_token_limit'] and not row['actual_has_think_marker'] and not row['parser_error']
            for key in row['relations']:
                assert key[0] == row['pmc_id']
                sets[stream].update(keys(key, u))
        streams[stream][row['row_id']] = row['pmc_id']
    assert len(streams['S1']) == len(streams['S2']) == expected_rows
    assert streams['S1'] == streams['S2'] and set(streams['S1'].values()) == set(u)
    return sets['S1'] & sets['S2'], {k: len(v) for k, v in sets.items()}


def candidates(plan):
    u = universe(plan['input'], plan['documents'], plan['patients'])
    g = literal_predictions(plan['raw_sources']['grounded'], u)
    q = literal_predictions(plan['raw_sources']['qwen3'], u)
    si, sample_counts = sampled_intersection(plan['raw_sources']['sampled_diagnostics'], u, plan['sample_rows_per_seed'])
    source_sets = (g, si, q)
    all_keys = sorted(set.union(*source_sets))
    rows = [{'key': list(k), 'features': [int(k in s) for s in source_sets]} for k in all_keys]
    assert all(any(r['features']) for r in rows)
    return u, rows, {'grounded': len(g), 'sampleintersection': len(si), 'qwen3': len(q), 'union': len(rows), **sample_counts}
