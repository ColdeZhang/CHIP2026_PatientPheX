"""Literal relation features; no baseline, gold, model inference or evaluator imports."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

FEATURES = ['grounded', 'sampleintersection', 'qwen3']
HPO = re.compile(r'HP:\d{7}\Z')
COMPOUND = re.compile(r'HP:\d{7}(;HP:\d{7})+\Z')


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    p = Path(path)
    b = p.read_bytes()
    return {'path': str(p.resolve()), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def checked(r):
    p = Path(r['path'])
    actual = ref(p)
    assert actual['bytes'] == r['bytes'] and actual['sha256'] == r['sha256'], str(p)
    return p


def read_json(r):
    return json.loads(checked(r).read_bytes())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir_new(path):
    p = Path(path)
    p.mkdir(exist_ok=False)
    fsync_dir(p.parent)
    return p


def write_bytes(path, body):
    p = Path(path)
    with p.open('xb') as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(p.parent)
    return ref(p)


def write_json(path, value):
    return write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def json_lines(r):
    return [json.loads(line) for line in checked(r).read_bytes().splitlines() if line.strip()]


def universe(r, documents, patients):
    rows = json_lines(r)
    assert len(rows) == documents
    u = {}
    for d in rows:
        pmc = d['pmc_id']
        assert isinstance(pmc, str) and pmc not in u
        ids = [x['patient_id'] for x in d['patient']]
        assert all(isinstance(x, str) for x in ids) and len(ids) == len(set(ids))
        u[pmc] = set(ids)
    assert sum(map(len, u.values())) == patients
    return u


def verify_key(key, u):
    assert len(key) == 3 and all(isinstance(v, str) for v in key)
    pmc, pid, term = key
    assert pmc in u and pid in u[pmc] and HPO.fullmatch(term), key
    return tuple(key)


def literal_predictions(r, u):
    """Read only association fields. Container entities are never features/selection."""
    rows = json_lines(r)
    assert len(rows) == len(u) and {x['pmc_id'] for x in rows} == set(u)
    result = set()
    for d in rows:
        pmc = d['pmc_id']
        seen_patients = set()
        for a in d['association']:
            pid = a['patient_id']
            assert pid not in seen_patients and pid in u[pmc]
            seen_patients.add(pid)
            assert len(a['phenotype']) == len(set(a['phenotype']))
            for term in a['phenotype']:
                result.add(verify_key((pmc, pid, term), u))
        assert seen_patients == u[pmc]
    return result


def sampled_intersection(r, u, expected_rows):
    streams = {'S1': {}, 'S2': {}}
    sets = {'S1': set(), 'S2': set()}
    for row in json_lines(r):
        stream = row['stream']
        if stream not in streams:
            assert stream == 'G'  # The historical greedy arm is deliberately ignored.
            continue
        idx = row['row_id']
        assert idx not in streams[stream]
        assert row['pmc_id'] in u
        assert row['strict_complete'] and row['actual_natural_eos'] and row['fresh']
        assert not row['actual_hit_token_limit'] and not row['actual_has_think_marker']
        assert not row['whole_row_abstention'] and not row['parser_error']
        for key in row['relations']:
            k = verify_key(key, u)
            assert k[0] == row['pmc_id']
            sets[stream].add(k)
        streams[stream][idx] = row['pmc_id']
    assert len(streams['S1']) == len(streams['S2']) == expected_rows
    assert streams['S1'] == streams['S2']
    assert set(streams['S1'].values()) == set(u)
    return sets['S1'] & sets['S2'], {k: len(v) for k, v in sets.items()}


def candidates(plan):
    u = universe(plan['input'], plan['documents'], plan['patients'])
    g = literal_predictions(plan['raw_sources']['grounded'], u)
    q = literal_predictions(plan['raw_sources']['qwen3'], u)
    si, sample_counts = sampled_intersection(plan['raw_sources']['sampled_diagnostics'], u, plan['sample_rows_per_seed'])
    source_sets = (g, si, q)
    keys = sorted(set.union(*source_sets))
    rows = [{'key': list(k), 'features': [int(k in s) for s in source_sets]} for k in keys]
    assert all(any(x['features']) for x in rows)
    return u, rows, {'grounded': len(g), 'sampleintersection': len(si), 'qwen3': len(q), 'union': len(rows), **sample_counts}


def gold_set(gold_ref, u):
    """Public v2 association semantics: compound HPO split, last patient assignment."""
    rows = json_lines(gold_ref)
    assert len(rows) == len(u) and {x['pmc_id'] for x in rows} == set(u)
    result = set()
    for doc in rows:
        patients = {}
        for association in doc['association']:
            terms = set()
            for value in association['phenotype']:
                terms.update(value.split(';') if COMPOUND.fullmatch(value) else [value])
            patients[association['patient_id']] = terms
        assert set(patients) <= u[doc['pmc_id']]
        for pid, terms in patients.items():
            result.update((doc['pmc_id'], pid, term) for term in terms)
    return result


def association_span(line):
    """Locate the one top-level association JSON value without touching other bytes."""
    decoder = json.JSONDecoder()
    i = 0
    while line[i].isspace():
        i += 1
    assert line[i] == '{'
    i += 1
    found = []
    while True:
        while line[i].isspace():
            i += 1
        if line[i] == '}':
            break
        key, i = decoder.raw_decode(line, i)
        while line[i].isspace():
            i += 1
        assert line[i] == ':'
        i += 1
        while line[i].isspace():
            i += 1
        start = i
        _, i = decoder.raw_decode(line, i)
        if key == 'association':
            found.append((start, i))
        while line[i].isspace():
            i += 1
        if line[i] == ',':
            i += 1
        else:
            assert line[i] == '}'
            break
    assert len(found) == 1
    return found[0]


def append_relations(base_ref, u, selected):
    """B0 used here only, after raw features/predictions: preserve complete base."""
    original = checked(base_ref).read_bytes()
    lines = original.decode().splitlines(keepends=True)
    assert len(lines) == len(u)
    output = []
    seen = set()
    counts = {'base_relations': 0, 'added_relations': 0, 'entities': 0}
    for line in lines:
        obj = json.loads(line)
        pmc = obj['pmc_id']
        assert pmc in u and pmc not in seen
        seen.add(pmc)
        associations = obj['association']
        ids = [a['patient_id'] for a in associations]
        assert len(ids) == len(set(ids)) and set(ids) == u[pmc]
        changed = json.loads(json.dumps(associations))
        for old, new in zip(associations, changed):
            assert len(old['phenotype']) == len(set(old['phenotype']))
            counts['base_relations'] += len(old['phenotype'])
            extra = sorted(t for d, p, t in selected if d == pmc and p == old['patient_id'] and t not in old['phenotype'])
            new['phenotype'] += extra
            counts['added_relations'] += len(extra)
            assert new['phenotype'][:len(old['phenotype'])] == old['phenotype']
        a, b = association_span(line)
        replacement = json.dumps(changed, ensure_ascii=False, separators=(',', ':'))
        new_line = line[:a] + replacement + line[b:]
        aa, bb = association_span(new_line)
        assert line[:a] == new_line[:aa] and line[b:] == new_line[bb:]
        counts['entities'] += len(obj['entities'])
        output.append(new_line)
    return ''.join(output).encode(), counts
