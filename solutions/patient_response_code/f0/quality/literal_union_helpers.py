"""Exact generic function copies only; no model/raw-response parser/evaluator."""

import copy

import json

def build_union(old, new, prompts):
    pool = {(str(p['pmc_id']), str(pid), hpo) for p in prompts
            for pid in p['patient_ids'] for hpo in p['code_map'].values()}
    assert len(old) == len(new)
    result = copy.deepcopy(old)
    old_pairs, new_pairs, result_pairs = set(), set(), set()
    for before, current, output in zip(old, new, result):
        assert {k:v for k,v in before.items() if k != 'association'} == {
            k:v for k,v in current.items() if k != 'association'}
        assert len(before['association']) == len(current['association'])
        for a, b, c in zip(before['association'], current['association'], output['association']):
            assert {k:v for k,v in a.items() if k != 'phenotype'} == {
                k:v for k,v in b.items() if k != 'phenotype'}
            pmc, pid = str(before['pmc_id']), str(a['patient_id'])
            left, right = set(a['phenotype']), set(b['phenotype'])
            assert len(left) == len(a['phenotype']) and len(right) == len(b['phenotype'])
            c['phenotype'] = sorted(left | right)
            old_pairs.update((pmc,pid,h) for h in left)
            new_pairs.update((pmc,pid,h) for h in right)
            result_pairs.update((pmc,pid,h) for h in c['phenotype'])
    assert old_pairs <= pool and new_pairs <= pool
    assert result_pairs == old_pairs | new_pairs
    return result, dict(pool_pairs=len(pool), old_relations=len(old_pairs),
                       primary_relations=len(new_pairs), union_relations=len(result_pairs),
                       added_over_old=len(result_pairs-old_pairs), removed_from_old=0,
                       all_nonassociation_exact=True, all_patient_metadata_exact=True)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def association_span(line):
    """Locate the one top-level value without reserializing any other JSON bytes."""
    decoder, position, names, found = json.JSONDecoder(), 0, set(), None
    def whitespace(index):
        while index < len(line) and line[index].isspace():
            index += 1
        return index
    position = whitespace(position)
    require(line[position:position + 1] == '{', 'Require JSON object line')
    position += 1
    while True:
        position = whitespace(position)
        if line[position:position + 1] == '}':
            position += 1
            break
        key, position = decoder.raw_decode(line, position)
        require(isinstance(key, str) and key not in names, 'Duplicate/nonstring top-level key')
        names.add(key)
        position = whitespace(position)
        require(line[position:position + 1] == ':', 'Missing object colon')
        begin = whitespace(position + 1)
        value, end = decoder.raw_decode(line, begin)
        if key == 'association':
            require(isinstance(value, list), 'Association must be a list')
            found = (begin, end)
        position = whitespace(end)
        if line[position:position + 1] == ',':
            position += 1
            continue
        require(line[position:position + 1] == '}', 'Invalid object separator')
        position += 1
        break
    require(not line[position:].strip() and found is not None, 'Missing association/trailing JSON')
    return found

def nonrelation_bytes(lines):
    output = []
    for line in lines:
        begin, end = association_span(line)
        output.append((line[:begin] + '[]' + line[end:]).encode('utf-8'))
    return b''.join(output)
