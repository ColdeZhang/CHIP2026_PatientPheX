"""Literal helpers copied verbatim from frozen R1 evaluator SHA
4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522.
Only generic JSON byte-preserving assembly logic; F0 retain is defined in its own entry; no R1 code runs.
The inherited local name `med` is only an argument/count label.
"""
import copy
import json
import math

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


def relation_map(document):
    result = {}
    for association in document['association']:
        require(set(association) == {'patient_id', 'phenotype'}, 'Unexpected association fields')
        patient = str(association['patient_id'])
        phenotypes = association['phenotype']
        require(patient not in result and isinstance(phenotypes, list)
                and all(isinstance(value, str) for value in phenotypes)
                and len(phenotypes) == len(set(phenotypes)), 'Duplicate/invalid patient or phenotype')
        result[patient] = set(phenotypes)
    return result


def assemble_two(old_lines, med_documents):
    baseline = [json.loads(line) for line in old_lines]
    require(len(baseline) == len(med_documents), 'Document coverage differs')
    require(len({str(d['pmc_id']) for d in baseline}) == len(baseline), 'Duplicate baseline document')
    primary, union = [], []
    counts = {'old': 0, 'med': 0, 'intersection': 0, 'union': 0}
    for raw, old, med in zip(old_lines, baseline, med_documents):
        require({k: v for k, v in old.items() if k != 'association'}
                == {k: v for k, v in med.items() if k != 'association'}, 'Nonrelation fields/document order changed')
        old_map, med_map = relation_map(old), relation_map(med)
        require(set(old_map) == set(med_map), 'Patient coverage differs')
        combined = copy.deepcopy(old['association'])
        for association in combined:
            patient = str(association['patient_id'])
            left, right = old_map[patient], med_map[patient]
            association['phenotype'] = sorted(left | right)
            counts['old'] += len(left)
            counts['med'] += len(right)
            counts['intersection'] += len(left & right)
            counts['union'] += len(left | right)
        begin, end = association_span(raw)
        primary.append(raw[:begin] + json.dumps(med['association'], ensure_ascii=False) + raw[end:])
        union.append(raw[:begin] + json.dumps(combined, ensure_ascii=False) + raw[end:])
    unchanged = nonrelation_bytes(old_lines)
    require(nonrelation_bytes(primary) == nonrelation_bytes(union) == unchanged, 'Nonrelation byte drift')
    return primary, union, counts


