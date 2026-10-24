"""Lossless source-unit addressing and strict copied-evidence parsing."""
import json

INPUT_KEYS = {'pmc_id', 'pmid', 'patient', 'full_text'}
SYSTEM = '''Extract observed phenotypic abnormalities of the TARGET patient from this report.
Use all patient mentions to distinguish the target from relatives, other cases, and general discussion.
Include positive clinical abnormalities; exclude negated findings, hypothetical risks, and causal syndrome names.
Return only a JSON array of [unit_id, exact_quote, occurrence_index] triples.
Copy a short complete phenotype phrase exactly from one supplied source unit. occurrence_index is zero-based among exact occurrences of that quote inside the unit.
Do not generate HPO codes, explanations, new wording, or information absent from the report. If no supported abnormality is found, return [].'''


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def occurrences(text, quote):
    if not isinstance(quote, str) or not quote:
        raise ValueError('Quote must be nonempty text')
    found = []; start = 0
    while True:
        offset = text.find(quote, start)
        if offset < 0:
            return found
        found.append(offset); start = offset + 1


def units(doc):
    result = []
    for index, passage in enumerate(doc['full_text']):
        assert isinstance(passage['text'], str)
        assert type(passage['offset']) is int and passage['offset'] >= 0
        result.append(dict(unit_id=f'U{index:04d}', offset=passage['offset'], text=passage['text'],
                           section_type=passage.get('section_type', ''), type=passage.get('type', '')))
    return result


def locate(doc, quote, offset):
    hits = []
    for unit in units(doc):
        local = offset - unit['offset']
        matches = occurrences(unit['text'], quote)
        if local in matches:
            hits.append([unit['unit_id'], quote, matches.index(local)])
    if len(hits) != 1:
        raise ValueError(f'Expected one exact source unit at supplied offset; got {len(hits)}')
    return hits[0]


def grounded(triple, unit_by_id):
    if not isinstance(triple, list) or len(triple) != 3:
        raise ValueError('Evidence must be a three-element array')
    uid, quote, occurrence = triple
    if not isinstance(uid, str) or uid not in unit_by_id or type(occurrence) is not int:
        raise ValueError('Unknown source unit or noninteger occurrence')
    unit = unit_by_id[uid]; matches = occurrences(unit['text'], quote)
    if not 0 <= occurrence < len(matches):
        raise ValueError('Quote/occurrence is absent from the original unit')
    return dict(unit_id=uid, quote=quote, occurrence_index=occurrence,
                offset=unit['offset'] + matches[occurrence], length=len(quote),
                matching_occurrences_in_unit=len(matches))


def messages(doc, target_id):
    assert target_id in {p['patient_id'] for p in doc['patient']}
    patient_table = []
    for patient in doc['patient']:
        mentions = []
        for mention in patient['mention']:
            triple = locate(doc, mention['text'], mention['offset'])
            mentions.append(triple)
        patient_table.append(dict(patient_id=patient['patient_id'], mentions=mentions))
    body = '\n\n'.join(f"[{u['unit_id']} | {u['section_type']} | {u['type']}]\n{u['text']}" for u in units(doc))
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
        f'TARGET: {target_id}\nPATIENT MENTIONS: {compact(patient_table)}\n\nFULL REPORT:\n{body}'}]


def contracts():
    doc = {'pmc_id': 'synthetic', 'pmid': '', 'patient': [{'patient_id': 'P1',
        'mention': [{'text': 'P1', 'offset': 100, 'length': 2}]}],
        'full_text': [{'offset': 100, 'text': 'P1: café; café.', 'section_type': 'CASE', 'type': 'paragraph'}]}
    by = {u['unit_id']: u for u in units(doc)}
    assert locate(doc, 'café', 110) == ['U0000', 'café', 1]
    assert grounded(['U0000', 'café', 1], by)['offset'] == 110
    assert grounded(['U0000', 'café', 0], by)['offset'] == 104
    rejected = 0
    for value in [['U9999', 'café', 0], ['U0000', 'cafe', 0], ['U0000', 'café', True],
                  ['U0000', 'café', -1], ['U0000', 'café', 2], ['U0000', '', 0], ['U0000', 'café']]:
        try: grounded(value, by)
        except ValueError: rejected += 1
    assert rejected == 7
    assert messages(doc, 'P1')[-1]['content'].endswith('P1: café; café.')
    return dict(real_codepoint_offsets=True,duplicate_occurrence_preserved=True,rejected_cases=rejected,
                exact_text_not_case_or_whitespace_repaired=True)
