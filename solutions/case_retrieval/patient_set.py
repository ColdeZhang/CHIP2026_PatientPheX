#!/usr/bin/env python3
"""Whole-report patient-set SFT preparation and strict, gold-free output parsing.

Training entities are used only to construct assistant quote targets. Literal
quote support does not establish patient attribution or HPO semantic validity.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from run import load_hpo, read_jsonl, sha256
from prepare_rag import mark_patients

HPO = re.compile(r'HP:[0-9]{7}')
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})*')
SYSTEM = '''Read the entire biomedical report for the single TARGET patient.
Return exactly {"phenotypes":[{"identifier":"HP:ddddddd","segment_id":"S000","quote":"verbatim phenotype phrase"}]}.
Use an HPO identifier for an equivalent phenotype; use identifier "-1" only when no equivalent HPO concept exists. quote must be the exact case-sensitive phenotype phrase occurring as whole words in the specified original SEGMENT. Never copy inline PATIENT markers into a quote. The quote proves only textual support; you must separately decide the finding belongs to TARGET and matches the HPO meaning.
Include observed findings of TARGET, including that patient's findings described across paragraphs. Exclude negated/normal findings, hypothetical findings, other patients and general disease descriptions. Do not infer symptoms from diagnoses. All supplied patients, including relatives, have distinct PATIENT codes; preserve their identities and resolve unambiguous references from the full report. Do not transfer one patient's findings to another.
Return one item per distinct phenotype, with one supporting quote. Use {"phenotypes":[]} if TARGET has no supported findings. No explanation, markdown, offsets, or additional fields.'''


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def read_rows(path):
    """Patient prompts/responses legitimately contain repeated document IDs."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def indexed(rows):
    result = {str(row['pmc_id']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate document IDs')
    return result


def word(char):
    return char.isalnum() or char == '_'


def boundary(text, start, end):
    return (0 <= start < end <= len(text)
            and not (start and word(text[start]) and word(text[start-1]))
            and not (end < len(text) and word(text[end-1]) and word(text[end])))


def literal_positions(text, quote):
    if not isinstance(quote, str) or not quote:
        return []
    return [match.start() for match in re.finditer(re.escape(quote), text)
            if boundary(text, match.start(), match.end())]


def strip_document(doc):
    return {key: doc[key] for key in ['pmc_id', 'pmid', 'patient', 'full_text'] if key in doc}


def make_prompts(doc):
    """This function explicitly discards gold fields before constructing inputs."""
    doc = strip_document(doc)
    patient_ids = [str(p['patient_id']) for p in doc['patient']]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError('Duplicate patient within document')
    mapping = {f'P{i+1:03d}': pid for i, pid in enumerate(patient_ids)}
    inverse = {pid: code for code, pid in mapping.items()}
    marked = dict(doc, patient=[dict(p, patient_id=inverse[str(p['patient_id'])]) for p in doc['patient']])
    segments = [dict(segment_id=f'S{i:03d}', offset=int(p['offset']), text=p['text'],
                     section_type=p.get('section_type', ''), type=p.get('type', ''))
                for i, p in enumerate(doc['full_text'])]
    for patient in doc['patient']:
        for mention in patient.get('mention', []):
            start, length = int(mention['offset']), int(mention['length'])
            if not any(s['offset'] <= start and start+length <= s['offset']+len(s['text'])
                       and s['text'][start-s['offset']:start-s['offset']+length] == mention['text']
                       for s in segments):
                raise ValueError('Patient mention does not match original text')
    report = '\n\n'.join(
        f"SEGMENT {s['segment_id']}; SECTION {s['section_type']} / {s['type']}; OFFSET {s['offset']}\n"
        + mark_patients(s['text'], s['offset'], marked) for s in segments)
    rows = []
    for code, pid in mapping.items():
        user = 'TARGET: ' + code + '\nSUPPLIED PATIENTS: ' + ', '.join(mapping) + '\n\nFULL REPORT:\n' + report
        rows.append({'row_id': f"{doc['pmc_id']}:patient-set:{code}", 'pmc_id': str(doc['pmc_id']),
                     'patient_id': pid, 'target_code': code, 'patient_map': mapping,
                     'segments': segments,
                     'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]})
    return rows


def atomize(value):
    return value.split(';') if COMPOUND.fullmatch(value) else [value]


def target_key(value, canonical):
    if HPO.fullmatch(value):
        return (canonical[value], None) if value in canonical else (None, 'hpo_outside_branch')
    if value.startswith('HP:') or ';HP:' in value:
        return None, 'malformed_hpo'
    if value == '-1' or not value:
        return None, 'invalid_unmapped_value'
    return value, None


def gap(left, right):
    return max(0, left[0]-right[1], right[0]-left[1])


def make_targets(doc, rows, canonical):
    patients = {str(p['patient_id']): p for p in doc['patient']}
    if 'entities' not in doc or 'association' not in doc:
        raise ValueError('Training document must provide entity and association labels')
    association_ids = [str(a['patient_id']) for a in doc['association']]
    if len(association_ids) != len(set(association_ids)) or set(association_ids) != set(patients):
        raise ValueError('Training association patient coverage differs')
    gold = {pid: set() for pid in patients}
    for association in doc.get('association', []):
        pid = str(association['patient_id'])
        if pid not in gold:
            raise ValueError('Unknown patient in training labels')
        gold[pid].update(x for value in association.get('phenotype', []) for x in atomize(str(value)))
    entity_map = defaultdict(list)
    for entity in doc.get('entities', []):
        if str(entity.get('note') or '').upper() == 'NO':
            continue
        raw = str(entity.get('identifier', ''))
        if raw == '-1':
            keys = [entity['text']]
        elif COMPOUND.fullmatch(raw):
            keys = [canonical[x] for x in raw.split(';') if x in canonical]
        else:
            continue
        for key in keys:
            entity_map[key].append(entity)
    decisions = []; targets = []
    for row in rows:
        pid = row['patient_id']; mentions = patients[pid].get('mention', [])
        accepted = {}
        for raw in sorted(gold[pid]):
            value, error = target_key(raw, canonical)
            decision = {'pmc_id': str(doc['pmc_id']), 'patient_id': pid, 'raw_target': raw,
                        'canonical_target': value}
            choices = []; rejected = Counter()
            if error is None:
                for entity in entity_map.get(value, []):
                    start = int(entity['offset']); end = start+int(entity['length']); quote = entity['text']
                    matching = [s for s in row['segments'] if s['offset'] <= start
                                and end <= s['offset']+len(s['text'])
                                and len(quote) == end-start
                                and s['text'][start-s['offset']:end-s['offset']] == quote]
                    if not matching:
                        rejected['nonliteral_gold_entity'] += 1; continue
                    segment = matching[0]
                    if not boundary(segment['text'], start-segment['offset'], end-segment['offset']):
                        rejected['word_boundary'] += 1; continue
                    distance = min((gap((start, end), (int(m['offset']), int(m['offset'])+int(m['length'])))
                                    for m in mentions), default=10**15)
                    choices.append((distance, start, end-start, quote, segment['segment_id']))
                if not choices:
                    error = 'no_positive_entity' if not entity_map.get(value) else 'no_literal_boundary_evidence'
            if error:
                decision.update(status='excluded', reason=error, rejected_entity_counts=dict(rejected))
            else:
                distance, start, length, quote, segment = min(choices)
                item = {'identifier': value if HPO.fullmatch(value) else '-1',
                        'segment_id': segment, 'quote': quote}
                accepted[(item['identifier'], '' if item['identifier'] != '-1' else quote)] = item
                decision.update(status='included', target=item, gold_entity_offset=start,
                                gold_entity_length=length, nearest_patient_mention_gap=distance,
                                attribution_verified=False, hpo_semantics_verified=False)
            decisions.append(decision)
        answer = {'phenotypes': [accepted[k] for k in sorted(accepted)]}
        targets.append(dict(row, answer=json.dumps(answer, ensure_ascii=False, separators=(',', ':'))))
    return targets, decisions


def prepare(args):
    docs = read_jsonl(args.train if args.command == 'prepare-train' else args.input)
    indexed(docs); _, valid, canonical, version = load_hpo(args.hpo)
    excluded = None
    if args.command == 'prepare-train':
        excluded = read_jsonl(args.excluded_input)
        if any('entities' in d or 'association' in d for d in excluded):
            raise ValueError('Excluded input must be label free')
        if set(indexed(docs)) & set(indexed(excluded)):
            raise ValueError('Training and excluded documents overlap')
    elif any('entities' in d or 'association' in d for d in docs):
        raise ValueError('Inference input must be label free')
    rows = []; decisions = []
    for doc in docs:
        prompts = make_prompts(doc)
        if args.command == 'prepare-train':
            supervised, audit = make_targets(doc, prompts, canonical)
            rows.extend(supervised); decisions.extend(audit)
        else:
            rows.extend(prompts)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    target = out / ('train.jsonl' if args.command == 'prepare-train' else 'prompts.jsonl')
    write_rows(target, rows)
    audit_path = out / 'target-decisions.jsonl'
    if decisions:
        write_rows(audit_path, decisions)
    counts = Counter(d['reason'] if d['status'] == 'excluded' else 'included' for d in decisions)
    stats = {'documents': len(docs), 'patients': len(rows), 'raw_atomic_target_units': len(decisions),
             'target_decisions': dict(counts)}
    if args.command == 'prepare-train':
        answers = [json.loads(r['answer'])['phenotypes'] for r in rows]
        stats.update(assistant_items=sum(map(len, answers)), empty_assistant_targets=sum(not a for a in answers))
    files = {name: sha256(getattr(args, name)) for name in
             (['train', 'excluded_input', 'hpo'] if args.command == 'prepare-train' else ['input', 'hpo'])}
    dump(out / 'manifest.json', {'configuration': vars(args), 'source_sha256': sha256(__file__),
         'marker_source_sha256': sha256(Path(__file__).with_name('prepare_rag.py')),
         'hpo_loader_source_sha256': sha256(Path(__file__).with_name('run.py')),
         'hpo_version': version, 'valid_concepts': len(valid), 'files': files,
         'document_ids': sorted(indexed(docs)), 'excluded_document_ids': sorted(indexed(excluded)) if excluded else [],
         'stats': stats, 'output_sha256': sha256(target),
         'decisions_sha256': sha256(audit_path) if decisions else None,
         'prompt_has_gold': False, 'candidate_hpo_input': False,
         'quote_selection': 'Training-only nearest given patient mention interval; ties offset/length/text/segment; no attribution proof.',
         'format': 'One row per patient, full original report, no truncation; all given patients marked.'})
    print(json.dumps(stats), flush=True)


def parse_response(response, row, valid):
    """Return raw valid IDs plus grounded NIL, and the strictly grounded subset."""
    stats = Counter(); raw = set(); grounded = set()
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return raw, grounded, {'invalid_json': 1}
    if not isinstance(value, dict) or set(value) != {'phenotypes'} or not isinstance(value['phenotypes'], list):
        return raw, grounded, {'invalid_schema': 1}
    segments = {s['segment_id']: s['text'] for s in row['segments']}
    stats['valid_json_schema'] += 1
    for item in value['phenotypes']:
        if not isinstance(item, dict) or set(item) != {'identifier', 'segment_id', 'quote'} or not all(isinstance(v, str) for v in item.values()):
            stats['invalid_item_schema'] += 1; continue
        identifier, segment, quote = item['identifier'], item['segment_id'], item['quote']
        if identifier == '-1' and quote == '-1':
            stats['invalid_nil_text'] += 1; continue
        if identifier != '-1' and identifier not in valid:
            stats['invalid_hpo'] += 1; continue
        if identifier != '-1':
            raw.add(identifier)
        supported = segment in segments and bool(literal_positions(segments[segment], quote))
        if not supported:
            stats['unsupported_quote'] += 1; continue
        association = quote if identifier == '-1' else identifier
        grounded.add(association)
        if identifier == '-1':
            raw.add(association)
        stats['grounded_items'] += 1
    return raw, grounded, dict(stats)


def merge(args):
    docs = read_jsonl(args.input); inputs = indexed(docs); baseline = indexed(read_jsonl(args.baseline))
    if set(inputs) != set(baseline) or any('entities' in d or 'association' in d for d in docs):
        raise ValueError('Baseline coverage differs or inference input has labels')
    _, valid, _, _ = load_hpo(args.hpo)
    expected = [r for doc in docs for r in make_prompts(doc)]
    prompts = read_rows(args.prompts)
    if prompts != expected:
        raise ValueError('Prompts/metadata must reconstruct exactly from label-free input')
    responses = read_rows(args.responses)
    response_map = {r['row_id']: r for r in responses}
    if len(response_map) != len(responses) or set(response_map) != {r['row_id'] for r in prompts}:
        raise ValueError('Response row coverage differs')
    predictions = {'raw': defaultdict(dict), 'grounded': defaultdict(dict)}; stats = Counter(); audits = []
    for row in prompts:
        response = response_map[row['row_id']]
        raw, grounded, counters = parse_response(response.get('response'), row, valid)
        hit_limit = bool(response.get('hit_token_limit', False))
        status = ('invalid_json' if counters.get('invalid_json') else
                  'invalid_schema' if counters.get('invalid_schema') else 'parsed')
        if hit_limit:
            counters['token_limit_grounded_row_rejected'] = 1
            grounded = set()
            status = 'token_limit_rejected'
        for mode, values in [('raw', raw), ('grounded', grounded)]:
            predictions[mode][row['pmc_id']][row['patient_id']] = sorted(values)
        stats.update(counters)
        audits.append({'row_id': row['row_id'], 'patient_id': row['patient_id'], 'counts': counters,
                       'raw_relations': len(raw), 'grounded_relations': len(grounded),
                       'hit_token_limit': hit_limit, 'grounded_row_status': status})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); outputs = {}
    for mode in ['raw', 'grounded']:
        rows = []
        for doc in docs:
            pmc = str(doc['pmc_id']); result = predictions[mode][pmc]
            association = [{'patient_id': p['patient_id'], 'phenotype': result[str(p['patient_id'])]}
                           for p in doc['patient']]
            rows.append(dict(baseline[pmc], association=association))
        path = out / (mode + '.predictions.jsonl'); write_rows(path, rows); outputs[mode] = sha256(path)
    write_rows(out / 'parse-audit.jsonl', audits)
    dump(out / 'manifest.json', {'configuration': vars(args), 'source_sha256': sha256(__file__),
         'files': {k: sha256(getattr(args, k)) for k in ['input', 'baseline', 'hpo', 'prompts', 'responses']},
         'outputs': outputs, 'stats': dict(stats), 'expected_patients': len(prompts),
         'raw_policy': 'Valid branch HPO IDs even if quote fails; NIL only when strict quote passes. Never bare -1. Parsed token-limit rows retained only as partial-set diagnostics.',
         'grounded_policy': 'Reject whole token-limit rows; otherwise valid branch HPO IDs or NIL text with whole-word literal quote in specified original segment. Invalid rows explicit empty predictions with separate audit status.',
         'entity_policy': 'Unchanged baseline entities.', 'gold_used': False,
         'limitations': 'Literal support proves neither patient attribution nor HPO semantic equivalence.'})
    print(json.dumps(dict(stats)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('prepare-train')
    for name in ['train', 'excluded-input', 'hpo', 'output-dir']:
        training.add_argument('--' + name, required=True)
    inference = sub.add_parser('prepare-inference')
    for name in ['input', 'hpo', 'output-dir']:
        inference.add_argument('--' + name, required=True)
    merging = sub.add_parser('merge')
    for name in ['input', 'baseline', 'hpo', 'prompts', 'responses', 'output-dir']:
        merging.add_argument('--' + name, required=True)
    args = parser.parse_args()
    (merge if args.command == 'merge' else prepare)(args)


if __name__ == '__main__':
    main()
