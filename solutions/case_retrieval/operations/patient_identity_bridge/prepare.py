#!/usr/bin/env python3
"""Build a training-only, label-independent patient-identity input control.

This entry point has no model, optimizer, evaluator, remote or GPU interface.
The original CE function is reused through a restricted AST extraction. All
active baseline token IDs, types and positions remain an exact prefix; bridge
text is appended after its existing final SEP, never replacing old evidence.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, Counter
import argparse
import ast
import datetime
import hashlib
import json
import math
import re
import time
from tokenizers import Tokenizer


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def checked_source(ref):
    path = Path(ref['path'])
    require(path.is_file() and sha(path) == ref['sha256'], 'Source changed: ' + str(path))
    return path


def helpers(base_path, ce_path, hpo_path):
    wanted = {'parse_hpo', 'passage_for', 'sentence_bounds', 'pair_features',
              'association_keys', 'gold_associations'}
    namespace = {'re': re, 'math': math, 'Path': Path,
                 'FAMILY': re.compile(r'\b(?:mother|father|sister|brother|parent|parents|sibling)\b', re.I),
                 'COMPOUND_HPO': re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')}
    nodes = [node for node in ast.parse(base_path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    require({node.name for node in nodes} == wanted, 'Missing baseline helper')
    for node in nodes:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(base_path), 'exec'), namespace)
    terms, allowed, alt, _ = namespace['parse_hpo'](str(hpo_path))

    def canonical(identifier):
        if str(identifier) == '-1':
            return '-1'
        parts = [alt.get(x.strip(), x.strip()) for x in str(identifier).split(';')]
        parts = sorted(set(x for x in parts if x in allowed))
        return ';'.join(parts) if parts else None

    linker = SimpleNamespace(terms=terms, allowed=allowed, alt=alt, canonical=canonical)
    base = SimpleNamespace(**{name: namespace[name] for name in wanted})
    node = next(node for node in ast.parse(ce_path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'make_examples')
    env = {'base': base, 'defaultdict': defaultdict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ce_path), 'exec'), env)
    make = env['make_examples']
    # Add provenance only to the exact old selection loop, without changing it.
    original = ast.get_source_segment(ce_path.read_text(), node)
    replacements = [
        ('contexts=[];seen=set()', 'contexts=[];seen=set();trace=[]'),
        ("contexts.append(str(p.get('section_type',''))+': '+snippet)",
         "trace.append(dict(passage_index=pi, local_start=a, local_end=b, occurrence=dict(e)))\n"
         "                contexts.append(str(p.get('section_type',''))+': '+snippet)"),
        ("if training:row['label']=int(key in gold.get(pid,set()))",
         "row['trace']=trace\n            if training:row['label']=int(key in gold.get(pid,set()))")]
    traced = original
    for old, new in replacements:
        require(traced.count(old) == 1, 'Provenance instrumentation no longer exact')
        traced = traced.replace(old, new)
    traced_env = dict(env)
    exec(compile(traced, str(ce_path) + ':provenance-only', 'exec'), traced_env)
    return base, linker, make, traced_env['make_examples']


def anchors_from_input(doc, base):
    """Only supplied identities at byte-exact character coordinates are usable."""
    anchors = []
    rejected = []
    for patient in doc.get('patient', []):
        for mention in patient.get('mention', []):
            offset = int(mention['offset'])
            length = int(mention.get('length', len(mention.get('text', ''))))
            index, passage = base.passage_for(doc, offset)
            local = offset - int(passage.get('offset', 0))
            text = passage.get('text', '')
            item = dict(patient_id=patient['patient_id'], offset=offset, length=length,
                        text=mention.get('text', ''), passage_index=index)
            if index < 0 or length <= 0 or local < 0 or local + length > len(text) or text[local:local + length] != item['text']:
                rejected.append(item)
            else:
                anchors.append(item)
    unique = {(a['patient_id'], a['offset'], a['length']): a for a in anchors}
    return sorted(unique.values(), key=lambda a: (a['offset'], a['patient_id'], a['length'])), rejected


def bridge_candidates(doc, row, anchors, base):
    """Nearest actual anchor, never nearest *target*; no inferred assignment."""
    passages = doc['full_text']
    choices = []
    decisions = []
    seen = set()
    for rank, trace in enumerate(row['trace']):
        index = trace['passage_index']
        occurrence = trace['occurrence']
        occurrence_offset = int(occurrence['offset'])
        reason = dict(rank=rank, occurrence_offset=occurrence_offset, passage_index=index)
        if index < 0:
            decisions.append(dict(reason, reason='occurrence_outside_input'))
            continue
        passage = passages[index]
        candidates = [a for a in anchors if a['passage_index'] == index]
        scope = 'same_passage'
        if not candidates and index > 0:
            previous = passages[index - 1]
            if previous.get('type') == passage.get('type') == 'paragraph' and previous.get('section_type', '') == passage.get('section_type', ''):
                candidates = [a for a in anchors if a['passage_index'] == index - 1]
                scope = 'immediately_previous_same_type_and_section'
        if not candidates:
            decisions.append(dict(reason, reason='no_anchor_within_structural_boundary'))
            continue
        nearest_distance = min(abs(a['offset'] - occurrence_offset) for a in candidates)
        tied = [a for a in candidates if abs(a['offset'] - occurrence_offset) == nearest_distance]
        tie_clauses = set()
        for item in tied:
            source = passages[item['passage_index']]
            left, right = base.sentence_bounds(source['text'], item['offset'] - int(source['offset']))
            tie_clauses.add((item['passage_index'], left, right))
        if len({item['patient_id'] for item in tied}) > 1 and len(tie_clauses) > 1:
            decisions.append(dict(reason, reason='equidistant_different_patient_different_clause'))
            continue
        anchor = min(tied, key=lambda a: (a['offset'], a['patient_id']))
        # If the selected anchor was already in any old selected window, do not
        # repeat it merely because the occurrence-local window differs.
        if any(t['passage_index'] == anchor['passage_index'] and
               t['local_start'] + int(passages[t['passage_index']]['offset']) <= anchor['offset'] and
               anchor['offset'] + anchor['length'] <= t['local_end'] + int(passages[t['passage_index']]['offset'])
               for t in row['trace']):
            decisions.append(dict(reason, reason='anchor_already_in_old_selected_window'))
            continue
        source = passages[anchor['passage_index']]
        source_offset = int(source['offset'])
        text = source['text']
        left, right = base.sentence_bounds(text, anchor['offset'] - source_offset)
        if right < anchor['offset'] - source_offset + anchor['length']:
            decisions.append(dict(reason, reason='anchor_crosses_source_clause_boundary'))
            continue
        identities = [a for a in anchors if a['passage_index'] == anchor['passage_index'] and
                      source_offset + left <= a['offset'] and a['offset'] + a['length'] <= source_offset + right]
        require(anchor in identities, 'Selected anchor missing from clause')
        touching = [a for a in anchors if a['passage_index'] == anchor['passage_index'] and
                    a['offset'] < source_offset + right and source_offset + left < a['offset'] + a['length']]
        if any(a['patient_id'] != b['patient_id'] and a['offset'] < b['offset'] + b['length'] and
               b['offset'] < a['offset'] + a['length'] for a in identities for b in touching):
            decisions.append(dict(reason, reason='conflicting_overlapping_patient_identity'))
            continue
        # Multiple nonoverlapping identities in a clause are all retained;
        # the bridge does not resolve coreference or create a relation.
        key = (anchor['passage_index'], left, right)
        if key in seen:
            decisions.append(dict(reason, reason='duplicate_source_clause'))
            continue
        seen.add(key)
        quote = text[left:right]
        identity_text = '; '.join(('TARGET' if a['patient_id'] == row['patient_id'] else 'OTHER') +
                                  ' ' + str(a['patient_id']) + ' at ' + str(a['offset']) for a in identities)
        bridge = ('Evidence ' + str(rank + 1) + ' at ' + str(occurrence_offset) + '; ' +
                  'anchor passage ' + str(anchor['passage_index']) + ' ' + str(source.get('section_type', '')) +
                  '; ' + identity_text + '; distance ' + str(anchor['offset'] - occurrence_offset) +
                  '. Source: ' + quote)
        item = dict(reason, structural_scope=scope, anchor=dict(anchor), identities=identities,
                    source_passage_index=anchor['passage_index'], source_start=source_offset + left,
                    source_end=source_offset + right, quote=quote, text=bridge,
                    source_type=source.get('type', ''), source_section_type=source.get('section_type', ''))
        choices.append(item)
        decisions.append(dict(reason, reason='candidate_bridge_clause', source_start=item['source_start']))
    return choices, decisions


def pack(tokenizer, row, choices, maximum=512, bridge_budget=128, raw_tokenizer=None):
    encoded = tokenizer.encode(row['header'], row['evidence'])
    old_ids, old_types = encoded.ids, encoded.type_ids
    require(0 < len(old_ids) <= 384 and old_ids[-1] == 3, 'Unexpected old CE encoding/SEP')
    raw = raw_tokenizer
    if raw is None:
        raw = Tokenizer.from_str(tokenizer.to_str())
        raw.no_padding()
        raw.no_truncation()
    text = ''
    kept = []
    rejected = []
    for choice in choices:
        proposed = text + ('\n' if text else '') + choice['text']
        proposal_ids = raw.encode(proposed, add_special_tokens=False).ids
        if len(proposal_ids) > bridge_budget - 1:
            rejected.append(dict(choice, reject_reason='whole_clause_exceeds_remaining_127_token_budget'))
            continue
        text = proposed
        kept.append(choice)
    extra = raw.encode(text, add_special_tokens=False).ids + [3] if text else []
    require(len(extra) <= bridge_budget and len(old_ids) + len(extra) <= maximum, 'Bridge over budget')
    pad = tokenizer.token_to_id('[PAD]')
    require(pad == 0, 'Unexpected pad ID')
    result = {}
    for arm in ['baseline', 'bridge']:
        addition = extra if arm == 'bridge' else []
        active = len(old_ids) + len(addition)
        ids = old_ids + addition + [pad] * (maximum - active)
        types = old_types + [1] * len(addition) + [0] * (maximum - active)
        mask = [1] * active + [0] * (maximum - active)
        require(ids[:len(old_ids)] == old_ids and types[:len(old_ids)] == old_types, 'Baseline prefix changed')
        result[arm] = dict(input_ids=ids, token_type_ids=types, attention_mask=mask)
    return dict(arms=result, old_active_tokens=len(old_ids), bridge_active_tokens=len(extra),
                kept_bridges=kept, rejected_bridges=rejected, bridge_text=text)


def prepare(plan_path, output):
    started = time.monotonic()
    require(not output.exists(), 'Use a new output directory; never overwrite a CPU freeze')
    plan = json.loads(plan_path.read_text())
    require(plan['scope'] == 'train48_only_CPU_no_model_no_outer_or_originaldev', 'Wrong scope')
    require(plan['source_sha256'] == sha(__file__), 'Plan/source snapshot mismatch')
    sources = {name: checked_source(ref) for name, ref in plan['sources'].items()}
    docs = read_rows(sources['train48'])
    require(len(docs) == len({str(d['pmc_id']) for d in docs}) == 48, 'Wrong training documents')
    predictions = {}
    for role in ['candidates_inner1', 'candidates_inner2', 'candidates_inner3']:
        for prediction in read_rows(sources[role]):
            key = str(prediction['pmc_id'])
            require(key not in predictions, 'Duplicate candidate document')
            predictions[key] = prediction['entities']
    require(set(predictions) == {str(d['pmc_id']) for d in docs}, 'Candidate document coverage changed')
    base, linker, make, traced_make = helpers(sources['base_source'], sources['ce_source'], sources['hpo'])
    tokenizer = Tokenizer.from_file(str(sources['tokenizer']))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer.enable_truncation(max_length=384, strategy='only_second', direction='right')
    require(tokenizer.get_vocab_size() == 30522, 'Do not add marker embeddings in this control')
    raw_tokenizer = Tokenizer.from_str(tokenizer.to_str())
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    all_inputs = []
    labels = []
    counts = Counter()
    rejections = Counter()
    input_ids = set()
    invalid_anchors = []
    for original in docs:
        # Annotation-bearing original is used only by the separate labels step.
        doc = {key: value for key, value in original.items() if key not in {'entities', 'association'}}
        require('entities' not in doc and 'association' not in doc, 'Annotations passed to input builder')
        for left, right in zip(doc['full_text'], doc['full_text'][1:]):
            require(int(left['offset']) + len(left['text']) <= int(right['offset']), 'Passage order/overlap invalid')
        entities = predictions[str(doc['pmc_id'])]
        old = make(doc, entities, linker, training=False)
        traced = traced_make(doc, entities, linker, training=False)
        require(old == [{key: value for key, value in row.items() if key != 'trace'} for row in traced], 'Instrumented old selection changed')
        anchors, invalid = anchors_from_input(doc, base)
        invalid_anchors.extend([dict(pmc_id=str(doc['pmc_id']), **a) for a in invalid])
        for row in traced:
            key = (str(row['pmc_id']), row['patient_id'], row['concept'])
            require(key not in input_ids, 'Duplicate patient-concept row')
            input_ids.add(key)
            choices, decisions = bridge_candidates(doc, row, anchors, base)
            packed = pack(tokenizer, row, choices, raw_tokenizer=raw_tokenizer)
            record = dict(pmc_id=key[0], patient_id=key[1], concept=key[2], header=row['header'], evidence=row['evidence'],
                          old_occurrences=row['trace'], decisions=decisions, **packed)
            require('label' not in record, 'Label leaked into input artifact')
            all_inputs.append(record)
            for decision in decisions:
                rejections[decision['reason']] += 1
            rejections.update(x['reject_reason'] for x in packed['rejected_bridges'])
    # A second pass first accesses relations only after ALL input construction.
    # Labels never influence ranking, anchor selection, packing or acceptance.
    gold_by_document = {str(original['pmc_id']): base.gold_associations(original, linker) for original in docs}
    for row in all_inputs:
        gold = gold_by_document[row['pmc_id']]
        labels.append(dict(pmc_id=row['pmc_id'], patient_id=row['patient_id'], concept=row['concept'],
                           label=int(row['concept'] in gold.get(row['patient_id'], set()))))
    require(len(all_inputs) == 4127 and len(labels) == 4127 and sum(r['label'] for r in labels) == 969, 'Candidate/label contract changed')
    sft = read_rows(sources['sft_rows'])
    expected_keys = set()
    for row in sft:
        # Public preparation stores every CID map with all supplied patient IDs.
        for concept in row['code_map'].values():
            for patient in row['patient_ids']:
                expected_keys.add((str(row['pmc_id']), patient, concept))
    require(input_ids == expected_keys, 'Candidate keys differ from old frozen SFT')
    for record, label in zip(all_inputs, labels):
        require(all(record[k] == label[k] for k in ['pmc_id', 'patient_id', 'concept']), 'Label-row mismatch')
        own = '[unused1]' in record['evidence']
        added_target = any(any(a['patient_id'] == record['patient_id'] for a in bridge['identities']) for bridge in record['kept_bridges'])
        flags = dict(rows=1, old_active_tokens=record['old_active_tokens'], extra_active_tokens=record['bridge_active_tokens'],
                     bridge_rows=bool(record['kept_bridges']), added_target_identity_rows=added_target,
                     no_old_target_marker_rows=not own, newly_visible_target_rows=not own and added_target,
                     same_passage_bridge_rows=any(x['structural_scope'] == 'same_passage' for x in record['kept_bridges']),
                     previous_passage_bridge_rows=any(x['structural_scope'] != 'same_passage' for x in record['kept_bridges']))
        for key, value in flags.items():
            counts[key] += int(value)
            if label['label']:
                counts['positive_' + key] += int(value)
    output.mkdir(parents=True)
    for name, rows in [('inputs.jsonl', all_inputs), ('labels.jsonl', labels)]:
        with (output / name).open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    actual = dict(status='complete_cpu_only_no_quality_result', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  plan_sha256=sha(plan_path), source_sha256=sha(__file__), counts=dict(counts), selection_decisions=dict(rejections),
                  invalid_supplied_anchor_count=len(invalid_anchors), invalid_supplied_anchors=invalid_anchors,
                  all_old_active_ids_types_positions_unchanged=True, original_final_sep_unchanged=True,
                  baseline_and_bridge_padded_tokens=512, maximum_added_tokens=max(r['bridge_active_tokens'] for r in all_inputs),
                  same4127_candidate_keys=True, labels_canonical969=True, input_builder_annotation_access=False,
                  model_loaded=False, gpu_used=False, outer_or_originaldev_read=False,
                  tokenizer_version=__import__('tokenizers').__version__, tokenizer_backend_only=True,
                  training_steps_if_4epochs_batch16=4 * math.ceil(len(all_inputs) / 16),
                  training_time_measured=False, cpu_seconds=time.monotonic() - started,
                  outputs={name: {'path': str(output / name), 'bytes': (output / name).stat().st_size, 'sha256': sha(output / name)}
                           for name in ['inputs.jsonl', 'labels.jsonl']}, sources=plan['sources'],
                  limitations=['Anchor visibility is only a structural proxy, not proof of correct relation evidence.',
                               'Anchor source clauses can also contain clinical information; this is an identity-context intervention, not pure identity-token isolation.',
                               'Prior384 tokens are unchanged; token-level appended context requires the same custom collator in any future GPU arm.',
                               'No learned identity resolution, no new candidate spans or aliases, no automatic case-wide association.',
                               'Old CE trained on gold entities. Both future paired arms would instead use these same OOF candidates; old quality is not a single-variable baseline.'])
    write_new(output / 'cpu-result.json', actual)
    print(json.dumps(dict(path=str(output / 'cpu-result.json'), sha256=sha(output / 'cpu-result.json'), counts=dict(counts), cpu_seconds=actual['cpu_seconds']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
