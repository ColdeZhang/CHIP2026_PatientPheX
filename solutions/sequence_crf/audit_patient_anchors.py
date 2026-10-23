#!/usr/bin/env python3
"""Training-only aggregate anchor coverage; distances are proxies, not coreference labels."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

import run as base
import prepare_pair_rag as prompts


def span(item):
    start = int(item['offset'])
    return start, start + int(item['length'])


def distance(left, right):
    return max(0, left[0] - right[1], right[0] - left[1])


def passage_index(doc, item):
    start, end = span(item)
    for index, passage in enumerate(doc['full_text']):
        offset = int(passage['offset'])
        if offset <= start < end <= offset + len(passage['text']):
            assert passage['text'][start-offset:end-offset] == item['text']
            return index
    raise ValueError('Span outside original passage')


def tagged_patients(context):
    # Patient IDs can themselves contain colons, so split at the literal ': '.
    return {pid for group in re.findall(r'\[PATIENT (.*?): ', context)
            for pid in group.split('/')}


def distribution(values):
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'max': max(values), 'mean': statistics.mean(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'excluded-input', 'sft-train', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    docs = base.read_jsonl(args.train)
    excluded = base.read_jsonl(args.excluded_input)
    assert all('entities' not in d and 'association' not in d for d in excluded)
    by_id = {str(d['pmc_id']): d for d in docs}
    assert len(by_id) == len(docs)
    assert not set(by_id) & {str(d['pmc_id']) for d in excluded}
    rows = base.read_jsonl(args.sft_train)
    actual = {}
    for row in rows:
        pmc = str(row['pmc_id'])
        assert pmc in by_id
        text = row['messages'][-1]['content']
        assert text.startswith('Patient identities and global anchors:\n')
        anchors, payload = text.split('\n\nCandidate phenotypes:\n', 1)
        assert anchors.split('\n', 1)[1] == prompts.patient_anchors(by_id[pmc])
        items = {item['code']: item for item in json.loads(payload)}
        for code, concept in row['code_map'].items():
            key = pmc, concept
            assert key not in actual
            actual[key] = items[code]['evidence']
        gold = base.gold_associations(by_id[pmc])
        expected = {code: [pid for pid in row['patient_ids'] if hp in gold.get(pid, set())]
                    for code, hp in row['code_map'].items()}
        assert json.loads(row['answer']) == expected
    counts = Counter()
    by_stratum = defaultdict(Counter)
    mention_counts, patient_counts, densities, distances = [], [], [], []
    document_summaries = []
    for pmc, doc in by_id.items():
        patient_counts.append(len(doc['patient']))
        chars = sum(len(p['text']) for p in doc['full_text'])
        entity_map = defaultdict(list)
        for entity in doc['entities']:
            if str(entity.get('note') or '').upper() == 'NO':
                continue
            passage_index(doc, entity)
            for key in base.association_keys(entity):
                entity_map[key].append(entity)
        gold = base.gold_associations(doc)
        anchors = {}
        for patient in doc['patient']:
            pid = str(patient['patient_id'])
            mentions = patient.get('mention', [])
            for mention in mentions:
                passage_index(doc, mention)
            anchors[pid] = mentions
            mention_counts.append(len(mentions))
            counts['patients'] += 1
            counts['patient_mentions'] += len(mentions)
            counts['patients_at_most_1_mention'] += len(mentions) <= 1
            counts['patients_at_most_3_mentions'] += len(mentions) <= 3
            counts['patients_all_mentions_in_one_passage'] += len({passage_index(doc, m) for m in mentions}) <= 1
        densities.append(sum(len(m) for m in anchors.values()) / chars * 1000)
        local = Counter()
        group = 'one_patient' if len(doc['patient']) == 1 else 'multiple_patients'
        for pid, concepts in gold.items():
            mentions = anchors[str(pid)]
            for concept in concepts:
                fields = Counter({'gold_relations': 1})
                entities = entity_map.get(concept, [])
                if not entities:
                    fields['no_matching_positive_gold_entity'] += 1
                else:
                    fields['has_matching_positive_gold_entity'] += 1
                    gap = min((distance(span(e), span(m)) for e in entities for m in mentions), default=None)
                    if gap is not None:
                        distances.append(gap)
                        for threshold in [100, 500, 1000, 2000]:
                            fields[f'any_gold_span_within_{threshold}_chars_of_target'] += gap <= threshold
                        fields['every_gold_span_beyond_500_chars_of_target'] += gap > 500
                    else:
                        fields['target_has_no_provided_mentions'] += 1
                    same = any(passage_index(doc, e) == passage_index(doc, m) for e in entities for m in mentions)
                    fields['any_gold_span_same_passage_as_target'] += same
                    visible = any(str(pid) in tagged_patients(c['context']) for e in entities
                                  for c in prompts.evidence(doc, [e], 500))
                    fields['any_gold_span_target_marker_visible_radius500'] += visible
                    fields['no_gold_span_target_marker_visible_radius500'] += not visible
                selected = actual.get((pmc, concept))
                if selected is None:
                    fields['not_in_sft_candidates'] += 1
                else:
                    fields['in_sft_candidates'] += 1
                    present = set().union(*(tagged_patients(c['context']) for c in selected))
                    target_visible = str(pid) in present
                    fields['sft_selected_evidence_target_marker_visible'] += target_visible
                    fields['sft_selected_evidence_target_marker_absent'] += not target_visible
                    fields['sft_selected_evidence_no_patient_marker'] += not present
                    fields['sft_selected_evidence_only_other_patient_markers'] += bool(present) and not target_visible
                counts.update(fields)
                local.update(fields)
                by_stratum[group].update(fields)
        document_summaries.append({'pmc_id': pmc, 'patients': len(doc['patient']),
                                   'provided_mentions': sum(len(m) for m in anchors.values()),
                                   'text_chars': chars, 'counts': dict(local)})
    result = {'scope': 'outer fold0 train48 only; aggregate gold-aware diagnostic, not model score',
              'configuration': vars(args), 'counts': dict(counts),
              'strata': {k: dict(v) for k, v in by_stratum.items()},
              'patient_mention_count': distribution(mention_counts),
              'document_patient_count': distribution(patient_counts),
              'document_mentions_per_1000_chars': distribution(densities),
              'minimum_gold_entity_to_target_mention_gap_chars': distribution(distances),
              'documents': len(docs), 'sft_prompts': len(rows),
              'source_sha256': base.sha(__file__), 'base_sha256': base.sha(base.__file__),
              'prompt_source_sha256': base.sha(prompts.__file__),
              'files': {k: base.sha(getattr(args, k)) for k in ['train', 'excluded_input', 'sft_train']},
              'limitations': [
                  'A gold patient-concept association has no gold evidence-span pointer; any positive matching concept span is used. This is optimistic anchor availability, not proof that a span belongs to that patient.',
                  'Raw evaluator-compatible compound splitting and NIL text keys; no alt-ID canonicalization or score change.',
                  'Distances use interval gaps over original global coordinates. Visible markers additionally require the original patient mention to fit inside the same passage-clipped radius500 context.',
                  'Actual SFT selected contexts are read from the frozen 204 prompts. Global patient anchor text is present separately even when no target marker is in local candidate evidence.',
                  'No full-document LLM coreference extension was used in this route or its SFT data. O-ID control was separate and not adopted.',
                  'No gold labels are written into inference resources or used to generate new candidates.'
              ], 'per_document': document_summaries}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['per_document', 'limitations']}))


if __name__ == '__main__':
    main()
