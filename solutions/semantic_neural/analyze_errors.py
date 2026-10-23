"""Read-only post-hoc diagnosis; never used by training or prediction."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from run import hpo_terms, norm, read_jsonl


def expand(value):
    return value.split(';') if re.fullmatch(r'HP:\d{7}(?:;HP:\d{7})*', value) else [value]


def associations(document):
    return {x['patient_id']: {v for item in x['phenotype'] for v in expand(item)} for x in document['association']}


def concepts(document):
    return {v for entity in document['entities'] if str(entity.get('note', '')).upper() != 'NO'
            for v in (expand(entity['identifier']) if entity['identifier'] != '-1' else [entity['text']])}


def main():
    parser = argparse.ArgumentParser()
    for name in ['train', 'gold', 'pred', 'hpo', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    terms, canonical = hpo_terms(args.hpo)
    aliases = {norm(label) for key, term in terms.items() if key != 'HP:0000118'
               for label in [term['name']] + term['synonyms'] if len(label) >= 3}
    for document in read_jsonl(args.train):
        for entity in document['entities']:
            valid = entity['identifier'] == '-1' or any(x.strip() in canonical for x in entity['identifier'].split(';'))
            if valid and str(entity.get('note', '')).upper() != 'NO' and len(entity['text'].strip()) >= 2:
                aliases.add(norm(entity['text']))
    predictions = {x['pmc_id']: x for x in read_jsonl(args.pred)}
    counts, examples, novel = Counter(), [], []
    for gold in read_jsonl(args.gold):
        predicted = predictions[gold['pmc_id']]
        detected, gold_concepts = concepts(predicted), concepts(gold)
        actual, guess = associations(gold), associations(predicted)
        for patient, truth in actual.items():
            assigned = guess[patient]
            counts['patients_empty_prediction'] += not assigned
            for concept in truth - assigned:
                counts['fn_recognized_but_not_assigned' if concept in detected else 'fn_missing_entity_concept'] += 1
            for concept in assigned - truth:
                counts['fp_gold_document_concept_wrong_patient' if concept in gold_concepts else 'fp_not_gold_document_concept'] += 1
            if len(truth - assigned) >= 10:
                examples.append({'pmc_id': gold['pmc_id'], 'patient_id': patient, 'gold': len(truth), 'predicted': len(assigned), 'missing_recognized': len((truth - assigned) & detected)})
        for entity in predicted['entities']:
            if norm(entity['text']) in aliases:
                counts['entities_with_known_lexical_surface'] += 1
                continue
            matched = any(x['offset'] == entity['offset'] and x['length'] == entity['length']
                          and str(x.get('note', '')).upper() != 'NO'
                          and (x['identifier'] == '-1' or set(expand(x['identifier'])) & set(expand(entity['identifier'])))
                          for x in gold['entities'])
            counts['dense_new_surface_entities'] += 1
            counts['dense_new_surface_matching_gold_span_id'] += bool(matched)
            novel.append({'pmc_id': gold['pmc_id'], 'text': entity['text'], 'identifier': entity['identifier'], 'matched': bool(matched)})
    result = {'purpose': 'Read-only diagnosis after frozen predictions; no model fitting or threshold selection.',
              'counts': dict(counts), 'large_patient_misses': examples[:8], 'novel_surface_examples': novel[:10],
              'caveat': 'Novel entity exact-match count is a diagnostic, not a separately official metric.'}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
