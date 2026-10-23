#!/usr/bin/env python3
"""Aggregate only outer training labels against already frozen nested OOF candidates.

This is a planning diagnostic, not a new validation score or inference resource.
No per-example gold facts or corrected predictions are written.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import run as base
import prepare_pair_rag as prompts
import audit_patient_anchors
from audit_patient_anchors import tagged_patients

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import evaluate


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[0] + a[1], b[0] + b[1])


def missing_units(truth, guess):
    """Same one-to-one NIL wildcard consumption as evaluator v2."""
    for span, gold_ids in truth.items():
        left, right = set(gold_ids), set(guess.get(span, set()))
        common = left & right
        left -= common
        right -= common
        if '-1' in left and right:
            left.remove('-1')
        for identifier in left:
            yield span, identifier


def diagnose(doc, pred, seen_ids, seen_aliases):
    count = Counter()
    truth = evaluate.positive_units(doc, doc, 'all')
    guess = evaluate.positive_units(pred, doc, 'all')
    gold_entities = {(int(e['offset']), int(e['length'])): e for e in doc['entities']
                     if str(e.get('note') or '').upper() != 'NO'}
    predicted_aliases = defaultdict(set)
    for entity in pred['entities']:
        for identifier in entity['identifier'].split(';'):
            predicted_aliases[base.norm(entity['text'])].add(identifier)
    tp, fp, fn = evaluate.mention_counts(truth, guess)
    count.update(entity_tp=tp, entity_fp=fp, entity_fn=fn,
                 gold_entity_spans=len(truth), predicted_entity_spans=len(guess))
    for span, hid in missing_units(truth, guess):
        text = base.norm(gold_entities[span]['text'])
        if span in guess:
            category = 'exact_span_wrong_identifier'
        elif any(overlaps(span, other) for other in guess):
            category = 'boundary_or_nested_overlap'
        else:
            category = 'no_overlapping_prediction'
        count['entity_fn_' + category] += 1
        count['entity_fn_identifier_seen_in_nested_train'] += hid in seen_ids
        count['entity_fn_surface_seen_in_nested_train'] += text in seen_aliases
        count['entity_fn_surface_and_identifier_seen_in_nested_train'] += hid in seen_aliases.get(text, set())
        count['entity_fn_same_document_predicted_surface_and_identifier'] += hid in predicted_aliases.get(text, set())
    assert sum(count['entity_fn_' + k] for k in ['exact_span_wrong_identifier', 'boundary_or_nested_overlap', 'no_overlapping_prediction']) == fn
    groups = prompts.grouped_concepts(pred['entities'])
    gold_groups = prompts.grouped_concepts([e for e in doc['entities'] if str(e.get('note') or '').upper() != 'NO'])
    expected, proposed = evaluate.associations(doc), evaluate.associations(pred)
    all_gold_concepts = set().union(*expected.values())
    count['candidate_concepts'] += len(groups)
    count['candidate_concepts_with_more_than4_spans'] += sum(len(es) > 4 for es in groups.values())
    selected_markers, any_markers = {}, {}
    for concept, entities in groups.items():
        selected_markers[concept] = set().union(*(tagged_patients(e['context']) for e in prompts.evidence(doc, entities, 500)))
        any_markers[concept] = set().union(*(tagged_patients(e['context']) for entity in entities for e in prompts.evidence(doc, [entity], 500)))
    for patient, concepts in expected.items():
        guess_concepts = proposed[patient]
        tp, fp, fn = evaluate.compare_sets(concepts, guess_concepts)
        count.update(relation_tp=tp, relation_fp=fp, relation_fn=fn, gold_relations=len(concepts))
        for concept in concepts:
            covered = concept in groups
            count['gold_relation_candidate_covered' if covered else 'gold_relation_candidate_absent'] += 1
            if not covered:
                count['candidate_absent_relation_has_positive_gold_entity'] += concept in gold_groups
            else:
                count['covered_positive_relation_selected_target_marker'] += str(patient) in selected_markers[concept]
                count['covered_positive_relation_any_occurrence_target_marker'] += str(patient) in any_markers[concept]
                count['covered_positive_relation_marker_lost_by4_context_cap'] += str(patient) in any_markers[concept] and str(patient) not in selected_markers[concept]
            count['gold_relation_concept_shared_across_patients'] += sum(concept in hs for hs in expected.values()) > 1
        for concept in concepts - guess_concepts:
            count['relation_fn_candidate_present' if concept in groups else 'relation_fn_candidate_absent'] += 1
        for concept in guess_concepts - concepts:
            if concept in all_gold_concepts:
                category = 'wrong_patient_but_gold_for_another'
            elif concept in gold_groups:
                category = 'gold_entity_but_no_patient_relation'
            else:
                category = 'concept_absent_from_positive_gold_entities_and_relations'
            count['relation_fp_' + category] += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.sft_manifest).read_text())
    train_path = manifest['configuration']['train']
    assert base.sha(train_path) == manifest['files']['train']
    docs = {str(d['pmc_id']): d for d in base.read_jsonl(train_path)}
    assert set(docs) == set(manifest['training_document_ids'])
    excluded = set(manifest['excluded_document_ids'])
    assert not set(docs) & excluded
    total, strata, folds, seen, predictions = Counter(), defaultdict(Counter), [], set(), []
    sources = []
    for item in manifest['provenance']:
        job = item['oof_job']
        command = job['command']
        source_train = command[command.index('--train')+1]
        fit = base.read_jsonl(source_train)
        train_ids = {str(d['pmc_id']) for d in fit}
        assert train_ids == set(job['train_ids'])
        candidate_path = item['augment_command'][item['augment_command'].index('--output')+1]
        assert base.sha(candidate_path) == item['candidate_sha256']
        assert base.sha(source_train) == job['train_sha256']
        rows = base.read_jsonl(candidate_path)
        ids = {str(d['pmc_id']) for d in rows}
        assert ids == set(job['input_ids']) and len(ids) == len(rows)
        assert not train_ids & ids and train_ids | ids == set(docs)
        assert not ids & seen and not (ids | train_ids) & excluded
        seen |= ids
        seen_ids, aliases = set(), defaultdict(set)
        for d in fit:
            for e in d['entities']:
                if str(e.get('note') or '').upper() == 'NO':
                    continue
                hids = set(e['identifier'].split(';'))
                seen_ids |= hids
                aliases[base.norm(e['text'])] |= hids
        local = Counter()
        for pred in rows:
            doc = docs[str(pred['pmc_id'])]
            count = diagnose(doc, pred, seen_ids, aliases)
            local.update(count)
            strata['one_patient' if len(doc['patient']) == 1 else 'multiple_patients'].update(count)
        total.update(local)
        predictions.extend(rows)
        folds.append({'inner_fold': item['inner_fold'], 'documents': len(rows), 'counts': dict(local)})
        sources.append({'candidate_path': candidate_path, 'candidate_sha256': item['candidate_sha256'],
                        'training_path': source_train, 'training_sha256': job['train_sha256']})
    assert seen == set(docs) and len(docs) == 48
    metrics = evaluate.evaluate(list(docs.values()), predictions)
    assert total['gold_relations'] == manifest['stats']['gold_relations_total']
    assert total['gold_relation_candidate_absent'] == manifest['stats']['gold_relations_not_in_candidates']
    assert total['relation_fn'] == total['relation_fn_candidate_present'] + total['relation_fn_candidate_absent']
    result = {'scope': 'outer train48 nested train32->16 OOF aggregate diagnostic, not original dev or outer fold0 validation',
              'manifest': args.sft_manifest, 'manifest_sha256': base.sha(args.sft_manifest),
              'train_sha256': base.sha(train_path), 'source_sha256': base.sha(__file__),
              'dependencies': {str(Path(p).relative_to(Path(__file__).resolve().parents[2])): base.sha(p)
                               for p in [base.__file__, prompts.__file__, evaluate.__file__, audit_patient_anchors.__file__]},
              'frozen_oof_rule_metrics': metrics['metrics'], 'counts': dict(total), 'folds': folds,
              'strata': {k: dict(v) for k, v in strata.items()}, 'sources': sources,
              'limitations': ['Gold-aware aggregate planning only; no new prediction, gold candidate, or per-example correction exported.',
                             'Patient relation labels are document bags, without human evidence-span pointers; local marker availability is not attribution correctness.',
                             'Same-document exact normalized surface+identifier FN count is a propagation opportunity upper bound, not a tested gain.',
                             'Nested training32 models use less training data than final training48; thresholds originally selected on development fold0, so no blind-test claim.',
                             'All reported splits are members of outer training48; no new dev or outer-fold0 gold read.']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'metrics': metrics['metrics'], 'counts': dict(total)}))


if __name__ == '__main__':
    main()
