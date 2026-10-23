#!/usr/bin/env python3
"""Training48-only aggregate audit of frozen nested OOF misses and HPO/Sap coverage.

No new predictions, GPU inference, gold surfaces or case-specific corrections are
exported. Reuse the exact evaluator-v2 missing-unit definition of the old audit.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import gzip
import json
from pathlib import Path
import re

import run as base
import diagnose_nested_losses as old
from combine import ontology


def exact_ontology(path, allowed):
    names, exact = defaultdict(set), defaultdict(set)
    identifier = None
    for line in Path(path).read_text().splitlines():
        if line == '[Term]':
            identifier = None
        elif line.startswith('id: '):
            identifier = line[4:]
        elif identifier in allowed:
            if line.startswith('name: '):
                names[base.norm(line[6:])].add(identifier)
            elif line.startswith('synonym: '):
                match = re.match(r'synonym: "((?:[^"\\]|\\.)*)"\s+EXACT(?:\s|$)', line)
                if match:
                    exact[base.norm(match[1].replace('\\"', '"'))].add(identifier)
    combined = defaultdict(set)
    for source in [names, exact]:
        for text, ids in source.items():
            combined[text].update(ids)
    return names, exact, combined


def flags(doc, candidate):
    if candidate['exact']:
        return 'exact_excluded'
    if candidate['probability'] < .65:
        return 'probability_below_0.65'
    if candidate['similarity'] < .92:
        return 'similarity_below_0.92'
    _, passage = base.passage_for(doc, candidate['offset'])
    start = candidate['offset'] - int(passage.get('offset', 0))
    if ontology.is_negated(passage['text'], start, start + candidate['length']):
        return 'negation_rejected'
    return 'passes_pre_overlap_gates'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-manifest', required=True)
    parser.add_argument('--hpo', required=True)
    parser.add_argument('--prepared-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.sft_manifest).read_text())
    train_path = manifest['configuration']['train']
    assert base.sha(train_path) == manifest['files']['train']
    assert base.sha(args.hpo) == manifest['files']['hpo']
    docs = {str(doc['pmc_id']): doc for doc in base.read_jsonl(train_path)}
    assert len(docs) == 48 and set(docs) == set(manifest['training_document_ids'])
    assert not set(docs) & set(manifest['excluded_document_ids'])
    _, allowed, alt, version = base.parse_hpo(args.hpo)
    allowed.discard('HP:0000118')
    names, exact, combined = exact_ontology(args.hpo, allowed)
    totals, folds, sources, seen = defaultdict(Counter), [], [], set()
    for item in manifest['provenance']:
        command = item['augment_command']
        get = lambda name: command[command.index('--' + name) + 1]
        source_train, raw_path, raw_manifest_path = get('train'), get('candidates'), get('candidate-manifest')
        prediction_path = get('output')
        assert base.sha(source_train) == item['oof_job']['train_sha256']
        assert base.sha(prediction_path) == item['candidate_sha256']
        assert base.sha(raw_manifest_path) == item['semantic_manifest_sha256']
        raw_manifest = json.loads(Path(raw_manifest_path).read_text())
        assert base.sha(raw_path) == raw_manifest['output']['sha256']
        assert raw_manifest['inputs']['train'] == base.sha(source_train)
        assert raw_manifest['inputs']['hpo'] == base.sha(args.hpo)
        fit = base.read_jsonl(source_train)
        fit_ids = {str(doc['pmc_id']) for doc in fit}
        seen_hpo, seen_surfaces = set(), set()
        for doc in fit:
            for entity in doc['entities']:
                if str(entity.get('note') or '').upper() != 'NO':
                    seen_hpo.update(alt.get(x, x) for x in entity['identifier'].split(';') if alt.get(x, x) in allowed)
                    seen_surfaces.add(base.norm(entity['text']))
        predictions = base.read_jsonl(prediction_path)
        fold_ids = {str(doc['pmc_id']) for doc in predictions}
        assert not fold_ids & seen and not fold_ids & fit_ids
        assert fold_ids | fit_ids == set(docs)
        seen |= fold_ids
        with gzip.open(raw_path, 'rt') as stream:
            raw_docs = {str(row['pmc_id']): row['candidates'] for row in map(json.loads, stream)}
        assert set(raw_docs) == fold_ids
        surface_ids, surface_high_ids = defaultdict(set), defaultdict(set)
        for candidates in raw_docs.values():
            for candidate in candidates:
                surface = base.norm(candidate['text'])
                ids = set(candidate['identifier'].split(';'))
                surface_ids[surface].update(ids)
                if candidate['probability'] >= .65 and candidate['similarity'] >= .92:
                    surface_high_ids[surface].update(ids)
        local = defaultdict(Counter)
        for pred in predictions:
            doc = docs[str(pred['pmc_id'])]
            truth = old.evaluate.positive_units(doc, doc, 'all')
            guess = old.evaluate.positive_units(pred, doc, 'all')
            entity_text = {(int(e['offset']), int(e['length'])): e['text'] for e in doc['entities']
                           if str(e.get('note') or '').upper() != 'NO'}
            by_span = defaultdict(list)
            for candidate in raw_docs[str(doc['pmc_id'])]:
                by_span[(candidate['offset'], candidate['length'])].append(candidate)
            for span, hid in old.missing_units(truth, guess):
                category = ('exact_span_wrong_identifier' if span in guess else
                            'boundary_or_nested_overlap' if any(old.overlaps(span, s) for s in guess)
                            else 'no_overlapping_prediction')
                counts = local[category]
                counts['units'] += 1
                text = base.norm(entity_text[span])
                canonical = alt.get(hid, hid)
                hpo = canonical in allowed
                counts['raw_hpo_syntax_units'] += bool(re.fullmatch(r'HP:\d{7}', hid))
                counts['alt_identifier_canonicalized'] += canonical != hid
                counts['allowed_hpo_units' if hpo else 'nil_or_nonallowed_units'] += 1
                counts['surface_seen_in_train32'] += text in seen_surfaces
                counts['hpo_seen_in_train32'] += hpo and canonical in seen_hpo
                counts['hpo_name_correct'] += canonical in names.get(text, set())
                counts['hpo_exact_synonym_correct'] += canonical in exact.get(text, set())
                known = canonical in combined.get(text, set())
                counts['hpo_name_or_exact_correct'] += known
                counts['hpo_name_or_exact_any_id'] += text in combined
                counts['hpo_name_or_exact_correct_and_unseen_id'] += known and canonical not in seen_hpo
                counts['hpo_name_or_exact_correct_unambiguous'] += known and len(combined[text]) == 1
                cs = by_span.get(span, [])
                correct = [c for c in cs if hid in c['identifier'].split(';')]
                high = [c for c in correct if c['probability'] >= .65 and c['similarity'] >= .92]
                counts['raw_sap_exact_span_present'] += bool(cs)
                counts['raw_sap_exact_span_correct_id'] += bool(correct)
                counts['raw_sap_exact_span_correct_id_high_confidence'] += bool(high)
                counts['raw_sap_anywhere_normalized_surface_correct_id'] += hid in surface_ids.get(text, set())
                counts['raw_sap_anywhere_normalized_surface_correct_id_high_confidence'] += hid in surface_high_ids.get(text, set())
                counts['hpo_exact_or_actual_sap_high_correct'] += known or bool(high)
                counts['hpo_exact_and_actual_sap_high_correct'] += known and bool(high)
                counts['hpo_exact_correct_but_raw_span_missing'] += known and not cs
                counts['hpo_exact_correct_but_raw_identifier_wrong'] += known and bool(cs) and not correct
                counts['sap_high_correct_excluded_by_exact_gate'] += any(c['exact'] for c in high)
                pattern = ('raw_span_missing' if not cs else 'raw_span_wrong_id' if not correct
                           else '|'.join(sorted({flags(doc, c) for c in correct})))
                counts['pipeline_' + pattern] += 1
                if category == 'no_overlapping_prediction' and pattern == 'passes_pre_overlap_gates':
                    # These would have been added by the frozen augmentation.
                    raise AssertionError('Unexpected gate-surviving no-overlap correct candidate')
        for category, count in local.items():
            totals[category].update(count)
        folds.append({'inner_fold': item['inner_fold'], 'train_documents': len(fit_ids),
                      'train_seen_allowed_hpo': len(seen_hpo),
                      'counts': {key: dict(value) for key, value in local.items()}})
        sources.append({'inner_fold': item['inner_fold'], 'train': source_train,
                        'train_sha256': base.sha(source_train), 'predictions': prediction_path,
                        'predictions_sha256': base.sha(prediction_path), 'raw_sap': raw_path,
                        'raw_sap_sha256': base.sha(raw_path), 'raw_manifest': raw_manifest_path,
                        'raw_manifest_sha256': base.sha(raw_manifest_path)})
    assert seen == set(docs)
    assert {key: value['units'] for key, value in totals.items()} == {
        'no_overlapping_prediction': 908, 'boundary_or_nested_overlap': 411,
        'exact_span_wrong_identifier': 45}
    prepared = json.loads(Path(args.prepared_manifest).read_text())
    assert prepared['files']['train'] == base.sha(train_path)
    result = {'scope': 'Aggregate training48 nested OOF diagnostic only; no new predictions/GPU/dev or outerfold0 gold',
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'unit': 'v2 missing deduplicated (exact span, atomic identifier), including one-to-one NIL matching',
              'configuration': vars(args), 'source_sha256': base.sha(__file__),
              'dependencies': {name: base.sha(path) for name, path in {
                  'sequence_base': base.__file__, 'old_diagnostic': old.__file__,
                  'public_evaluator': old.evaluate.__file__, 'negation_rules': ontology.__file__}.items()},
              'input_hashes': {'sft_manifest': base.sha(args.sft_manifest), 'hpo': base.sha(args.hpo),
                               'prepared_manifest': base.sha(args.prepared_manifest), 'train48': base.sha(train_path)},
              'hpo': {'version': version, 'allowed_concepts_without_root': len(allowed),
                      'name_surfaces': len(names), 'exact_synonym_surfaces': len(exact),
                      'combined_surfaces': len(combined)},
              'counts': {key: dict(value) for key, value in totals.items()}, 'folds': folds, 'sources': sources,
              'current_gliner_prepared': {'totals': prepared['totals'],
                  'training48_rows': prepared['artifacts']['train48']['rows'],
                  'nested32_rows': prepared['artifacts']['nested_train32']['rows'],
                  'label': 'single phenotype span type, not 19119-way HPO classification'},
              'limitations': [
                  'Gold-aware aggregate planning only. No per-example surfaces, IDs or corrections exported.',
                  'Exact ontology coverage uses existing lowercased CRF tokens/whitespace, retaining punctuation tokens; it is not literal-string identity.',
                  'Correct Sap high confidence means actual same-fold cached exact-span candidate, before exact/negation/overlap gates.',
                  'Anywhere normalized-surface cache coverage is only a diagnostic proxy; those scores were not recomputed on missing spans.',
                  'The old raw Sap pool is bounded and filtered during generation; absence is not proof that the encoder cannot link a supplied phrase.',
                  'Current supervised GLiNER has not been trained separately on all three nested folds; no claim it still misses all 908 spans.',
                  'High confidence or ontology matching is not evidence of correct patient attribution.'
              ]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'hpo': result['hpo'], 'counts': result['counts']}))


if __name__ == '__main__':
    main()
