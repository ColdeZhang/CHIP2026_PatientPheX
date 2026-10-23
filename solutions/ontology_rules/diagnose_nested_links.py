#!/usr/bin/env python3
"""Aggregate training-only linkage availability; no repaired predictions.

Reuse the established selected CRF+Sap OOF loss diagnosis, then check only its
exact-span missing HPO atoms against the same-fold full candidate/dictionary
resources. This neither computes new top-k embeddings nor estimates model Score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import calibrate_multi as multi


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ('manifest', 'existing-diagnostic', 'selected-root', 'semantic-source', 'hpo', 'output'):
        ap.add_argument('--' + n, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    import evaluate
    old = json.loads(Path(a.existing_diagnostic).read_text())
    spec = importlib.util.spec_from_file_location('frozen_semantic_link_diagnostic', a.semantic_source)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    assert sha(a.semantic_source) == 'b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a'
    terms, canonical = sem.hpo_terms(a.hpo)
    ontology = defaultdict(set)
    for hid, term in terms.items():
        if hid != 'HP:0000118':
            for surface in [term['name']] + term['synonyms']:
                if len(surface) >= 3:
                    ontology[sem.norm(surface)].add(hid)
    specs = json.loads(Path(a.manifest).read_text())
    total, folds, seen = Counter(), [], set()
    for i, item in enumerate(specs, 1):
        train = multi.keyed(item['generator_train'])
        docs = multi.keyed(item['gold'])
        assert len(train) == 32 and len(docs) == 16 and not train.keys() & docs.keys()
        assert not seen & docs.keys()
        seen.update(docs)
        selected_path = Path(a.selected_root) / f'inner{i}/candidates.jsonl'
        selected_manifest = json.loads(selected_path.with_suffix('.manifest.json').read_text())
        assert selected_manifest['inputs']['train'] == sha(item['generator_train'])
        assert selected_manifest['inputs']['input'] == sha(item['input'])
        assert selected_manifest['output_sha256'] == sha(selected_path)
        assert selected_manifest['inputs']['candidates'] == sha(item['candidates'])
        selected = multi.keyed(selected_path)
        assert set(selected) == set(docs)
        aliases = defaultdict(set)
        for doc in train.values():
            for e in doc['entities']:
                if str(e.get('note') or '').upper() == 'NO' or len(e['text'].strip()) < 2:
                    continue
                ids = {canonical[h.strip()] for h in e['identifier'].split(';') if h.strip() in canonical}
                if e['identifier'] == '-1':
                    ids.add('-1')
                aliases[sem.norm(e['text'])].update(ids)
        base = multi.keyed(item['base'])
        raw = multi.single.read_candidates(item['candidates'])
        crf = multi.checked_crf(item['crf'], item['crf_manifest'], item['generator_train'], item['input'], set(train))
        c = Counter()
        for did, doc in docs.items():
            truth = evaluate.positive_units(doc, doc, 'all')
            guess = evaluate.positive_units(selected[did], doc, 'all')
            tp, fp, fn = evaluate.mention_counts(truth, guess)
            c.update(entity_tp=tp, entity_fp=fp, entity_fn=fn)
            full = defaultdict(set)
            separate = {name: defaultdict(set) for name in ('semantic', 'dictionary_output', 'CRF')}
            for name, cs in [('semantic', raw[did]), ('dictionary_output', base[did]['entities']), ('CRF', crf[did]['entities'])]:
                for candidate in cs:
                    span = candidate['offset'], candidate['length']
                    hids = set(candidate['identifier'].split(';'))
                    separate[name][span].update(hids)
                    full[span].update(hids)
            for span in set(truth) | set(guess):
                expected, predicted = set(truth.get(span, set())), set(guess.get(span, set()))
                matches = expected & predicted
                expected -= matches
                predicted -= matches
                if '-1' in expected and predicted:
                    expected.remove('-1')
                    predicted.remove(sorted(predicted)[0])
                c['FP_exact_positive_span_wrong_identifier' if span in truth else 'FP_other_span'] += len(predicted)
                if span not in guess:
                    continue
                for hid in expected:
                    c['FN_exact_span_wrong_identifier'] += 1
                    source = next(p for p in doc['full_text'] if p['offset'] <= span[0] and span[0] + span[1] <= p['offset'] + len(p['text']))
                    surface = source['text'][span[0] - source['offset']:span[0] - source['offset'] + span[1]]
                    norm = sem.norm(surface)
                    available = hid in full[span]
                    lexical = hid in ontology.get(norm, set()) or hid in aliases.get(norm, set())
                    c['wrong_ID_target_atomic_allowed_HPO'] += hid in terms
                    c['wrong_ID_target_alt_ID_in_canonical_map'] += hid not in terms and hid in canonical
                    c['wrong_ID_target_outside_canonical_map'] += hid not in canonical
                    c['wrong_ID_target_NIL'] += hid == '-1'
                    c['wrong_ID_target_non_HPO_string'] += not hid.startswith('HP:') and hid != '-1'
                    resolved = canonical.get(hid, hid)
                    c['raw_alt_target_canonical_already_selected'] += resolved != hid and resolved in guess[span]
                    c['raw_alt_target_canonical_in_full_same_span_pool'] += resolved != hid and resolved in full[span]
                    c['raw_alt_target_canonical_in_exact_dictionary'] += resolved != hid and (resolved in ontology.get(norm, set()) or resolved in aliases.get(norm, set()))
                    c['wrong_ID_correct_atomic_in_full_same_span_pool'] += available
                    c['wrong_ID_correct_atomic_absent_full_same_span_pool'] += not available
                    c['wrong_ID_correct_atomic_in_ontology_exact_surface'] += hid in ontology.get(norm, set())
                    c['wrong_ID_correct_atomic_in_train32_exact_surface'] += hid in aliases.get(norm, set())
                    c['wrong_ID_correct_atomic_in_union_exact_surface'] += lexical
                    c['wrong_ID_absent_pool_but_present_exact_dictionary'] += lexical and not available
                    for name, pool in separate.items():
                        c['wrong_ID_correct_atomic_in_' + name] += hid in pool[span]
        total.update(c)
        folds.append({'inner_fold': i, 'counts': dict(c), 'source': {
            k: {'path': item[k], 'sha256': sha(item[k])} for k in ('generator_train', 'gold', 'input', 'base', 'candidates', 'crf')},
            'selected': {'path': str(selected_path), 'sha256': sha(selected_path)}})
    assert len(seen) == 48
    assert all(total[k] == old['counts'][k] for k in ('entity_tp', 'entity_fp', 'entity_fn'))
    assert total['FN_exact_span_wrong_identifier'] == old['counts']['entity_fn_exact_span_wrong_identifier'] == 45
    assert total['FP_exact_positive_span_wrong_identifier'] + total['FP_other_span'] == total['entity_fp']
    result = {'scope': 'Only train48, three nested train32->16 selected OOF pools; aggregate planning, not prediction or Score',
        'source_sha256': sha(__file__), 'semantic_source_sha256': sha(a.semantic_source), 'hpo_sha256': sha(a.hpo),
        'existing_diagnostic': {'path': a.existing_diagnostic, 'sha256': sha(a.existing_diagnostic)},
        'reused_prior_counts': {k: old['counts'][k] for k in ('entity_fn_no_overlapping_prediction', 'entity_fn_boundary_or_nested_overlap')},
        'counts': dict(total), 'folds': folds, 'originaldev_read': False, 'outerfold0_gold_read': False,
        'top_k_status': 'No frozen top-k cache for these three selected nested pools located. Existing sequence Top8 diagnostic concerns outerfold0, and Pheno top3 is a different model; neither is substituted. No embeddings/ranking generated.',
        'limitations': ['Missing correct gold atoms and wrong predicted atoms use different denominators; no oracle predictions are created.',
            'Raw ID mismatch follows frozen evaluator v2. Four gold target atoms use HPO alt IDs; canonical-equivalence availability is reported separately, not used to change evaluation labels or predictions.',
            'Full same-span pool is current ontology/Sap/CRF union, not a semantic top-k list.',
            'Exact-surface dictionary uses frozen Sap name+EXACT terms and corresponding train32 aliases, with lower/whitespace normalization. Presence is optimistic availability, not correct routing or calibrated Score.',
            'The 45 missing HPO atoms describe already selected strong CRF+Sap spans; new GLiNER spans may have a different linking difficulty.']}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
