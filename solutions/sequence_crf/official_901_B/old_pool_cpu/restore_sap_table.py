#!/usr/bin/env python3
"""Restore frozen Sap64 lexical counts, then emit unencoded B query requests.

Only the original count algorithm is restored, under explicit parent/root
authorization. System.__init__, Encoder and fit_detector are never executed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from load_predict import HPO_SHA, TRAIN_SHA, forbidden, load_module, read_jsonl, ref, sha, write_jsonl

SEMANTIC_SHA = "b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a"
BANK_LABELS_SHA = "a369ec3de1f32764272748f30d0de4244b743fbd28f527adea767bd5fa0b2cd2"
DETECTOR_SHA = "adb8ff8c8f395d74e2d76dc02995b5c51bf6e0fde818de7a7f4e8667e8b6b626"


def restore(sem, train, terms, canonical):
    system = sem.System.__new__(sem.System)
    system.args = argparse.Namespace(seed=20261023, max_candidates=6000, negatives_per_doc=500)
    system.canonical = canonical
    system.surface = defaultdict(Counter)
    system.train_surface = defaultdict(Counter)
    system.concept_count = Counter()
    # Exact original System.__init__ lexical/count block, before encoder/fit.
    for identifier, term in terms.items():
        if identifier == 'HP:0000118':
            continue
        for label in [term['name']] + term['synonyms']:
            if len(label) >= 3:
                system.surface[sem.norm(label)][identifier] += 0.05
    system.positive_heads = Counter()
    for doc in train:
        for e in doc.get('entities', []):
            if str(e.get('note', '')).upper() == 'NO':
                continue
            identifier = system.canonicalize(e.get('identifier', ''))
            text = str(e.get('text', '')).strip()
            if not identifier or len(text) < 2:
                continue
            system.surface[sem.norm(text)][identifier] += 1
            system.train_surface[sem.norm(text)][identifier] += 1
            system.concept_count[identifier] += 1
            words = sem.WORD.findall(text.lower())
            if words:
                system.positive_heads[words[-1]] += 1
    system.labels = sorted(system.surface)
    system.label_ids = [system.best_id(k) for k in system.labels]
    system.automaton = sem.ahocorasick.Automaton()
    for text in system.labels:
        system.automaton.add_word(text, text)
    system.automaton.make_automaton()
    system.word_count = Counter(w for text in system.labels for w in set(sem.WORD.findall(text)))
    system.heads = {sem.WORD.findall(x)[-1] for x in system.labels if sem.WORD.findall(x)} - sem.STOP
    system.heads |= set(system.positive_heads) - sem.STOP
    return system


def main():
    ap = argparse.ArgumentParser()
    for name in ['train', 'hpo', 'semantic-source', 'bank-labels', 'detector', 'dev-input',
                 'dev-candidates', 'b-input', 'output-dir']:
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    assert sha(args.train) == TRAIN_SHA and sha(args.hpo) == HPO_SHA
    assert sha(args.bank_labels) == BANK_LABELS_SHA and sha(args.detector) == DETECTOR_SHA
    assert sha(args.dev_input) == 'c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e'
    assert sha(args.dev_candidates) == '8e96fe840fa9520d7792b522583e263a35dcf89e114c021143362f3820470197'
    assert sha(args.b_input) == '878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists()
    sem = load_module('frozen_sap_counts', args.semantic_source, SEMANTIC_SHA)
    sem.System.__init__ = sem.System.fit_detector = sem.Encoder.__init__ = forbidden
    train, dev, bdocs = [read_jsonl(p) for p in [args.train, args.dev_input, args.b_input]]
    assert len(train) == 64 and len(dev) == 16 and len(bdocs) == 100
    assert not {str(d['pmc_id']) for d in train} & {str(d['pmc_id']) for d in dev+bdocs}
    assert all('entities' not in d and 'association' not in d for d in dev+bdocs)
    system = restore(sem, train, *sem.hpo_terms(args.hpo))
    bank_labels = json.loads(Path(args.bank_labels).read_text())
    assert system.labels == bank_labels['labels'] and system.label_ids == bank_labels['label_ids']
    assert len(system.labels) == 42575
    table = {k: getattr(system, k) for k in ['canonical', 'surface', 'train_surface', 'concept_count',
              'positive_heads', 'labels', 'label_ids', 'word_count']}
    table['heads'] = sorted(system.heads)
    table['configuration'] = vars(system.args)
    (out/'sap64-lexical-state.json').write_text(json.dumps(table, sort_keys=True, ensure_ascii=False)+'\n')
    old = read_jsonl(args.dev_candidates)
    assert [str(d['pmc_id']) for d in old] == [str(d['pmc_id']) for d in dev]
    replay = []
    for d, cached in zip(dev, old):
        candidates = system.candidates(d)
        assert len(candidates) == len(cached['candidates'])
        assert all(all(c[k] == r[k] for k in c) for c, r in zip(candidates, cached['candidates']))
        replay.append({'pmc_id': d['pmc_id'], 'candidates': candidates})
    write_jsonl(out/'dev.raw-candidates.jsonl', replay)
    raw, requests = [], []
    for d in bdocs:
        candidates = system.candidates(d)
        texts = [sem.norm(c['text']) for c in candidates]
        unique = sorted(set(texts))
        lookup = {text: i for i, text in enumerate(unique)}
        raw.append({'pmc_id': d['pmc_id'], 'candidates': candidates})
        requests.append({'pmc_id': d['pmc_id'], 'texts': unique,
            'candidate_to_unique': [lookup[text] for text in texts],
            'exact_identifiers': [system.best_id(text) if c['exact'] else None for c, text in zip(candidates, texts)]})
    write_jsonl(out/'B.raw-candidates.jsonl', raw)
    write_jsonl(out/'B.query-requests.jsonl', requests)
    manifest = {'config': vars(args), 'source': ref(__file__), 'semantic_source': ref(args.semantic_source),
        'inputs': {k: ref(getattr(args, k)) for k in ['train', 'hpo', 'bank_labels', 'detector', 'dev_input', 'dev_candidates', 'b_input']},
        'authorization': 'parent/root approved original train64 deterministic Sap lexical/count recovery, without encoder or detector fitting',
        'initial_gap': 'Original positive_heads lexical state was not serialized; restored using unchanged frozen source count block',
        'outputs': {p.name: ref(p) for p in out.iterdir() if p.is_file()},
        'checks': {'bank_labels_and_ids_exact': True, 'all_dev_candidates_exact_original_fields': True},
        'ordered_labels_sha256': hashlib.sha256(json.dumps(system.labels, ensure_ascii=False).encode()).hexdigest(),
        'dev_documents': len(dev), 'dev_candidates': sum(len(r['candidates']) for r in replay),
        'b_documents': len(bdocs), 'b_candidates': sum(len(r['candidates']) for r in raw),
        'b_unique_queries_per_document_total': sum(len(r['texts']) for r in requests),
        'train_documents_read': 64, 'dev_gold_read': False, 'encoder_loaded': False,
        'fit_detector_called': False, 'learned_model_fit': False, 'quality_evaluation': False,
        'gpu_query_contract': {'model_revision': '090663c3ae57bf35ffe4d0d468a2a88d03051a4d',
          'fp16_backbone': True, 'batch': 256, 'max_length': 32, 'pooling': 'CLS float32 then L2 normalize',
          'bank_sha256': '4fc0e062ea7940365222aad637058aadfb37dc57681e444eb2091577b591ae03',
          'detector': 'load existing coef/intercept/classes NPZ; float64 logistic predict_proba',
          'nearest': 'original GPU float32 query @ bank.T in blocks512, first argmax',
          'requests': 'Each row texts is sorted(set(norm(candidate.text))); candidate_to_unique restores original order'},
        'seconds': time.monotonic()-started}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['checks', 'dev_candidates', 'b_candidates', 'b_unique_queries_per_document_total', 'outputs']}), flush=True)


if __name__ == '__main__':
    main()
