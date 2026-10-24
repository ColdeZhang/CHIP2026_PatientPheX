#!/usr/bin/env python3
"""Prepare frozen GLiNER Sap queries or append them with the original policy.

No training data are opened and no estimator is fitted. The linker label map,
corrected relation model and HPO scope are all loaded from frozen artifacts.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from load_predict import (TRAIN_SHA, HPO_SHA, SEQUENCE_SHA, ONTOLOGY_SHA, augment,
                          load_module, load_pipeline, read_jsonl, ref, sha, write_jsonl)

BANK_LABELS_SHA = 'a369ec3de1f32764272748f30d0de4244b743fbd28f527adea767bd5fa0b2cd2'
GLINER48_SHA = '25bb0b35782f0a9c84ce7e3c564c4561c3fbb051cd98fcac139a2ed0b0602f23'


def norm(text):
    return ' '.join(text.lower().split()).strip()


def prepare(args, out, docs, rows):
    assert sha(args.bank_labels) == BANK_LABELS_SHA
    bank = json.loads(Path(args.bank_labels).read_text())
    lookup = dict(zip(bank['labels'], bank['label_ids']))
    assert len(lookup) == 42575
    raw, requests = [], []
    for doc, row in zip(docs, rows):
        assert doc['pmc_id'] == row['pmc_id']
        candidates = row['candidates']
        assert len({(c['offset'], c['length']) for c in candidates}) == len(candidates)
        for c in candidates:
            assert not {'identifier', 'probability', 'similarity', 'ontology_exact'} & set(c)
            assert any(int(p.get('offset', 0)) <= c['offset'] and
                c['offset']+c['length'] <= int(p.get('offset', 0))+len(p.get('text', '')) and
                p.get('text', '')[c['offset']-int(p.get('offset', 0)):c['offset']-int(p.get('offset', 0))+c['length']] == c['text']
                for p in doc.get('full_text', []))
        texts = [norm(c['text']) for c in candidates]
        unique = sorted(set(texts))
        index = {text: i for i, text in enumerate(unique)}
        raw.append({'pmc_id': row['pmc_id'], 'candidates': [dict(c, ontology_exact=t in lookup)
                    for c, t in zip(candidates, texts)]})
        requests.append({'pmc_id': row['pmc_id'], 'texts': unique,
            'candidate_to_unique': [index[t] for t in texts],
            'exact_identifiers': [lookup.get(t) for t in texts]})
    write_jsonl(out/'raw-with-exact.jsonl', raw)
    write_jsonl(out/'query-requests.jsonl', requests)
    return {'documents': len(docs), 'candidates': sum(len(r['candidates']) for r in raw),
            'unique_queries_per_document_total': sum(len(r['texts']) for r in requests),
            'encoder_batches_256': sum(math.ceil(len(r['texts'])/256) for r in requests),
            'nearest_blocks_512': sum(math.ceil(len(r['candidate_to_unique'])/512) for r in requests),
            'maximum_unique_queries_in_document': max(len(r['texts']) for r in requests)}


def assemble(args, out, docs, rows):
    sequence = load_module('frozen_sequence', args.sequence_source, SEQUENCE_SHA)
    ontology = load_module('frozen_ontology', args.ontology_source, ONTOLOGY_SHA)
    _, linker, vectorizer, relation, _, _ = load_pipeline(sequence, args.pipeline, args.crf_model)
    assert sha(args.base) == args.base_sha256
    priors = read_jsonl(args.base)
    assert len(priors) == len(docs)
    allowed = set(linker.allowed) - {'HP:0000118'}
    counts = Counter()
    results = []
    for doc, prior, row in zip(docs, priors, rows):
        assert doc['pmc_id'] == prior['pmc_id'] == row['pmc_id']
        assert augment(sequence, ontology, doc, prior, [], vectorizer, relation)['association'] == prior['association']
        entities = list(prior['entities'])
        for candidate in sorted(row['candidates'], key=lambda e: (-e['score'], -e['length'], e['offset'])):
            counts['all_candidates'] += 1
            if candidate['probability'] < .65 or candidate['similarity'] < .92:
                counts['semantic_filter'] += 1
                continue
            ids = candidate['identifier'].split(';')
            if not ids or not all(identifier in allowed for identifier in ids):
                counts['no_valid_hpo'] += 1
                continue
            _, passage = sequence.passage_for(doc, candidate['offset'])
            local = candidate['offset']-int(passage.get('offset', 0))
            assert passage['text'][local:local+candidate['length']] == candidate['text']
            if ontology.is_negated(passage['text'], local, local+candidate['length']):
                counts['negated'] += 1
                continue
            def overlap(e):
                return candidate['offset'] < e['offset']+e['length'] and e['offset'] < candidate['offset']+candidate['length']
            if any(overlap(e) for e in prior['entities']):
                counts['overlap_original'] += 1
                continue
            if any(overlap(e) for e in entities):
                counts['overlap_added'] += 1
                continue
            entities.append({k: candidate[k] for k in ['identifier', 'offset', 'length', 'text']} |
                            {'type': 'Phenotype', 'note': None})
            counts['accepted'] += 1
        result = augment(sequence, ontology, doc, {'entities': entities}, [], vectorizer, relation)
        results.append(result)
    write_jsonl(out/'full-lr.jsonl', results)
    if args.reference:
        assert results == read_jsonl(args.reference)
        assert sha(out/'full-lr.jsonl') == sha(args.reference)
    return {'documents': len(results), 'patients': sum(len(d['association']) for d in results),
            'entities': sum(len(d['entities']) for d in results),
            'patient_phenotypes': sum(len(a['phenotype']) for d in results for a in d['association']),
            'counts': dict(counts), 'reference_bytes_exact': bool(args.reference)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'assemble'])
    for name in ['input', 'input-sha256', 'candidates', 'candidates-sha256', 'output-dir']:
        ap.add_argument('--'+name, required=True)
    for name in ['bank-labels', 'base', 'base-sha256', 'pipeline', 'crf-model', 'sequence-source', 'ontology-source', 'reference']:
        ap.add_argument('--'+name)
    args = ap.parse_args()
    assert sha(args.input) == args.input_sha256 and sha(args.candidates) == args.candidates_sha256
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    docs, rows = read_jsonl(args.input), read_jsonl(args.candidates)
    assert len(docs) == len(rows) and len({str(d['pmc_id']) for d in docs}) == len(docs)
    assert all('entities' not in d and 'association' not in d for d in docs)
    stats = prepare(args, out, docs, rows) if args.mode == 'prepare' else assemble(args, out, docs, rows)
    manifest = {'config': vars(args), 'source': ref(__file__), 'implementation': ref(Path(__file__).with_name('load_predict.py')),
        'input_sha256': args.input_sha256, 'train64_sha256': TRAIN_SHA, 'hpo_sha256': HPO_SHA,
        'gliner48_checkpoint_sha256': GLINER48_SHA, 'stats': stats,
        'inputs': {k: ref(getattr(args, k)) for k in ['input', 'candidates', 'bank_labels', 'base', 'pipeline', 'crf_model', 'sequence_source', 'ontology_source', 'reference'] if getattr(args, k)},
        'outputs': {p.name: ref(p) for p in out.glob('*.jsonl')},
        'policy': {'probability': .65, 'similarity': .92, 'relation_threshold': .55,
                   'order': 'GLiNER score descending,length descending,offset ascending',
                   'relation': 'existing corrected train64 LR all-pairs; no fit'},
        'fit_called': False, 'train_or_gold_read': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'stats': stats, 'outputs': manifest['outputs']}), flush=True)


if __name__ == '__main__':
    main()
