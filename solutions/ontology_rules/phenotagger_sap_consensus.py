#!/usr/bin/env python3
"""One fixed Sap agreement control on already accepted Pheno additions.

The reconstructed detector and original bank are read only. Bank label order is
rebuilt from the original HPO/train48 algorithm without fitting any classifier.
Only the score command initializes a GPU encoder; no gold is used here.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def index(rows):
    result = {str(r['pmc_id']): r for r in rows}
    assert len(result) == len(rows), 'Duplicate documents'
    return result


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def key(entity):
    return entity['offset'], entity['length'], entity['identifier']


def load_semantic(path):
    spec = importlib.util.spec_from_file_location('frozen_semantic', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(args):
    rebuilt = json.loads(Path(args.reconstruction_manifest).read_text())
    assert rebuilt['passed'] and sha(args.semantic_source) == rebuilt['semantic_source_sha256']
    for name in ['train', 'input', 'hpo']:
        assert sha(getattr(args, name)) == rebuilt['inputs'][name], name
    sem = load_semantic(args.semantic_source)
    train = read(args.train); docs = index(read(args.input))
    bases = index(read(args.base)); parents = index(read(args.parent))
    assert {str(d['pmc_id']) for d in train}.isdisjoint(docs)
    assert set(docs) == set(bases) == set(parents)
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    terms, canonical = sem.hpo_terms(args.hpo)
    # Same prefix of System.__init__, with NO detector/encoder construction.
    state = SimpleNamespace(canonical=canonical, surface=defaultdict(Counter), concept_count=Counter())
    for identifier, term in terms.items():
        if identifier == 'HP:0000118': continue
        for label in [term['name']] + term['synonyms']:
            if len(label) >= 3: state.surface[sem.norm(label)][identifier] += 0.05
    for doc in train:
        for entity in doc.get('entities', []):
            if str(entity.get('note', '')).upper() == 'NO': continue
            identifier = sem.System.canonicalize(state, entity.get('identifier', ''))
            text = str(entity.get('text', '')).strip()
            if not identifier or len(text) < 2: continue
            state.surface[sem.norm(text)][identifier] += 1
            state.concept_count[identifier] += 1
    labels = sorted(state.surface)
    label_ids = [sem.System.best_id(state, text) for text in labels]
    text_digest = hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest()
    cache_key = hashlib.sha256((rebuilt['encoder_signature'] + '\0' + '\0'.join(labels)).encode()).hexdigest()
    assert text_digest == rebuilt['bank_cache']['ordered_texts_sha256']
    assert cache_key == Path(rebuilt['bank_cache']['path']).stem
    assert len(labels) == rebuilt['training']['semantic_bank']
    requests = []; counts = Counter()
    for pmc, doc in docs.items():
        base_entities = {key(e): e for e in bases[pmc]['entities']}
        parent_entities = {key(e): e for e in parents[pmc]['entities']}
        assert len(base_entities) == len(bases[pmc]['entities'])
        assert len(parent_entities) == len(parents[pmc]['entities'])
        assert all(parent_entities.get(k) == e for k, e in base_entities.items())
        candidates = []
        for entity in parents[pmc]['entities']:
            if key(entity) in base_entities: continue
            offset, length, _ = key(entity)
            assert length == len(entity['text']) > 0
            assert any(p['offset'] <= offset and offset + length <= p['offset'] + len(p['text'])
                       and p['text'][offset-p['offset']:offset-p['offset']+length] == entity['text']
                       for p in doc['full_text'])
            assert not any(offset < e['offset'] + e['length'] and e['offset'] < offset + length
                           for e in bases[pmc]['entities'])
            candidates.append(copy.deepcopy(entity)); counts['additions'] += 1
        requests.append({'pmc_id': doc['pmc_id'], 'candidates': candidates})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in requests))
    save(out/'bank-labels.json', {'labels': labels, 'label_ids': label_ids})
    manifest = {'source_sha256': sha(__file__), 'semantic_source_sha256': sha(args.semantic_source),
        'files': {k: {'path': str(Path(getattr(args, k)).resolve()), 'sha256': sha(getattr(args, k))}
                  for k in ['train', 'input', 'hpo', 'base', 'parent', 'reconstruction_manifest']},
        'request_sha256': sha(out/'requests.jsonl'), 'bank_labels_sha256': sha(out/'bank-labels.json'),
        'ordered_texts_sha256': text_digest, 'bank_cache_key': cache_key,
        'counts': dict(counts, documents=len(requests), bank_rows=len(labels)),
        'policy': {'sap_p_gte': .65, 'sap_cos_gte': .92, 'pure_nearest_id_equals_pheno_id': True,
                   'preserve_base': True, 'reassign_only_retained_additions': True},
        'gold_used': False, 'classifier_fit': False}
    save(out/'request.manifest.json', manifest)
    print(json.dumps({'counts': manifest['counts'], 'request_sha256': manifest['request_sha256']}))


def score(args):
    import time
    import numpy as np
    from scipy.special import expit
    started = time.monotonic(); reqdir = Path(args.request_dir)
    request = json.loads((reqdir/'request.manifest.json').read_text())
    assert sha(__file__) == request['source_sha256']
    rebuilt_path = Path(args.reconstruction_dir)/'reconstruction.manifest.json'
    rebuilt = json.loads(rebuilt_path.read_text())
    assert rebuilt['passed'] and sha(rebuilt_path) == request['files']['reconstruction_manifest']['sha256']
    assert sha(args.semantic_source) == request['semantic_source_sha256'] == rebuilt['semantic_source_sha256']
    for name in ['train', 'input', 'hpo']: assert rebuilt['inputs'][name] == request['files'][name]['sha256']
    assert sha(reqdir/'requests.jsonl') == request['request_sha256']
    assert sha(reqdir/'bank-labels.json') == request['bank_labels_sha256']
    mapping = json.loads((reqdir/'bank-labels.json').read_text()); label_ids = mapping['label_ids']
    assert hashlib.sha256(json.dumps(mapping['labels'], ensure_ascii=False).encode()).hexdigest() == rebuilt['bank_cache']['ordered_texts_sha256']
    detector_path = Path(args.reconstruction_dir)/'reconstructed-detector.npz'
    assert sha(detector_path) == rebuilt['detector']['sha256']
    assert sha(args.bank) == rebuilt['bank_cache']['sha256']
    for filename, digest in rebuilt['actual_model_files_sha256'].items():
        assert sha(Path(args.model_dir)/filename) == digest, filename
    sem = load_semantic(args.semantic_source)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    encoder = sem.Encoder(args.model_dir, 'sapbert', 'cuda', 256, out/'embeddings', 1)
    assert encoder.signature == rebuilt['encoder_signature']
    detector = np.load(detector_path, allow_pickle=False)
    assert detector['classes'].tolist() == [0, 1] and detector['coef'].shape == (1, 768)
    bank = np.load(args.bank, allow_pickle=False)
    assert bank.shape == (len(label_ids), 768)
    requests = index(read(reqdir/'requests.jsonl')); assert set(requests) == set(rebuilt['input_ids'])
    rows = []; counts = Counter()
    for pmc, row in requests.items():
        candidates = row['candidates']; texts = [sem.norm(c['text']) for c in candidates]
        names = sorted(set(texts)); lookup = {text: i for i, text in enumerate(names)}
        values = encoder.encode(names, 'pheno-additions-'+pmc)
        vectors = values[np.asarray([lookup[t] for t in texts], dtype=np.int64)]
        probabilities = expit(vectors @ detector['coef'].T + detector['intercept']).ravel()
        nearest, similarities = encoder.nearest(vectors, bank)
        scored = []
        for candidate, p, idx, cosine in zip(candidates, probabilities, nearest, similarities):
            scored.append(dict(candidate, probability=float(p), similarity=float(cosine),
                               nearest_index=int(idx), sap_identifier=label_ids[int(idx)]))
            counts['occurrences'] += 1
            counts['passing_fixed_filter'] += int(p >= .65 and cosine >= .92 and label_ids[int(idx)] == candidate['identifier'])
        rows.append({'pmc_id': row['pmc_id'], 'candidates': scored})
    (out/'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    manifest = {'source_sha256': sha(__file__), 'semantic_source_sha256': sha(args.semantic_source),
        'request_manifest_sha256': sha(reqdir/'request.manifest.json'), 'request_sha256': request['request_sha256'],
        'bank_labels_sha256': request['bank_labels_sha256'], 'bank_sha256': sha(args.bank),
        'reconstruction_manifest_sha256': sha(rebuilt_path), 'detector_sha256': sha(detector_path),
        'model_files_sha256': rebuilt['actual_model_files_sha256'], 'output_sha256': sha(out/'scores.jsonl'),
        'counts': dict(counts, documents=len(rows)), 'gold_used': False, 'classifier_fit': False,
        'configuration': {'precision': 'FP16 backbone / FP32 CLS and bank dot product', 'batch_size': 256,
                          'max_length': 32, 'threads': 1, 'unique_policy': 'sorted norm(surface) per document',
                          'id_policy': 'pure nearest bank label, no exact-surface override'},
        'elapsed_seconds': time.monotonic()-started}
    save(out/'scores.manifest.json', manifest); print(json.dumps(manifest), flush=True)


def merge(args):
    import merge_semantic
    request = json.loads((Path(args.request_dir)/'request.manifest.json').read_text())
    score_manifest = json.loads(Path(args.scores).with_name('scores.manifest.json').read_text())
    assert score_manifest['request_manifest_sha256'] == sha(Path(args.request_dir)/'request.manifest.json')
    assert score_manifest['output_sha256'] == sha(args.scores)
    assert sha(args.base) == request['files']['base']['sha256']
    assert sha(args.input) == request['files']['input']['sha256']
    docs = index(read(args.input)); bases = index(read(args.base)); scores = index(read(args.scores))
    requested = index(read(Path(args.request_dir)/'requests.jsonl'))
    assert set(docs) == set(bases) == set(scores) == set(requested)
    output = []; counts = Counter()
    for pmc, doc in docs.items():
        pred = copy.deepcopy(bases[pmc]); additions = []
        cs = scores[pmc]['candidates']; rs = requested[pmc]['candidates']
        assert len(cs) == len(rs)
        for candidate, original in zip(cs, rs):
            assert all(candidate[k] == v for k, v in original.items())
            p, cosine = candidate['probability'], candidate['similarity']
            assert math.isfinite(p) and 0 <= p <= 1 and math.isfinite(cosine) and -1.00001 <= cosine <= 1.00001
            counts['original_additions'] += 1
            if p < .65: counts['reject_probability'] += 1; continue
            if cosine < .92: counts['reject_similarity'] += 1; continue
            if candidate['sap_identifier'] != candidate['identifier']: counts['reject_id_disagreement'] += 1; continue
            additions.append(copy.deepcopy(original)); counts['retained_additions'] += 1
        pred['entities'].extend(additions)
        pred['entities'].sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
        links = {row['patient_id']: set(row['phenotype']) for row in merge_semantic.assign(doc, additions)}
        for row in pred['association']:
            old = set(row['phenotype']); row['phenotype'] = sorted(old | links[row['patient_id']])
            counts['added_relations'] += len(set(row['phenotype'])-old)
        output.append(pred)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output))
    manifest = {'source_sha256': sha(__file__), 'association_source_sha256': sha(merge_semantic.__file__),
        'request_manifest_sha256': sha(Path(args.request_dir)/'request.manifest.json'),
        'score_manifest_sha256': sha(Path(args.scores).with_name('scores.manifest.json')),
        'counts': dict(counts), 'output_sha256': sha(target), 'gold_used': False}
    save(target.with_suffix('.manifest.json'), manifest); print(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ['train', 'input', 'hpo', 'base', 'parent', 'semantic-source', 'reconstruction-manifest', 'output-dir']:
        p.add_argument('--'+name, required=True)
    p = sub.add_parser('score')
    for name in ['request-dir', 'reconstruction-dir', 'semantic-source', 'model-dir', 'bank', 'output-dir']:
        p.add_argument('--'+name, required=True)
    p = sub.add_parser('merge')
    for name in ['request-dir', 'scores', 'base', 'input', 'output']: p.add_argument('--'+name, required=True)
    args = parser.parse_args(); globals()[args.command](args)


if __name__ == '__main__': main()
