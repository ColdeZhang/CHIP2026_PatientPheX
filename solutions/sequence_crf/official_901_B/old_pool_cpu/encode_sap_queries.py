#!/usr/bin/env python3
"""Score prepared queries with frozen Sap64; never construct System or fit.

GPU mode calls the original Encoder on sorted unique queries, expands vectors
back to original candidate order, then calls the original nearest routine.
CPU-cache mode opens SHA-verified old embeddings only and never loads a model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

SEMANTIC_SHA = 'b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a'
BANK_SHA = '4fc0e062ea7940365222aad637058aadfb37dc57681e444eb2091577b591ae03'
BANK_BYTES = 130790528
LABELS_SHA = 'a369ec3de1f32764272748f30d0de4244b743fbd28f527adea767bd5fa0b2cd2'
DETECTOR_SHA = 'adb8ff8c8f395d74e2d76dc02995b5c51bf6e0fde818de7a7f4e8667e8b6b626'
RECONSTRUCTION_SHA = '469e5299cdc9593a53efae907741b4df9bd3ee7e6b15ed2eebfd48062a775a5f'
MODEL_HASHES = {
    'config.json': 'c0fa35def52bf7d81865d23dba0417fdbb67d416fdde70cae5c7beab476b584e',
    'tokenizer_config.json': '76eb84b4b5b0d1bdba02906d7fae249be541c41d72e2024094aea513f98f2be7',
    'special_tokens_map.json': '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3',
    'vocab.txt': '79489a52be45e6fa033521e8ce8e4f62aedc0a742ee2aa6fc04667e5b0b1454d',
    'model.safetensors': 'dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814',
    'manifest.json': '0f4f8a8c16d0d7f978ab544c80d117b930666de2793276797991175ea52a81d8',
}
SIGNATURE = 'sapbert' + MODEL_HASHES['config.json'] + MODEL_HASHES['manifest.json']


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode())


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checked(path, expected):
    actual = ref(path)
    if actual['sha256'] != expected:
        raise ValueError('SHA mismatch: ' + str(path))
    return actual


def forbidden(*args, **kwargs):
    raise RuntimeError('Fit/System initialization forbidden in frozen query scorer')


def load_detector(path):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    checked(path, DETECTOR_SHA)
    with np.load(path, allow_pickle=False) as state:
        assert set(state.files) == {'coef', 'intercept', 'classes'}
        model = LogisticRegression(C=.6, class_weight='balanced', max_iter=450, random_state=20261023)
        model.coef_ = state['coef'].copy()
        model.intercept_ = state['intercept'].copy()
        model.classes_ = state['classes'].copy()
    assert model.coef_.shape == (1, 768) and model.coef_.dtype == np.float64
    assert model.intercept_.shape == (1,) and model.intercept_.dtype == np.float64
    assert np.array_equal(model.classes_, np.array([0, 1], dtype=np.int64))
    model.n_features_in_ = 768
    model.fit = forbidden
    return model


def validate_request(sem, raw, query, doc, exact_lookup):
    assert str(raw['pmc_id']) == str(query['pmc_id']) == str(doc['pmc_id'])
    assert 'entities' not in doc and 'association' not in doc
    assert ('candidates' in raw) != ('entities' in raw)
    candidates = raw.get('candidates', raw.get('entities'))
    texts = query['texts']
    mapping = query['candidate_to_unique']
    exact_ids = query['exact_identifiers']
    assert texts == sorted(set(texts))
    assert all(isinstance(text, str) and text == sem.norm(text) for text in texts)
    assert len(candidates) == len(mapping) == len(exact_ids)
    assert texts == sorted(set(sem.norm(c['text']) for c in candidates))
    for candidate, index, exact_id in zip(candidates, mapping, exact_ids):
        assert type(index) is int and 0 <= index < len(texts)
        assert texts[index] == sem.norm(candidate['text'])
        assert type(candidate['offset']) is int and type(candidate['length']) is int
        assert candidate['length'] > 0
        assert any(int(p.get('offset', 0)) <= candidate['offset'] and
            candidate['offset'] + candidate['length'] <= int(p.get('offset', 0)) + len(p.get('text', '')) and
            p.get('text', '')[candidate['offset'] - int(p.get('offset', 0)):
                candidate['offset'] - int(p.get('offset', 0)) + candidate['length']] == candidate['text']
            for p in doc.get('full_text', []))
        if 'exact' in candidate:
            assert isinstance(candidate['exact'], bool)
            assert candidate['exact'] == (exact_id is not None)
        if exact_id is not None:
            assert exact_lookup[texts[index]] == exact_id
        assert not {'identifier', 'probability', 'similarity'} & set(candidate), 'Expected unscored candidates'
    return candidates


def replay_difference(scored, reference):
    import numpy as np
    assert str(scored['pmc_id']) == str(reference['pmc_id'])
    new, old = scored['candidates'], reference['candidates']
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert {k: v for k, v in a.items() if k not in ['probability', 'similarity']} == {
            k: v for k, v in b.items() if k not in ['probability', 'similarity']}
    p = np.asarray([c['probability'] for c in new])
    op = np.asarray([c['probability'] for c in old])
    s = np.asarray([c['similarity'] for c in new])
    oscore = np.asarray([c['similarity'] for c in old])
    return {'pmc_id': scored['pmc_id'], 'candidates': len(new), 'objects_exact': scored == reference,
        'max_abs_probability': float(np.max(np.abs(p-op), initial=0)),
        'max_abs_similarity': float(np.max(np.abs(s-oscore), initial=0)),
        'probability_decision_mismatches': int(np.count_nonzero((p >= .65) != (op >= .65))),
        'similarity_decision_mismatches': int(np.count_nonzero((s >= .92) != (oscore >= .92)))}


def run(args, out, status):
    import numpy as np
    started = time.monotonic()
    if args.mode == 'cpu-cache':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        assert args.expected_gpu_uuid and os.environ.get('CUDA_VISIBLE_DEVICES') == args.expected_gpu_uuid
    inputs = {name: checked(getattr(args, name), expected) for name, expected in [
        ('semantic_source', SEMANTIC_SHA), ('bank', BANK_SHA), ('bank_labels', LABELS_SHA),
        ('detector', DETECTOR_SHA), ('requests', args.requests_sha256),
        ('raw_candidates', args.raw_candidates_sha256), ('input', args.input_sha256)]}
    assert inputs['bank']['bytes'] == BANK_BYTES
    model_files = {name: checked(Path(args.model_dir)/name, expected) for name, expected in MODEL_HASHES.items()}
    known_caches = {}
    if args.reconstruction_manifest:
        inputs['reconstruction_manifest'] = checked(args.reconstruction_manifest, RECONSTRUCTION_SHA)
        reconstruction = json.loads(Path(args.reconstruction_manifest).read_text())
        assert reconstruction['encoder_signature'] == SIGNATURE
        known_caches = {Path(c['path']).name: c['sha256'] for c in reconstruction['cache_reads']}
    if args.mode == 'cpu-cache':
        assert args.reconstruction_manifest and args.reference_scored
    reference = None
    if args.reference_scored:
        assert args.reference_scored_sha256
        inputs['reference_scored'] = checked(args.reference_scored, args.reference_scored_sha256)
        reference = read_jsonl(args.reference_scored)
    spec = importlib.util.spec_from_file_location('frozen_sap_encoder', args.semantic_source)
    sem = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sem
    spec.loader.exec_module(sem)
    sem.System.__init__ = sem.System.fit_detector = forbidden
    detector = load_detector(args.detector)
    bank = np.load(args.bank, allow_pickle=False)
    assert bank.shape == (42575, 768) and bank.dtype == np.float32 and np.isfinite(bank).all()
    labels = json.loads(Path(args.bank_labels).read_text())
    assert len(labels['labels']) == len(labels['label_ids']) == len(bank)
    exact_lookup = dict(zip(labels['labels'], labels['label_ids']))
    requests, raw, docs = [read_jsonl(p) for p in [args.requests, args.raw_candidates, args.input]]
    assert len(requests) == len(raw) == len(docs)
    assert len({str(q['pmc_id']) for q in requests}) == len(requests)
    if reference is not None:
        assert len(reference) == len(requests)
    # Full request validation precedes any neural model loading.
    for row, query, doc in zip(raw, requests, docs):
        validate_request(sem, row, query, doc, exact_lookup)
    cache_dir = Path(args.cache_dir)
    if args.mode == 'cpu-cache':
        assert cache_dir.is_dir()
        encoder = sem.Encoder.__new__(sem.Encoder)
        encoder.kind, encoder.device = 'sapbert', 'cpu'
        encoder.encode = forbidden
    else:
        encoder = sem.Encoder(args.model_dir, 'sapbert', 'cuda', 256, cache_dir, 1)
        assert encoder.signature == SIGNATURE and encoder.batch == 256
        assert encoder.torch.cuda.device_count() == 1
    gpu = None if args.mode == 'cpu-cache' else {
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'name': encoder.torch.cuda.get_device_name(0),
        'properties': str(encoder.torch.cuda.get_device_properties(0)),
        'tf32_matmul': encoder.torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': encoder.torch.get_float32_matmul_precision()}
    (out/'documents').mkdir()
    counts, comparisons, output_rows = [], [], []
    for index, (row, query, doc) in enumerate(zip(raw, requests, docs)):
        status.update(document_index=index, pmc_id=query['pmc_id'], stage='encoding_or_cache_load')
        before = time.monotonic()
        texts = query['texts']
        digest = hashlib.sha256((SIGNATURE+'\0'+'\0'.join(texts)).encode()).hexdigest()
        cache = cache_dir/(digest+'.npy')
        receipt = cache_dir/(digest+'.frozen-query.json')
        existed = cache.exists()
        expected_cache_sha = known_caches.get(cache.name)
        if existed and expected_cache_sha is None:
            if not receipt.exists():
                raise ValueError('Existing query cache has no verified provenance: '+str(cache))
            record = json.loads(receipt.read_text())
            assert record['encoder_signature'] == SIGNATURE and record['cache_key'] == digest
            expected_cache_sha = record['embedding']['sha256']
        if existed:
            checked(cache, expected_cache_sha)
        if args.mode == 'cpu-cache':
            assert existed, 'CPU replay forbids cache misses'
            unique_vectors = np.load(cache, allow_pickle=False)
        else:
            unique_vectors = encoder.encode(texts, 'candidates-'+str(query['pmc_id'])+'-unique')
        assert unique_vectors.shape == (len(texts), 768) and unique_vectors.dtype == np.float32
        assert np.isfinite(unique_vectors).all()
        cache_ref = ref(cache)
        if args.mode == 'gpu' and not existed:
            save_json(receipt, {'encoder_signature': SIGNATURE, 'cache_key': digest,
                'embedding': cache_ref, 'requests': inputs['requests'], 'document_index': index,
                'pmc_id': query['pmc_id'], 'source': ref(__file__), 'origin': args.origin})
        # Match UniqueEncoder: expand before LR and before nearest block512.
        vectors = unique_vectors[np.asarray(query['candidate_to_unique'], dtype=np.int64)]
        status['stage'] = 'frozen_scoring'
        probs = detector.predict_proba(vectors)[:, 1] if len(vectors) else np.empty(0, dtype=np.float64)
        nearest, similarities = encoder.nearest(vectors, bank)
        candidates = row.get('candidates', row.get('entities'))
        scored = {'pmc_id': row['pmc_id'], 'candidates': [dict(candidate,
            identifier=exact_id if exact_id is not None else labels['label_ids'][int(nearest_id)],
            probability=float(probability), similarity=float(similarity))
            for candidate, exact_id, nearest_id, probability, similarity in zip(
                candidates, query['exact_identifiers'], nearest, probs, similarities)]}
        assert len(scored['candidates']) == len(candidates)
        assert np.isfinite(probs).all() and np.isfinite(similarities).all()
        # Save full raw scores before comparisons or other acceptance checks.
        target = out/'documents'/f'{index:04d}.json'
        save_json(target, scored)
        entry = {'document_index': index, 'pmc_id': row['pmc_id'], 'origin': args.origin,
            'candidates': len(candidates), 'unique_queries': len(texts),
            'embedding_identity': 'verified-cache' if existed else 'fresh-encoding',
            'embedding': cache_ref, 'scored': ref(target), 'seconds': time.monotonic()-before}
        save_json(out/'documents'/f'{index:04d}.manifest.json', entry)
        counts.append(entry)
        output_rows.append(scored)
        if reference is not None:
            difference = replay_difference(scored, reference[index])
            comparisons.append(difference)
            save_json(out/'documents'/f'{index:04d}.comparison.json', difference)
            assert difference['max_abs_probability'] <= 1e-8
            assert difference['max_abs_similarity'] <= 2e-6
            assert difference['probability_decision_mismatches'] == difference['similarity_decision_mismatches'] == 0
        print(json.dumps({'event': 'scored', **entry}), flush=True)
    status['stage'] = 'finalizing'
    payload = ''.join(json.dumps(row)+'\n' for row in output_rows).encode()
    atomic_bytes(out/'scored-candidates.jsonl', payload)
    atomic_bytes(out/'scored-candidates.jsonl.gz', gzip.compress(payload, mtime=0))
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn']}
    if args.mode == 'gpu':
        versions.update({name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers']})
    manifest = {'status': 'complete', 'config': vars(args), 'source': ref(__file__),
        'inputs': inputs, 'model_files': model_files, 'encoder_signature': SIGNATURE,
        'policy': {'batch_size': 256, 'max_length': 32, 'fp16_backbone': True,
            'cls_and_l2': 'float32', 'nearest': 'original Encoder.nearest; GPU full candidate order chunks512',
            'unique_policy': 'sorted norm(text) per document then expand before scoring',
            'exact_id_policy': 'prepared fixed ID when non-null; otherwise pure nearest bank label',
            'bank_reencoded': False, 'fit_called': False, 'train_or_gold_read': False},
        'documents': counts, 'document_count': len(counts),
        'candidate_count': sum(c['candidates'] for c in counts),
        'unique_queries_per_document_total': sum(c['unique_queries'] for c in counts),
        'fresh_encoding_documents': sum(c['embedding_identity'] == 'fresh-encoding' for c in counts),
        'verified_cache_documents': sum(c['embedding_identity'] == 'verified-cache' for c in counts),
        'comparisons': comparisons, 'replay_tolerances': {'probability': 1e-8, 'similarity': 2e-6, 'decision_mismatches': 0},
        'gpu': gpu, 'runtime': {'python': platform.python_version(), **versions},
        'outputs': {p.name: ref(p) for p in [out/'scored-candidates.jsonl', out/'scored-candidates.jsonl.gz']},
        'seconds': time.monotonic()-started}
    save_json(out/'manifest.json', manifest)
    print(json.dumps({'event': 'complete', 'manifest': ref(out/'manifest.json'),
                      'documents': len(counts), 'candidates': manifest['candidate_count']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ['semantic-source', 'model-dir', 'bank', 'bank-labels', 'detector', 'requests',
                 'requests-sha256', 'raw-candidates', 'raw-candidates-sha256', 'input',
                 'input-sha256', 'cache-dir', 'output-dir', 'origin']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=['gpu', 'cpu-cache'], required=True)
    parser.add_argument('--expected-gpu-uuid')
    parser.add_argument('--reconstruction-manifest')
    parser.add_argument('--reference-scored')
    parser.add_argument('--reference-scored-sha256')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    status = {'stage': 'starting', 'document_index': None, 'pmc_id': None}
    save_json(out/'run.started.json', {'config': vars(args), 'source': ref(__file__), 'started_unix': time.time()})
    try:
        run(args, out, status)
    except BaseException as exc:
        save_json(out/'failure.json', {'status': 'failed', **status, 'exception_type': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(), 'source': ref(__file__),
            'existing_output_files': sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()),
            'ended_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
