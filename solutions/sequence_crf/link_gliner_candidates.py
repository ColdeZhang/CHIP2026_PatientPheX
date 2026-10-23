#!/usr/bin/env python3
"""Fold-local frozen SapBERT scoring of already frozen GLiNER spans; no gold input."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'input', 'hpo', 'candidates', 'candidate-freeze',
                 'span-training-manifest', 'span-data-manifest',
                 'semantic-source', 'model-dir', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Use a new immutable output directory'
    freeze = json.loads(Path(args.candidate_freeze).read_text())
    assert freeze['gold_read'] is False
    assert sha(args.candidates) == freeze['candidate_sha256']
    assert sha(args.input) == freeze['input_sha256']
    training = json.loads(Path(args.span_training_manifest).read_text())
    prepared = json.loads(Path(args.span_data_manifest).read_text())
    assert training['fingerprint']['data_manifest_sha256'] == sha(args.span_data_manifest)
    assert training['fingerprint']['train_sha256'] == prepared['artifacts']['nested_train32']['sha256']
    assert training['fingerprint']['excluded_input_sha256'] == freeze['input_sha256']
    assert training['epoch_results'][-1]['model_sha256'] == freeze['model_sha256']
    assert prepared['files']['nested_train'] == sha(args.train)
    spec = importlib.util.spec_from_file_location('gliner_semantic_source', args.semantic_source)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    train, inputs = sem.read_jsonl(args.train), sem.read_jsonl(args.input)
    candidates = {str(d['pmc_id']): d['candidates'] for d in sem.read_jsonl(args.candidates)}
    assert not {str(d['pmc_id']) for d in train} & {str(d['pmc_id']) for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    assert set(candidates) == {str(d['pmc_id']) for d in inputs}
    model_dir = Path(args.model_dir)
    model_source = json.loads((model_dir/'manifest.json').read_text())
    actual_model_files = {p.name: sha(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
    assert model_source['derived_weights']['sha256'] in actual_model_files.values()
    encoder = sem.Encoder(model_dir, 'sapbert', 'cuda', 256, out/'embeddings', 1)

    class UniqueEncoder:
        def encode(self, texts, label):
            import numpy as np
            names = sorted(set(texts))
            index = {text: i for i, text in enumerate(names)}
            encoded = encoder.encode(names, label+'-unique')
            return encoded[np.array([index[text] for text in texts], dtype=np.int64)]

        def nearest(self, *values):
            return encoder.nearest(*values)

    terms, canonical = sem.hpo_terms(args.hpo)
    settings = argparse.Namespace(seed=20261023, max_candidates=6000, negatives_per_doc=500)
    system = sem.System(train, terms, canonical, UniqueEncoder(), settings)
    counts = {}
    destination = out/'scored-candidates.jsonl'
    with destination.open('w') as stream:
        for doc in inputs:
            rows = candidates[str(doc['pmc_id'])]
            assert len({(e['offset'], e['length']) for e in rows}) == len(rows)
            for entity in rows:
                offset, length = int(entity['offset']), int(entity['length'])
                matches = [p for p in doc['full_text'] if int(p.get('offset', 0)) <= offset
                           and offset+length <= int(p.get('offset', 0))+len(p.get('text', ''))]
                assert len(matches) == 1
                passage = matches[0]
                local = offset-int(passage.get('offset', 0))
                assert passage['text'][local:local+length] == entity['text']
            vectors = system.encoder.encode([sem.norm(e['text']) for e in rows], 'candidate-'+str(doc['pmc_id']))
            probs = system.detector.predict_proba(vectors)[:, 1] if len(rows) else []
            indices, sims = system.encoder.nearest(vectors, system.bank)
            scored = []
            for entity, probability, index, similarity in zip(rows, probs, indices, sims):
                normalized = sem.norm(entity['text'])
                exact = normalized in system.surface
                identifier = system.best_id(normalized) if exact else system.label_ids[int(index)]
                scored.append(dict(entity, identifier=identifier, probability=float(probability),
                                   similarity=float(similarity), ontology_exact=exact))
            stream.write(json.dumps({'pmc_id': doc['pmc_id'], 'candidates': scored}, ensure_ascii=False)+'\n')
            counts[str(doc['pmc_id'])] = len(scored)
    manifest = {'config': vars(args), 'source_sha256': sha(__file__),
                'semantic_source_sha256': sha(args.semantic_source),
                'inputs': {name: sha(getattr(args, name)) for name in
                           ['train', 'input', 'hpo', 'candidates', 'candidate_freeze',
                            'span_training_manifest', 'span_data_manifest']},
                'training_document_ids': sorted(str(d['pmc_id']) for d in train),
                'input_document_ids': sorted(str(d['pmc_id']) for d in inputs),
                'training': system.training_stats, 'counts': counts,
                'model_source': model_source, 'actual_model_file_shas': actual_model_files,
                'versions': {name: importlib.metadata.version(name) for name in
                             ['torch', 'transformers', 'scikit-learn', 'numpy']},
                'output': {'path': str(destination), 'sha256': sha(destination)},
                'elapsed_seconds': time.monotonic()-started,
                'gpu': encoder.torch.cuda.get_device_name(0),
                'max_gpu_allocated_bytes': encoder.torch.cuda.max_memory_allocated(),
                'input_annotations_used': False, 'prediction_scoring': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'completed': True, 'seconds': manifest['elapsed_seconds'],
                      'counts': counts}), flush=True)


if __name__ == '__main__':
    main()
