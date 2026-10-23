#!/usr/bin/env python3
"""Frozen biomedical encoder + jointly trained lexical/occurrence heads.

Train only from the supplied document set and designated ontology. SapBERT
linking remains frozen and independent. No input labels or early stopping.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    for name in ['train', 'input', 'train-rows', 'input-rows', 'data-manifest',
                 'inference-manifest', 'hpo', 'model', 'link-model', 'semantic-source', 'output-dir']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=128)
    a = ap.parse_args()
    import numpy as np
    import torch
    import transformers
    import distant_model as dm
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 1
    random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
    torch.cuda.manual_seed_all(20261023)
    started = time.monotonic()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, docs = read(a.train), read(a.input)
    train_ids = {str(d['pmc_id']) for d in train}; input_ids = {str(d['pmc_id']) for d in docs}
    assert not train_ids & input_ids and all('entities' not in d and 'association' not in d for d in docs)
    rows, inference = read(a.train_rows), read(a.input_rows)
    data_meta = json.loads(Path(a.data_manifest).read_text())
    inf_meta = json.loads(Path(a.inference_manifest).read_text())
    assert data_meta['inputs']['train']['sha256'] == inf_meta['train_sha256'] == sha(a.train)
    assert data_meta['inputs']['hpo']['sha256'] == sha(a.hpo)
    assert inf_meta['input_sha256'] == sha(a.input)
    def uncompressed_sha(path):
        raw = Path(path).read_bytes()
        return hashlib.sha256(gzip.decompress(raw) if str(path).endswith('.gz') else raw).hexdigest()
    assert data_meta['output']['sha256'] == uncompressed_sha(a.train_rows)
    assert inf_meta['output_sha256'] == uncompressed_sha(a.input_rows)
    assert {r['pmc_id'] for r in rows if 'pmc_id' in r} == train_ids
    assert {r['pmc_id'] for r in inference} == input_ids
    assert all('lexical_y' not in r and 'occurrence_y' not in r for r in inference)
    tokenizer = dm.AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    max_first = max(len(tokenizer.encode(r['span'], add_special_tokens=False)) for r in rows + inference)
    assert max_first <= 125, 'Candidate must fit without truncation'
    model = dm.SpanDetector(a.model, frozen=True).cuda()
    model.eval()

    def encode(records):
        blocks = []
        with torch.inference_mode():
            for start in range(0, len(records), a.batch_size):
                batch = records[start:start + a.batch_size]
                packed = tokenizer([r['span'] for r in batch], [r['context'] for r in batch],
                    padding=True, truncation='only_second', max_length=128,
                    return_special_tokens_mask=True, return_tensors='pt')
                packed = {k: v.cuda() for k, v in packed.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    blocks.append(model.features(packed).half().cpu())
        return torch.cat(blocks).clone()

    train_features = encode(rows)
    print(json.dumps({'stage': 'encoded_training', 'rows': len(rows), 'seconds': time.monotonic() - started}), flush=True)
    targets = {k: torch.tensor([r[k] for r in rows], dtype=torch.float32) for k in
               ['lexical_y', 'occurrence_y', 'weight']}
    weights = dm.class_weights(rows)
    optimizer = dm.make_optimizer(model, finetune=False)
    model.train(); losses = []
    for epoch in range(a.epochs):
        order = torch.randperm(len(rows)); total = 0.
        for start in range(0, len(rows), 256):
            indices = order[start:start + 256]
            outputs = model.from_features(train_features[indices].cuda().float())
            batch = {k: v[indices].cuda() for k, v in targets.items()}
            loss = dm.objective(outputs, batch, weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(rows))
        print(json.dumps({'stage': 'frozen_head_training', 'epoch': epoch + 1, 'loss': losses[-1]}), flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('encoder.')}, out / 'heads.pt')
    model.eval(); infer_features = encode(inference); scores = []
    with torch.inference_mode():
        for start in range(0, len(inference), 512):
            output = model.from_features(infer_features[start:start + 512].cuda().float())
            scores.extend(zip(output['lexical'].sigmoid().cpu().tolist(), output['occurrence'].sigmoid().cpu().tolist()))
    del train_features, infer_features, model
    torch.cuda.empty_cache()
    spec = importlib.util.spec_from_file_location('semantic_encoder', a.semantic_source)
    sem = importlib.util.module_from_spec(spec); spec.loader.exec_module(sem)
    encoder = sem.Encoder(a.link_model, 'sapbert', 'cuda', 256, out / 'link-embeddings', 1)

    class UniqueEncoder:
        def encode(self, texts, label):
            unique = sorted(set(texts)); index = {s: i for i, s in enumerate(unique)}
            vectors = encoder.encode(unique, label + '-unique')
            return vectors[np.array([index[t] for t in texts])]

        def nearest(self, *args):
            return encoder.nearest(*args)

    terms, canonical = sem.hpo_terms(a.hpo)
    system = sem.System(train, terms, canonical, UniqueEncoder(),
        argparse.Namespace(seed=20261023, max_candidates=6000, negatives_per_doc=500))
    bydoc = {}
    for r, (lexical, occurrence) in zip(inference, scores):
        bydoc.setdefault(r['pmc_id'], []).append(dict(r, lexical=lexical, occurrence=occurrence))
    output = out / 'scored-candidates.jsonl.gz'; counts = {}
    with gzip.open(output, 'wt') as f:
        for doc in docs:
            did = str(doc['pmc_id']); candidates = bydoc[did]
            vectors = system.encoder.encode([sem.norm(c['span']) for c in candidates], 'candidates-' + did)
            legacy = system.detector.predict_proba(vectors)[:, 1]
            indices, similarities = encoder.nearest(vectors, system.bank)
            cs = []
            for c, p, index, similarity in zip(candidates, legacy, indices, similarities):
                surface = sem.norm(c['span']); exact = surface in system.surface
                identifier = system.best_id(surface) if exact else system.label_ids[int(index)]
                cs.append({'offset': c['offset'], 'length': c['length'], 'text': c['span'],
                    'identifier': identifier, 'exact': exact, 'similarity': float(similarity),
                    'probability': c['occurrence'], 'legacy_probability': float(p),
                    'lexical_probability': c['lexical']})
            f.write(json.dumps({'pmc_id': doc['pmc_id'], 'candidates': cs}) + '\n'); f.flush()
            counts[did] = len(cs)
            print(json.dumps({'stage': 'linked', 'pmc_id': did, 'candidates': len(cs)}), flush=True)
    manifest = {'config': vars(a), 'source_sha256': sha(__file__), 'model_source_sha256': sha(dm.__file__),
        'semantic_source_sha256': sha(a.semantic_source), 'parameter_policy': 'frozen 110M encoder; only MLP/two heads trained',
        'inputs': {k: sha(getattr(a, k)) for k in ['train', 'input', 'train_rows', 'input_rows', 'data_manifest', 'inference_manifest', 'hpo']},
        'base_files': {p.name: sha(p) for p in Path(a.model).iterdir() if p.suffix in {'.bin', '.safetensors', '.json', '.txt'}},
        'link_model_manifest': json.loads((Path(a.link_model) / 'manifest.json').read_text()),
        'train_documents': len(train_ids), 'input_documents': len(input_ids), 'training_rows': len(rows),
        'class_weights': weights, 'losses': losses, 'max_first_sequence_tokens': max_first, 'counts': counts,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0),
        'max_gpu_allocated': torch.cuda.max_memory_allocated(), 'heads_sha256': sha(out / 'heads.pt'),
        'output_sha256': sha(output), 'seconds': time.monotonic() - started, 'input_annotations_used': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'finished', 'seconds': manifest['seconds']}), flush=True)


if __name__ == '__main__':
    main()
