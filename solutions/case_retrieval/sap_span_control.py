#!/usr/bin/env python3
"""Prepare, score and apply a frozen Sap span filter at one operating point.

No gold, classifier training or threshold search occurs here. Only the explicit
score subcommand loads an encoder. Existing linked HPO IDs and CRF entities are
retained verbatim. Association assignment is an unchanged downstream step for
both the unfiltered and filtered comparison.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

PROBABILITY = 0.65
SIMILARITY = 0.92


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(rows):
    result = {str(row['pmc_id']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate document IDs')
    return result


def key(entity):
    return int(entity['offset']), int(entity['length']), entity['text']


def overlap(entity, others):
    start, length, _ = key(entity)
    return any(start < other['offset'] + other['length']
               and other['offset'] < start + length for other in others)


def exact_slice(doc, entity):
    start, length, text = key(entity)
    return length == len(text) and length > 0 and any(
        int(p.get('offset', 0)) <= start
        and start + length <= int(p.get('offset', 0)) + len(p.get('text', ''))
        and p['text'][start-int(p.get('offset', 0)):start-int(p.get('offset', 0))+length] == text
        for p in doc.get('full_text', []))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(args):
    train = indexed(read(args.train)); inputs = indexed(read(args.input))
    baseline = indexed(read(args.baseline))
    if set(train) & set(inputs) or set(inputs) != set(baseline):
        raise ValueError('Training/input overlap or baseline coverage mismatch')
    if any('entities' in d or 'association' in d for d in inputs.values()):
        raise ValueError('Inference input must be label free')
    frozen = json.loads(Path(args.frozen_manifest).read_text())
    for field in ['train', 'input']:
        if sha(getattr(args, field)) != frozen['inputs'][field]:
            raise ValueError(f'Frozen {field} bytes differ')
    if sha(args.frozen_candidates) != frozen['output']['sha256']:
        raise ValueError('Frozen candidate bytes differ')
    if frozen['training']['training_docs'] != len(train):
        raise ValueError('Frozen training document count differs')
    found = json.loads(Path(args.items).read_text())['occurrences']
    if set(found) - set(inputs):
        raise ValueError('Unknown extraction document')
    rows = []; counts = Counter()
    for pmc, doc in inputs.items():
        candidates = []; seen = set()
        for item in found.get(pmc, []):
            k = key(item)
            if k in seen or not exact_slice(doc, item):
                raise ValueError('Duplicate or nonliteral extraction span')
            seen.add(k)
            candidates.append({'offset': k[0], 'length': k[1], 'text': k[2]})
            counts['occurrences'] += 1
            counts['nonoverlapping_baseline'] += not overlap(item, baseline[pmc]['entities'])
        rows.append({'pmc_id': doc['pmc_id'], 'candidates': candidates})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    target = out / 'requests.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = {
        'schema': 'frozen-sap-span-request-v1', 'source_sha256': sha(__file__),
        'files': {name: {'path': str(Path(getattr(args, name)).resolve()),
                         'sha256': sha(getattr(args, name))}
                  for name in ['train', 'input', 'baseline', 'items',
                               'frozen_manifest', 'frozen_candidates']},
        'train_ids': sorted(train), 'input_ids': sorted(inputs),
        'request_sha256': sha(target), 'counts': dict(counts),
        'operating_point': {'probability_gte': PROBABILITY, 'similarity_gte': SIMILARITY},
        'gold_used': False, 'scoring_performed': False,
        'semantic_source_sha256': frozen['semantic_source_sha256'],
        'model_manifest': frozen['model_manifest'],
    }
    save_json(out / 'request.manifest.json', manifest)
    print(json.dumps({'request_sha256': manifest['request_sha256'], 'counts': dict(counts)}))


def score(args):
    # Imports and device initialization are confined to this explicit operation.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import importlib.util
    import time
    import numpy as np
    from scipy.special import expit
    started = time.monotonic()
    reqdir = Path(args.request_dir); rebuilt_dir = Path(args.reconstruction_dir)
    request = json.loads((reqdir / 'request.manifest.json').read_text())
    reconstruction_path = rebuilt_dir / 'reconstruction.manifest.json'
    rebuilt = json.loads(reconstruction_path.read_text())
    if not rebuilt.get('passed') or sha(args.semantic_source) != rebuilt['semantic_source_sha256']:
        raise ValueError('Unverified reconstruction or different encoder source')
    if sha(reqdir / 'requests.jsonl') != request['request_sha256']:
        raise ValueError('Request bytes differ')
    if any(rebuilt['inputs'][k] != request['files'][k]['sha256'] for k in ['train', 'input']):
        raise ValueError('Request and classifier folds differ')
    if rebuilt['reference_candidate_sha256'] != request['files']['frozen_candidates']['sha256']:
        raise ValueError('Request and classifier reference pools differ')
    detector_path = rebuilt_dir / 'reconstructed-detector.npz'
    if sha(detector_path) != rebuilt['detector']['sha256'] or sha(args.bank) != rebuilt['bank_cache']['sha256']:
        raise ValueError('Detector or ontology/training bank bytes differ')
    for name, digest in rebuilt['actual_model_files_sha256'].items():
        if sha(Path(args.model_dir) / name) != digest:
            raise ValueError('Encoder bytes differ: ' + name)
    requests = indexed(read(reqdir / 'requests.jsonl'))
    if set(requests) != set(rebuilt['input_ids']):
        raise ValueError('Request document IDs differ')
    spec = importlib.util.spec_from_file_location('frozen_sap_encoder', args.semantic_source)
    sem = importlib.util.module_from_spec(spec); spec.loader.exec_module(sem)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    encoder = sem.Encoder(args.model_dir, 'sapbert', args.device, 256, out / 'embeddings', 1)
    if encoder.signature != rebuilt['encoder_signature']:
        raise ValueError('Encoder cache signature differs')
    detector = np.load(detector_path, allow_pickle=False)
    if detector['classes'].tolist() != [0, 1] or detector['coef'].shape != (1, 768):
        raise ValueError('Unexpected detector representation')
    bank = np.load(args.bank, allow_pickle=False)
    if bank.shape != (rebuilt['training']['semantic_bank'], 768):
        raise ValueError('Unexpected bank shape')
    rows = []; total = 0
    for pmc, row in requests.items():
        cs = row['candidates']; texts = [sem.norm(c['text']) for c in cs]
        names = sorted(set(texts)); lookup = {text: i for i, text in enumerate(names)}
        # Exactly the original UniqueEncoder policy and per-document batching.
        values = encoder.encode(names, 'candidates-' + pmc + '-unique')
        vectors = values[np.asarray([lookup[text] for text in texts], dtype=np.int64)]
        probabilities = expit(vectors @ detector['coef'].T + detector['intercept']).ravel()
        _, similarities = encoder.nearest(vectors, bank)
        rows.append({'pmc_id': row['pmc_id'], 'candidates': [
            dict(c, probability=float(p), similarity=float(s))
            for c, p, s in zip(cs, probabilities, similarities)]})
        total += len(cs)
        print(json.dumps({'event': 'scored', 'pmc_id': pmc, 'candidates': len(cs)}), flush=True)
    target = out / 'scores.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = {
        'request_sha256': request['request_sha256'],
        'reconstruction_manifest_sha256': sha(reconstruction_path),
        'output_sha256': sha(target), 'source_sha256': sha(__file__),
        'semantic_source_sha256': sha(args.semantic_source),
        'detector_sha256': sha(detector_path), 'bank_sha256': sha(args.bank),
        'model_files_sha256': rebuilt['actual_model_files_sha256'],
        'configuration': {'device': args.device, 'batch_size': 256, 'max_length': 32,
                          'encoder': 'sapbert', 'pooling': 'CLS_L2', 'threads': 1,
                          'unique_policy': 'sorted unique sem.norm(surface) per document',
                          'precision': 'FP16 backbone, FP32 normalized CLS' if args.device.startswith('cuda') else 'FP32 backbone and CLS'},
        'counts': {'documents': len(rows), 'occurrences': total},
        'gold_used': False, 'classifier_fit': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    save_json(out / 'scores.manifest.json', manifest)
    print(json.dumps({'event': 'complete', 'output_sha256': sha(target), 'occurrences': total}), flush=True)


def merge(args):
    reqdir = Path(args.request_dir)
    request = json.loads((reqdir / 'request.manifest.json').read_text())
    if sha(reqdir / 'requests.jsonl') != request['request_sha256']:
        raise ValueError('Request bytes differ')
    if sha(args.baseline) != request['files']['baseline']['sha256']:
        raise ValueError('Baseline differs from request')
    rebuilt = json.loads(Path(args.reconstruction_manifest).read_text())
    if not rebuilt.get('passed'):
        raise ValueError('Frozen classifier reconstruction has not passed replay')
    for name in ['train', 'input']:
        if rebuilt['inputs'][name] != request['files'][name]['sha256']:
            raise ValueError('Reconstructed classifier fold differs')
    if rebuilt['reference_candidate_sha256'] != request['files']['frozen_candidates']['sha256']:
        raise ValueError('Reconstruction reference pool differs')
    scored_manifest = json.loads(Path(args.score_manifest).read_text())
    expected = {'request_sha256': request['request_sha256'],
                'reconstruction_manifest_sha256': sha(args.reconstruction_manifest),
                'output_sha256': sha(args.scores)}
    if any(scored_manifest.get(k) != value for k, value in expected.items()):
        raise ValueError('Score provenance does not match request/reconstruction/output')
    scores = indexed(read(args.scores)); linked = indexed(read(args.linked))
    baseline = indexed(read(args.baseline)); requested = indexed(read(reqdir / 'requests.jsonl'))
    if not set(scores) == set(linked) == set(baseline) == set(requested):
        raise ValueError('Document coverage differs')
    output = []; counts = Counter()
    for pmc, base in baseline.items():
        cs = scores[pmc]['candidates']; bykey = {key(c): c for c in cs}
        if len(bykey) != len(cs) or set(bykey) != {key(c) for c in requested[pmc]['candidates']}:
            raise ValueError('Scores must cover each exact requested occurrence once')
        for c in cs:
            p, similarity = float(c['probability']), float(c['similarity'])
            if not (math.isfinite(p) and 0 <= p <= 1 and math.isfinite(similarity)
                    and -1.00001 <= similarity <= 1.00001):
                raise ValueError('Invalid Sap score')
        entities = list(base['entities'])
        for entity in linked[pmc]['entities']:
            k = key(entity)
            if k not in bykey:
                raise ValueError('Linked entity absent from original span request')
            counts['linked'] += 1
            if overlap(entity, base['entities']):
                counts['overlapping_baseline'] += 1
                continue
            counts['eligible_nonoverlap'] += 1
            score = bykey[k]
            if score['probability'] >= PROBABILITY and score['similarity'] >= SIMILARITY:
                entities.append(entity); counts['accepted_additions'] += 1
            else:
                counts['filtered_additions'] += 1
        unique = {(e['offset'], e['length'], e['identifier']): e for e in entities}
        output.append(dict(base, entities=[unique[k] for k in sorted(unique)]))
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output))
    save_json(target.with_suffix('.manifest.json'), {
        'schema': 'frozen-sap-span-filter-v1', 'source_sha256': sha(__file__),
        'files': {name: sha(getattr(args, name)) for name in
                  ['linked', 'scores', 'score_manifest', 'reconstruction_manifest', 'baseline']},
        'request_manifest_sha256': sha(reqdir / 'request.manifest.json'),
        'operating_point': {'probability_gte': PROBABILITY, 'similarity_gte': SIMILARITY},
        'counts': dict(counts), 'output_sha256': sha(target), 'gold_used': False,
        'association_policy': 'Preserved baseline; apply identical downstream relation assignment to both arms.',
        'link_policy': 'Preserve resolved HPO IDs; filter only LLM additions without any baseline character overlap.',
    })
    print(json.dumps(dict(counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ['train', 'input', 'baseline', 'items', 'frozen-manifest', 'frozen-candidates', 'output-dir']:
        prep.add_argument('--' + name, required=True)
    apply = commands.add_parser('merge')
    for name in ['request-dir', 'linked', 'scores', 'score-manifest', 'reconstruction-manifest', 'baseline', 'output']:
        apply.add_argument('--' + name, required=True)
    scoring = commands.add_parser('score')
    for name in ['request-dir', 'reconstruction-dir', 'semantic-source', 'model-dir', 'bank', 'output-dir']:
        scoring.add_argument('--' + name, required=True)
    scoring.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    {'prepare': prepare, 'score': score, 'merge': merge}[args.command](args)


if __name__ == '__main__':
    main()
