#!/usr/bin/env python3
"""Export discarded token/link confidence from an attributed CRF checkpoint.

Replicates the source's decoding and threshold exactly, then asserts that the
resulting candidate keys equal its frozen predictions. No model is retrained.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    for k in ['source', 'pipeline', 'manifest', 'train', 'input', 'prediction', 'hpo', 'output']:
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('crf_origin', a.source)
    crf = importlib.util.module_from_spec(spec); spec.loader.exec_module(crf)
    m = json.loads(Path(a.manifest).read_text())
    assert m['source_sha256'] == crf.sha(a.source)
    assert m['train_sha256'] == crf.sha(a.train) and m['input_sha256'] == crf.sha(a.input)
    assert m['output_sha256'] == crf.sha(a.prediction)
    train, inputs = crf.read_jsonl(a.train), crf.read_jsonl(a.input)
    assert not {d['pmc_id'] for d in train} & {d['pmc_id'] for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    fingerprint = {k: m[k] for k in ['train_sha256', 'hpo_sha256', 'source_sha256', 'iterations']}
    ner, linker, _, _, _ = crf.load_pipeline(a.pipeline, crf.sha(a.train), crf.sha(a.hpo), fingerprint)
    reference = {d['pmc_id']: d for d in crf.read_jsonl(a.prediction)}
    key = lambda e: (e['offset'], e['length'], e['identifier'])
    rows = []
    for doc in inputs:
        candidates = {}
        for passage in doc['full_text']:
            tokens = crf.tokenize(passage)
            if not tokens:
                continue
            features = crf.features(tokens, passage, linker)
            for begin, end in crf.chunks(tokens):
                labels = ner.predict_single(features[begin:end])
                marginal = ner.predict_marginals_single(features[begin:end]); start = None
                for j in range(len(labels) + 1):
                    label = labels[j] if j < len(labels) else 'O'
                    if start is not None and label != 'I':
                        span = tokens[begin + start:begin + j]
                        off, finish = span[0][1], span[-1][2]
                        text = passage['text'][off - passage['offset']:finish - passage['offset']]
                        confidence = float(crf.np.mean([marginal[k].get(labels[k], 0.) for k in range(start, j)]))
                        identifier, link_score = linker.link(text)
                        if text.strip() and any(ch.isalpha() for ch in text) and confidence >= .55:
                            row = {'offset': off, 'length': finish - off, 'identifier': identifier,
                                   'ner_probability': confidence, 'link_similarity': float(link_score)}
                            candidates.setdefault(key(row), row)
                        start = None
                    if label == 'B' or (label == 'I' and start is None):
                        start = j
        assert set(candidates) == {key(e) for e in reference[doc['pmc_id']]['entities']}
        rows.append({'pmc_id': doc['pmc_id'], 'candidates': [candidates[k] for k in sorted(candidates)]})
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps({'config': vars(a),
        'source_sha256': crf.sha(__file__), 'source_model_sha256': crf.sha(a.source),
        'inputs': {k: crf.sha(getattr(a, k)) for k in ['pipeline', 'manifest', 'train', 'input', 'prediction', 'hpo']},
        'candidate_keys_equal_frozen_predictions': True, 'input_annotations_used': False,
        'output_sha256': crf.sha(out)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
