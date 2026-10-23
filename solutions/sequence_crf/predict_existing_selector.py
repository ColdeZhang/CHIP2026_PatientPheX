#!/usr/bin/env python3
"""Load a trusted fitted selector; score a prepared candidate pool without fitting.

Candidate generation and Task2 inference are upstream. The optional association
file is copied by document after patient-key validation, never regenerated.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

PUBLIC_FIELDS = ['present', 'score', 'neural', 'dictionary', 'abbreviation', 'origin_unknown']
SOURCE_FIELDS = [prefix + name for prefix in ('public_', 'bioformer_') for name in PUBLIC_FIELDS]
BOOL_FIELDS = ['exact', 'is_base', 'crf', 'crf_span', 'semantic_source'] + [
    name for name in SOURCE_FIELDS if not name.endswith('_score')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    keyed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('pmc_id'), str), 'pmc_id must be a string')
        require(row['pmc_id'] not in keyed, f'duplicate document: {row["pmc_id"]}')
        keyed[row['pmc_id']] = row
    return keyed


def validate_span(doc, entity):
    start, length = entity.get('offset'), entity.get('length')
    require(type(start) is int and type(length) is int and start >= 0 and length > 0, 'invalid span coordinates')
    identifier = entity.get('identifier')
    require(isinstance(identifier, str) and re.fullmatch(r'(?:HP:\d{7}|-1)(?:;(?:HP:\d{7}|-1))*', identifier), 'invalid HPO identifier syntax')
    require(len(identifier.split(';')) == len(set(identifier.split(';'))), 'repeated atomic HPO identifier')
    passages = [p for p in doc['full_text'] if p['offset'] <= start and start + length <= p['offset'] + len(p['text'])]
    require(len(passages) == 1, 'span must belong to exactly one passage')
    passage = passages[0]
    require(passage['text'][start-passage['offset']:start-passage['offset']+length] == entity.get('text'), 'span text/offset mismatch')


def validate_sources(inputs, bases, pools, associations):
    require(bool(inputs), 'empty input document set')
    require(set(inputs) == set(bases) == set(pools), 'input/base/candidate document sets differ')
    if associations is not None:
        require(set(inputs) == set(associations), 'association document set differs')
    for did, doc in inputs.items():
        require(not {'entities', 'association'} & doc.keys(), 'input must be annotation-free')
        require(isinstance(doc.get('full_text'), list) and isinstance(doc.get('patient'), list), 'missing full_text/patient arrays')
        for passage in doc['full_text']:
            require(type(passage.get('offset')) is int and passage['offset'] >= 0 and isinstance(passage.get('text'), str), 'invalid passage')
        patients = [p['patient_id'] for p in doc['patient']]
        require(all(isinstance(p, str) for p in patients) and len(patients) == len(set(patients)), 'invalid/repeated patient IDs')
        base_keys = set()
        for entity in bases[did]['entities']:
            validate_span(doc, entity)
            key = entity['offset'], entity['length'], entity['identifier']
            require(key not in base_keys, 'duplicate base entity')
            base_keys.add(key)
        require(not {'entities', 'association'} & pools[did].keys(), 'candidate row must not contain annotation fields')
        seen = set()
        for candidate in pools[did]['candidates']:
            validate_span(doc, candidate)
            key = candidate['offset'], candidate['length'], candidate['identifier']
            require(key not in seen, 'duplicate candidate key')
            seen.add(key)
            for name in BOOL_FIELDS:
                require(type(candidate.get(name)) is bool, f'missing/nonboolean source field: {name}')
            for name in ['probability', 'similarity', 'public_score', 'bioformer_score']:
                value = candidate.get(name)
                require(type(value) in (int, float, bool) and math.isfinite(value), f'missing/nonfinite source score: {name}')
                # Normalized floating-point dot products can slightly exceed 1;
                # retain their actual values rather than changing trained features.
                lo, hi = (-1.00001, 1.00001) if name == 'similarity' else (0, 1)
                require(lo <= value <= hi, f'out-of-range source score: {name}')
            for name in ['crf_ner', 'crf_link']:
                require(name not in candidate or candidate[name] == 0, 'this selector was trained without CRF confidence; expected absent/zero')
        if associations is not None:
            records = associations[did]['association']
            require(isinstance(records, list), 'association must be a list')
            keys = [a['patient_id'] for a in records]
            require(len(keys) == len(set(keys)) and set(keys) == set(patients), 'association patient keys differ')
            require(all(isinstance(a.get('phenotype'), list) and all(isinstance(x, str) for x in a['phenotype']) for a in records), 'invalid phenotype list')


def decode(candidates, probabilities, cut):
    # Exact function body from frozen calibrate_public_selector_cut.py; the
    # original calibration module is not imported because it contains scoring.
    assert len(candidates)==len(probabilities)
    selected=[]
    for c,p in sorted(zip(candidates,probabilities),key=lambda cp:(-cp[1],-cp[0]['length'],cp[0]['offset'])):
        assert math.isfinite(float(p)) and 0<=p<=1
        if p<cut or any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in selected):
            continue
        selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
    return sorted(selected,key=lambda e:e['offset'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'candidates', 'base', 'ontology-source', 'model', 'model-manifest', 'output-dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cut', type=float, required=True)
    parser.add_argument('--associations', help='Optional prediction JSONL; preserve its association objects exactly.')
    args = parser.parse_args()
    started = time.monotonic()
    require(math.isfinite(args.cut) and 0 <= args.cut <= 1, 'cut must be finite within [0,1]')
    output = Path(args.output_dir)
    require(not output.exists(), 'output directory already exists; refusing overwrite')
    expected = json.loads(Path(args.model_manifest).read_text())
    require(sha(args.model) == expected['model_artifact_sha256'], 'model hash differs from its manifest')
    runtime = {'python': platform.python_version(), 'sklearn': sklearn.__version__}
    require(runtime == expected['runtime'], 'runtime differs from fitted model manifest')
    source = Path(args.ontology_source).resolve()
    sys.path.insert(0, str(source))
    multi = importlib.import_module('calibrate_multi')
    modules = [multi, multi.single, multi.single.baseline, multi.single.enhance, multi.single.merge_semantic]
    for module in modules:
        require(Path(module.__file__).resolve().parent == source, 'unexpected imported source directory')
        require(sha(module.__file__) == expected['dependencies'][module.__name__], f'feature dependency changed: {module.__name__}')
    # Only trusted local checkpoints created by this project should be loaded.
    with Path(args.model).open('rb') as stream:
        saved = pickle.load(stream)
    model, vectorizer = saved['model'], saved['vectorizer']
    require(type(model) is HistGradientBoostingClassifier and type(vectorizer) is DictVectorizer, 'unexpected model/vectorizer types')
    require(list(vectorizer.feature_names_) == expected['features'] and model.n_features_in_ == len(expected['features']), 'fitted feature schema differs')
    require(model.classes_.tolist() == [False, True], 'unexpected class order')
    inputs, bases, pools = [read_rows(p) for p in [args.input, args.base, args.candidates]]
    associations = read_rows(args.associations) if args.associations else None
    validate_sources(inputs, bases, pools, associations)
    predictions, scored, unknown_categories = [], [], set()
    for did, doc in inputs.items():
        # Existing selector probabilities are explicitly discarded. They are
        # never a feature or a shortcut to predict_proba.
        candidates = [{k: v for k, v in c.items() if k != 'selector_probability'} for c in pools[did]['candidates']]
        features = [multi.features(doc, c, bases[did]) | {name: c[name] for name in SOURCE_FIELDS} for c in candidates]
        for row in features:
            for key, value in row.items():
                if isinstance(value, str) and f'{key}={value}' not in vectorizer.vocabulary_:
                    unknown_categories.add(f'{key}={value}')
        probabilities = model.predict_proba(vectorizer.transform(features))[:, 1] if features else []
        require(all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probabilities), 'invalid model probabilities')
        prediction = {'pmc_id': did, 'pmid': doc.get('pmid'), 'entities': decode(candidates, probabilities, args.cut)}
        if associations is not None:
            prediction['association'] = associations[did]['association']
        predictions.append(prediction)
        scored.append({'pmc_id': did, 'candidates': [dict(c, selector_probability=float(p)) for c, p in zip(candidates, probabilities)]})
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('predictions.jsonl', predictions), ('selector-scores.jsonl', scored)]:
        (output/name).write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in rows))
    paths = {name: getattr(args, name) for name in ['input', 'base', 'candidates', 'model', 'model_manifest', 'associations'] if getattr(args, name)}
    manifest = {'arguments': vars(args), 'source_sha256': sha(__file__), 'inputs': {k: {'path': v, 'sha256': sha(v)} for k, v in paths.items()},
        'runtime': runtime, 'feature_dependencies': expected['dependencies'], 'features': expected['features'], 'unknown_categories_ignored_by_fitted_vectorizer': sorted(unknown_categories),
        'documents': len(inputs), 'candidates': sum(len(r['candidates']) for r in scored), 'cut': args.cut,
        'associations': 'preserved verbatim after document/patient key validation' if associations is not None else 'omitted: Task1-only output',
        'fit_performed': False, 'evaluation_performed': False, 'stored_selector_probabilities_used': False,
        'outputs': {name: sha(output/name) for name in ['predictions.jsonl', 'selector-scores.jsonl']}, 'seconds': time.monotonic()-started,
        'limitations': ['Upstream candidates and base predictions must follow the fitted source/normalization policy; supplied provenance is not inferred from spans.',
                       'Identifier syntax is checked; current-HPO membership and upstream training isolation remain candidate-generator responsibilities.',
                       'No Task2 candidate generation or reassignment. Model pickle must be from a trusted source.']}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'documents': len(inputs), 'outputs': manifest['outputs'], 'seconds': manifest['seconds']}))


if __name__ == '__main__':
    main()
