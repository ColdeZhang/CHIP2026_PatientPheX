#!/usr/bin/env python3
"""Fixed OOF source-HGB plus a separately verified public candidate source.

This is a new source-plus-features method increment. The frozen baseline is
reused, not refitted. No inference, validation gold, or threshold selection.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')

PUBLIC_FEATURES = ['public_present', 'public_score', 'public_neural',
                   'public_dictionary', 'public_abbreviation', 'public_origin_unknown']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(c):
    return int(c['offset']), int(c['length']), c['identifier']


def merge_pool(original, public):
    """Keep old source fields and order; merge public evidence by exact triple."""
    pooled = {key(c): dict(c, **{name: False for name in PUBLIC_FEATURES}) for c in original}
    assert len(pooled) == len(original)
    crf_spans = {(c['offset'], c['length']) for c in original if c['crf_span']}
    for c in public:
        assert c['public_present'] is True
        assert math.isfinite(c['public_score']) and 0 <= c['public_score'] <= 1
        assert all(isinstance(c[name], bool) for name in PUBLIC_FEATURES if name != 'public_score')
        k = key(c)
        if k not in pooled:
            # These are explicit missing semantic values, using the already
            # frozen CRF-only convention. They are not computed Sap scores.
            pooled[k] = {name: c[name] for name in ['offset', 'length', 'identifier', 'text']}
            pooled[k].update(probability=.5, similarity=0., exact=False, is_base=False,
                crf=False, crf_span=k[:2] in crf_spans, semantic_source=False,
                **{name: False for name in PUBLIC_FEATURES})
        target = pooled[k]
        assert target['text'] == c['text']
        target['public_score'] = max(float(target['public_score']), c['public_score'])
        # Preserve actual old-head provenance for review, without treating
        # arbitrary ontology IDs or head indices as learned feature columns.
        for name in ['public_raw_hpo_ids', 'public_matched_neural_head_columns']:
            target[name] = sorted(set(target.get(name, [])) | set(c.get(name, [])))
        for name in PUBLIC_FEATURES:
            if name != 'public_score':
                target[name] = target[name] or c[name]
    return list(pooled.values())


def select(candidates, probabilities):
    assert len(candidates) == len(probabilities)
    selected = []
    for c, p in sorted(zip(candidates, probabilities),
                       key=lambda cp: (-cp[1], -cp[0]['length'], cp[0]['offset'])):
        assert math.isfinite(float(p)) and 0 <= p <= 1
        if p < .4 or any(c['offset'] < e['offset']+e['length'] and
                         e['offset'] < c['offset']+c['length'] for e in selected):
            continue
        selected.append({k: c[k] for k in ['identifier', 'offset', 'length', 'text']}
                        | {'type': 'Phenotype', 'note': None})
    return sorted(selected, key=lambda e: e['offset'])


def public_rows(path, docs):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        did = row['pmc_id']
        assert did in docs and did not in result
        cs = row['candidates']
        for c in cs:
            offset, length, _ = key(c)
            assert offset >= 0 and length > 0
            matches = [p for p in docs[did]['full_text']
                       if int(p['offset']) <= offset and offset+length <= int(p['offset'])+len(p['text'])]
            assert len(matches) == 1
            passage = matches[0]
            start = offset-int(passage['offset'])
            assert passage['text'][start:start+length] == c['text']
        result[did] = cs
    assert set(result) == set(docs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['ontology-source', 'reference', 'public', 'public-manifest', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    start = time.monotonic()
    sys.path.insert(0, a.ontology_source)
    multi = importlib.import_module('calibrate_multi')
    single = multi.single
    reference = json.loads(Path(a.reference).read_text())
    config = reference['config']
    assert sha(a.reference) == '51c5b7199917aba56b893d45a04d7878e6df8454de0a1acb620372bc3a142813'
    assert sha(multi.__file__) == reference['source_sha256']
    for module in [single, single.baseline, single.enhance, single.merge_semantic]:
        assert sha(module.__file__) == reference['dependencies'][module.__name__]
    assert config['threshold'] == .4 and config['without_crf'] is False and config['crf_confidence'] is None
    assert all(sha(config[k]) == digest for k, digest in reference['inputs'].items())
    assert sha(config['output']) == reference['output_sha256']
    train, inputs = multi.keyed(config['train']), multi.keyed(config['input'])
    assert len(train) == 64 and len(inputs) == 16 and not train.keys() & inputs.keys()
    assert all('entities' not in d and 'association' not in d for d in inputs.values())
    public_manifest = json.loads(Path(a.public_manifest).read_text())
    assert public_manifest['output_sha256'] == sha(a.public)
    assert public_manifest['public_model_task_fit'] is False
    assert public_manifest['model_weight_sha256'] == 'd58af55a68840a24b052a775cf0cb32b8993a1fa4823c76aad10d3e6299d7ec3'
    # Public input80 is label-free: existing train64 cache plus independent
    # originaldev16 cache; no public model task fit in either forward pass.
    pub_input = multi.keyed(public_manifest['input_path'])
    assert sha(public_manifest['input_path']) == public_manifest['input_sha256']
    docs = dict(train, **inputs)
    assert len(pub_input) == 80 and set(pub_input) == set(docs)
    for did, doc in docs.items():
        assert not ({'entities', 'association'} & pub_input[did].keys())
        assert {k: v for k, v in doc.items() if k not in ['entities', 'association']} == pub_input[did]
    public = public_rows(a.public, pub_input)
    specs = json.loads(Path(config['training_manifest']).read_text())
    assert len(specs) == 4
    xs, ys, seen, origins = [], [], set(), []
    old_count = public_only_count = 0
    for spec in specs:
        fold = multi.keyed(spec['gold'])
        oldbase = multi.keyed(spec['base'])
        raw = single.read_candidates(spec['candidates'])
        cm = json.loads(Path(spec['candidate_manifest']).read_text())
        assert cm['inputs']['input'] == sha(spec['input'])
        assert cm['inputs']['train'] == sha(spec['generator_train'])
        assert cm['output']['sha256'] == sha(spec['candidates'])
        generator_ids = set(multi.keyed(spec['generator_train']))
        assert len(generator_ids) == 48 and generator_ids <= set(train)
        assert not generator_ids & set(fold) and not seen & set(fold)
        assert set(fold) <= set(train) and len(fold) == 16
        seen |= set(fold)
        crfs = multi.checked_crf(spec['crf'], spec['crf_manifest'],
                                 spec['generator_train'], spec['input'], set(train))
        for did, doc in fold.items():
            assert doc == train[did]
            truth = {}
            for e in doc['entities']:
                if str(e.get('note', '')).upper() != 'NO':
                    truth.setdefault((e['offset'], e['length']), set()).update(e['identifier'].split(';'))
            old = multi.pool(doc, oldbase[did], raw[did], crfs[did])
            cs = merge_pool(old, public[did])
            old_count += len(old)
            public_only_count += len(cs)-len(old)
            for c in cs:
                ids = truth.get((c['offset'], c['length']), set())
                ys.append(bool(ids and ('-1' in ids or set(c['identifier'].split(';')) <= ids)))
                xs.append(multi.features(doc, c, oldbase[did]) |
                          {name: c[name] for name in PUBLIC_FEATURES})
        origins.append({k: sha(spec[k]) for k in spec})
    assert seen == set(train) and old_count == reference['training_examples']
    vectorizer = single.DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(xs)
    model = single.HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=5., learning_rate=.06,
        random_state=20261023, early_stopping=False)
    assert model.get_params() == reference['estimator']
    model.fit(matrix, ys)
    bases, raw = multi.keyed(config['base']), single.read_candidates(config['candidates'])
    crfs = multi.checked_crf(config['crf'], config['crf_manifest'], config['train'], config['input'], set(train))
    prediction, scores = [], []
    old_input_count = public_only_input_count = 0
    for did, doc in inputs.items():
        old = multi.pool(doc, bases[did], raw[did], crfs[did])
        cs = merge_pool(old, public[did])
        old_input_count += len(old)
        public_only_input_count += len(cs)-len(old)
        probabilities = model.predict_proba(vectorizer.transform([
            multi.features(doc, c, bases[did]) | {name: c[name] for name in PUBLIC_FEATURES}
            for c in cs]))[:, 1]
        selected = select(cs, probabilities)
        prediction.append({'pmc_id': did, 'pmid': doc.get('pmid'), 'entities': selected,
                           'association': single.merge_semantic.assign(doc, selected)})
        scores.append({'pmc_id': did, 'candidates': [dict(c, selector_probability=float(p))
                       for c, p in zip(cs, probabilities)]})
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists()
    (out/'baseline.jsonl').write_bytes(Path(config['output']).read_bytes())
    for filename, rows in [('augmented.jsonl', prediction), ('selector-scores.jsonl', scores)]:
        (out/filename).write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in rows))
    manifest = dict(source_sha256=sha(__file__), arguments=vars(a),
        old_reference_sha256=sha(a.reference), baseline_prediction_sha256=sha(out/'baseline.jsonl'),
        public_sha256=sha(a.public), public_manifest_sha256=sha(a.public_manifest),
        dependencies={module.__name__: sha(module.__file__) for module in
                      [multi, single, single.baseline, single.enhance, single.merge_semantic]},
        training_origins=origins, training_documents=len(train), old_training_examples=old_count,
        added_public_training_examples=public_only_count, training_examples=len(ys),
        old_input_candidates=old_input_count, added_public_input_candidates=public_only_input_count,
        training_positive=sum(ys), features=vectorizer.feature_names_, estimator=model.get_params(),
        relation='Same legacy patient rules, all-pool learned selection permits deletions/replacements.',
        outputs={name: sha(out/name) for name in ['baseline.jsonl','augmented.jsonl','selector-scores.jsonl']},
        seconds=time.monotonic()-start, original_dev_read=False, evaluation_gold_read=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['training_examples','training_positive','seconds','outputs']}))


if __name__ == '__main__':
    main()
