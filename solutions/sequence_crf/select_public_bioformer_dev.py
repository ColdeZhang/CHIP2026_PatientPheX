#!/usr/bin/env python3
"""Full64 strict-OOF dual-public HGB; fixed .4 entities with original SFT associations."""
import argparse
import importlib
import json
import math
import pickle
import platform
import sklearn
import os
from pathlib import Path
import sys
import time
import select_public_candidates as legacy
sha,key,select,public_rows,merge_pool = legacy.sha,legacy.key,legacy.select,legacy.public_rows,legacy.merge_pool
PUBLIC_FEATURES = legacy.PUBLIC_FEATURES
BIOFORMER_FEATURES = [name.replace('public_', 'bioformer_', 1) for name in PUBLIC_FEATURES]
BIOFORMER_WEIGHT = '867819ac932e11250c42ca13c49eba3711a7f24b170e24bfb31b65589d0024a1'
LEGACY_SOURCES = {'3378df59be8c13c45d07f9c17575523d58906837bba9c6b3f3cc814d3a2f7bd6',
                  'c258115c054e3c32493de45291acb47a0680517c3ef138357b473454684daa54',
'0857b8ea41b0e585627afd7750dbca108aebb6d21620404811f6a2efd40d69a2'}
REFERENCES = {'51c5b7199917aba56b893d45a04d7878e6df8454de0a1acb620372bc3a142813'}


def merge_bioformer(original, bioformer):
    """Never overwrite PubMed evidence; preserve model-specific head metadata."""
    pooled = {key(c): dict(c, **{name: False for name in BIOFORMER_FEATURES}) for c in original}
    assert len(pooled) == len(original)
    crf_spans = {(c['offset'], c['length']) for c in original if c['crf_span']}
    for c in bioformer:
        assert c['bioformer_present'] is True
        assert math.isfinite(c['bioformer_score']) and 0 <= c['bioformer_score'] <= 1
        assert all(isinstance(c[name],bool) for name in BIOFORMER_FEATURES if name!='bioformer_score')
        k=key(c)
        if k not in pooled:
            pooled[k]={name:c[name] for name in ['offset','length','identifier','text']}
            pooled[k].update(probability=.5,similarity=0.,exact=False,is_base=False,
                crf=False,crf_span=k[:2] in crf_spans,semantic_source=False,
                **{name:False for name in PUBLIC_FEATURES+BIOFORMER_FEATURES})
        target=pooled[k]
        assert target['text']==c['text']
        target['bioformer_score']=max(float(target['bioformer_score']),c['bioformer_score'])
        for name in ['bioformer_raw_hpo_ids','bioformer_matched_neural_head_columns']:
            target[name]=sorted(set(target.get(name,[]))|set(c.get(name,[])))
        for name in BIOFORMER_FEATURES:
            if name!='bioformer_score':target[name]=target[name] or c[name]
    return list(pooled.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['ontology-source', 'reference', 'public', 'public-manifest', 'public-reference', 'bioformer', 'bioformer-manifest', 'baseline-prediction', 'association-prediction', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    start = time.monotonic()
    sys.path.insert(0, a.ontology_source)
    multi = importlib.import_module('calibrate_multi')
    single = multi.single
    reference = json.loads(Path(a.reference).read_text())
    config = reference['config']
    assert sha(a.reference) in REFERENCES
    previous = json.loads(Path(a.public_reference).read_text())
    assert previous['old_reference_sha256'] == sha(a.reference)
    assert previous['source_sha256'] in LEGACY_SOURCES
    previous_prediction = Path(previous['arguments']['output_dir'])/'augmented.jsonl'
    assert sha(previous_prediction) == previous['outputs']['augmented.jsonl']
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
    # Public input80 is label-free and exactly the same texts/patients as the
    # fixed training64 + originaldev input16. Its forward pass is fold-independent.
    pub_input = multi.keyed(public_manifest['input_path'])
    assert sha(public_manifest['input_path']) == public_manifest['input_sha256']
    docs = dict(train, **inputs)
    assert len(pub_input) == 80 and set(pub_input) == set(docs)
    for did, doc in docs.items():
        assert not ({'entities', 'association'} & pub_input[did].keys())
        assert {k: v for k, v in doc.items() if k not in ['entities', 'association']} == pub_input[did]
    baseline_rows=multi.keyed(a.baseline_prediction)
    associations=multi.keyed(a.association_prediction)
    assert set(baseline_rows)==set(associations)==set(inputs)
    assert sha(a.baseline_prediction)=='16cd8352a6e22d88350cf4d51eb0b7b255e7cbacdd32e440977550ddcf1fa89a'
    assert sha(a.association_prediction)=='aa10779059545b42369302d5888b526cd2f61e04ac36b89dc9564ab275e0c677'
    assert all(baseline_rows[did]['association']==associations[did]['association'] for did in inputs)
    public = public_rows(a.public, pub_input)
    assert sha(a.public) == previous['public_sha256'] and sha(a.public_manifest) == previous['public_manifest_sha256']
    bm = json.loads(Path(a.bioformer_manifest).read_text())
    assert bm['output_sha256'] == sha(a.bioformer) and bm['public_model_task_fit'] is False
    assert bm['model_weight_sha256'] == BIOFORMER_WEIGHT
    assert bm['input_sha256'] == public_manifest['input_sha256']
    bioformer = public_rows(a.bioformer, pub_input)
    specs = json.loads(Path(config['training_manifest']).read_text())
    assert len(specs) == 4
    xs, ys, seen, origins = [], [], set(), []
    old_count = public_only_count = bioformer_only_count = 0
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
            oldpublic = merge_pool(old, public[did])
            cs = merge_bioformer(oldpublic, bioformer[did])
            old_count += len(old)
            public_only_count += len(oldpublic)-len(old)
            bioformer_only_count += len(cs)-len(oldpublic)
            for c in cs:
                ids = truth.get((c['offset'], c['length']), set())
                ys.append(bool(ids and ('-1' in ids or set(c['identifier'].split(';')) <= ids)))
                xs.append(multi.features(doc, c, oldbase[did]) |
                          {name: c[name] for name in PUBLIC_FEATURES + BIOFORMER_FEATURES})
        origins.append({k: sha(spec[k]) for k in spec})
    assert seen == set(train) and old_count == reference['training_examples']
    assert old_count+public_only_count == previous['training_examples']
    assert origins == previous['training_origins']
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
    old_input_count = public_only_input_count = bioformer_only_input_count = 0
    for did, doc in inputs.items():
        old = multi.pool(doc, bases[did], raw[did], crfs[did])
        oldpublic = merge_pool(old, public[did])
        cs = merge_bioformer(oldpublic, bioformer[did])
        old_input_count += len(old)
        public_only_input_count += len(oldpublic)-len(old)
        bioformer_only_input_count += len(cs)-len(oldpublic)
        probabilities = model.predict_proba(vectorizer.transform([
            multi.features(doc, c, bases[did]) | {name: c[name] for name in PUBLIC_FEATURES + BIOFORMER_FEATURES}
            for c in cs]))[:, 1]
        selected = select(cs, probabilities)
        prediction.append({'pmc_id': did, 'pmid': doc.get('pmid'), 'entities': selected,
                           'association': associations[did]['association']})
        scores.append({'pmc_id': did, 'candidates': [dict(c, selector_probability=float(p))
                       for c, p in zip(cs, probabilities)]})
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists()
    (out/'baseline.jsonl').write_bytes(Path(a.baseline_prediction).read_bytes())
    with (out/'selector.pkl').open('wb') as stream:
        pickle.dump({'model':model,'vectorizer':vectorizer},stream,protocol=pickle.HIGHEST_PROTOCOL)
    with (out/'selector.pkl').open('rb') as stream:
        restored=pickle.load(stream)
    replay=restored['model'].predict_proba(restored['vectorizer'].transform([
        multi.features(doc,c,bases[did]) | {name:c[name] for name in PUBLIC_FEATURES+BIOFORMER_FEATURES}
        for c in cs]))[:,1]
    assert (replay==probabilities).all()
    for filename, rows in [('augmented.jsonl', prediction), ('selector-scores.jsonl', scores)]:
        (out/filename).write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in rows))
    manifest = dict(source_sha256=sha(__file__), arguments=vars(a),
        old_reference_sha256=sha(a.reference), public_reference_sha256=sha(a.public_reference),
        association_prediction_sha256=sha(a.association_prediction),
        model_artifact_sha256=sha(out/'selector.pkl'),model_artifact_bytes=(out/'selector.pkl').stat().st_size,
        serialized_model_last_document_probabilities_exact=True,
        runtime={'python':platform.python_version(),'sklearn':sklearn.__version__},
        legacy_source_sha256=sha(legacy.__file__), bioformer_sha256=sha(a.bioformer),
        bioformer_manifest_sha256=sha(a.bioformer_manifest), baseline_prediction_sha256=sha(out/'baseline.jsonl'),
        public_sha256=sha(a.public), public_manifest_sha256=sha(a.public_manifest),
        dependencies={module.__name__: sha(module.__file__) for module in
                      [multi, single, single.baseline, single.enhance, single.merge_semantic]},
        training_origins=origins, training_documents=len(train), old_training_examples=old_count,
        added_public_training_examples=public_only_count, added_bioformer_training_examples=bioformer_only_count, training_examples=len(ys),
        old_input_candidates=old_input_count, added_public_input_candidates=public_only_input_count, added_bioformer_input_candidates=bioformer_only_input_count,
        training_positive=sum(ys), features=vectorizer.feature_names_, estimator=model.get_params(),
        relation='Exactly preserve frozen aa107790 SFT association objects; no recomputation or entity-subset filtering.',
        outputs={name: sha(out/name) for name in ['baseline.jsonl','augmented.jsonl','selector-scores.jsonl','selector.pkl']},
        seconds=time.monotonic()-start, original_dev_read=False, evaluation_gold_read=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['training_examples','training_positive','seconds','outputs']}))


if __name__ == '__main__':
    main()
