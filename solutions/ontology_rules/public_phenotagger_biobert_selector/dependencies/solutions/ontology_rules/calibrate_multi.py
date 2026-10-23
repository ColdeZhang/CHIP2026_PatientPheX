#!/usr/bin/env python3
"""Document-isolated tree calibration of ontology, SapBERT and CRF spans.

CRF is an explicitly attributed additional candidate source. Each training
document must be excluded from both its semantic and CRF generators.
"""
import argparse
import json
import time
from pathlib import Path
import calibrate as single


def keyed(path):
    return {d['pmc_id']: d for d in single.baseline.read_jsonl(path)}


def checked_crf(prediction, manifest, generator_train, input_path, allowed_train):
    sha = single.baseline.sha256
    m = json.loads(Path(manifest).read_text())
    assert m['output_sha256'] == sha(prediction)
    assert m['train_sha256'] == sha(generator_train)
    assert m['input_sha256'] == sha(input_path)
    generator_ids, input_ids = set(keyed(generator_train)), set(keyed(input_path))
    assert generator_ids <= allowed_train and not generator_ids & input_ids
    pred = keyed(prediction)
    assert set(pred) == input_ids
    return pred


def with_confidence(predictions, path, prediction_path):
    if not path:
        return predictions
    m = json.loads(Path(path).with_suffix('.manifest.json').read_text())
    assert m['inputs']['prediction'] == single.baseline.sha256(prediction_path)
    assert m['output_sha256'] == single.baseline.sha256(path)
    key = lambda e: (e['offset'], e['length'], e['identifier'])
    scores = {r['pmc_id']: {key(c): c for c in r['candidates']} for r in single.baseline.read_jsonl(path)}
    assert set(scores) == set(predictions)
    for did, pred in predictions.items():
        assert set(scores[did]) == {key(e) for e in pred['entities']}
        pred['entities'] = [dict(e, crf_ner=scores[did][key(e)]['ner_probability'],
                                crf_link=scores[did][key(e)]['link_similarity']) for e in pred['entities']]
    return predictions


def pool(doc, base, raw, crf):
    key = lambda c: (c['offset'], c['length'], c['identifier'])
    cs = {key(c): dict(c, crf=False, semantic_source=True)
          for c in single.pool(doc, base, raw)}
    spans = {(e['offset'], e['length']) for e in crf['entities']}
    for e in crf['entities']:
        if key(e) in cs:
            cs[key(e)]['crf'] = True
            cs[key(e)].update({k: e[k] for k in ['crf_ner', 'crf_link'] if k in e})
        else:
            cs[key(e)] = dict(e, probability=.5, similarity=0., exact=False,
                             is_base=False, crf=True, semantic_source=False)
    for c in cs.values():
        c['crf_span'] = (c['offset'], c['length']) in spans
    return list(cs.values())


def features(doc, c, base):
    return single.features(doc, c, base) | {k: c[k] for k in
                                          ['crf', 'crf_span', 'semantic_source']} | {
                                              k: c.get(k, 0.) for k in ['crf_ner', 'crf_link']}


def main():
    ap = argparse.ArgumentParser()
    for name in ['training-manifest', 'train', 'input', 'base', 'candidates',
                 'crf', 'crf-manifest', 'output']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--threshold', type=float, default=.4)
    ap.add_argument('--without-crf', action='store_true', help='Matched label-union control without the additional source')
    ap.add_argument('--crf-confidence')
    a = ap.parse_args()
    started = time.monotonic()
    sha = single.baseline.sha256
    train, inputs = keyed(a.train), keyed(a.input)
    assert not set(train) & set(inputs)
    assert all('entities' not in d and 'association' not in d for d in inputs.values())
    specs = json.loads(Path(a.training_manifest).read_text())
    xs, ys, seen, sources = [], [], set(), []
    for spec in specs:
        docs, bases = keyed(spec['gold']), keyed(spec['base'])
        raw = single.read_candidates(spec['candidates'])
        m = json.loads(Path(spec['candidate_manifest']).read_text())
        assert m['inputs']['input'] == sha(spec['input'])
        assert m['inputs']['train'] == sha(spec['generator_train'])
        generator_ids = set(keyed(spec['generator_train']))
        assert generator_ids <= set(train) and not generator_ids & set(docs)
        assert not seen & set(docs)
        seen |= set(docs)
        crfs = checked_crf(spec['crf'], spec['crf_manifest'],
                           spec['generator_train'], spec['input'], set(train))
        crfs = with_confidence(crfs, spec.get('crf_confidence'), spec['crf'])
        for did, doc in docs.items():
            assert doc == train[did]
            truth = {}
            for e in doc['entities']:
                if str(e.get('note', '')).upper() != 'NO':
                    truth.setdefault((e['offset'], e['length']), set()).update(e['identifier'].split(';'))
            for c in pool(doc, bases[did], raw[did], {'entities': []} if a.without_crf else crfs[did]):
                ids = truth.get((c['offset'], c['length']), set())
                ys.append(bool(ids and ('-1' in ids or set(c['identifier'].split(';')) <= ids)))
                xs.append(features(doc, c, bases[did]))
        sources.append({k: sha(spec[k]) for k in ['gold', 'base', 'candidates', 'crf', 'crf_manifest', 'generator_train']})
        if spec.get('crf_confidence'):
            sources[-1]['crf_confidence'] = sha(spec['crf_confidence'])
    assert seen == set(train)
    vec = single.DictVectorizer(sparse=False)
    matrix = vec.fit_transform(xs)
    model = single.HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=5., learning_rate=.06,
        random_state=20261023, early_stopping=False).fit(matrix, ys)
    bases, raw = keyed(a.base), single.read_candidates(a.candidates)
    crfs = checked_crf(a.crf, a.crf_manifest, a.train, a.input, set(train))
    crfs = with_confidence(crfs, a.crf_confidence, a.crf)
    prediction, candidate_count = [], 0
    for did, doc in inputs.items():
        cs = pool(doc, bases[did], raw[did], {'entities': []} if a.without_crf else crfs[did]); candidate_count += len(cs)
        probabilities = model.predict_proba(vec.transform([features(doc, c, bases[did]) for c in cs]))[:, 1]
        selected = []
        for c, p in sorted(zip(cs, probabilities), key=lambda cp: (-cp[1], -cp[0]['length'], cp[0]['offset'])):
            if p < a.threshold or any(c['offset'] < e['offset'] + e['length'] and e['offset'] < c['offset'] + c['length'] for e in selected):
                continue
            selected.append({k: c[k] for k in ['identifier', 'offset', 'length', 'text']} | {'type': 'Phenotype', 'note': None})
        selected.sort(key=lambda e: e['offset'])
        prediction.append({'pmc_id': did, 'pmid': doc.get('pmid'), 'entities': selected,
                           'association': single.merge_semantic.assign(doc, selected)})
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in prediction))
    manifest = {'config': vars(a), 'source_sha256': sha(__file__),
        'dependencies': {m.__name__: sha(m.__file__) for m in [single, single.baseline, single.enhance, single.merge_semantic]},
        'inputs': {k: sha(getattr(a, k)) for k in ['training_manifest', 'train', 'input', 'base', 'candidates', 'crf', 'crf_manifest']},
        'training_sources': sources, 'training_examples': len(ys), 'training_positive': sum(ys),
        'training_documents': len(train), 'input_candidates': candidate_count,
        'features': len(vec.feature_names_), 'estimator': model.get_params(),
        'selection': 'fixed threshold .4; no original dev used',
        'output_sha256': sha(out), 'seconds': time.monotonic() - started}
    if a.crf_confidence:
        manifest['inputs']['crf_confidence'] = sha(a.crf_confidence)
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
