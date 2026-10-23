#!/usr/bin/env python3
"""One fixed .1 HGB gate for cost only; training filtering is resubstitution.

Never evaluates F1 or imports an evaluator. Gate probabilities are stored for
auditing but omitted from every LLM message and token sequence.
"""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from collections import defaultdict
import importlib
import json
import pickle
from pathlib import Path
import signal
import sys
import time
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from tokenizers import Tokenizer
import census as common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('gate census >60s')))
    signal.alarm(60)
    plan = json.loads((a.output/'plan.json').read_text())
    assert plan['cut'] == .1 and plan['fit_and_census_cap_seconds'] == 60
    freeze = json.loads((a.base/'input-only-freeze.json').read_text())
    for item in freeze['artifacts']:
        assert common.sha(item['path']) == item['sha256']
    for item in plan['legacy_sources']:
        assert common.sha(item['path']) == item['sha256']
    sys.path.insert(0, plan['legacy_directory'])
    legacy = importlib.import_module('calibrate')
    assert common.sha(legacy.__file__) == plan['legacy_sources'][0]['sha256']
    base, docs = {}, {}
    for origin in freeze['origins']:
        for item in origin['source_paths'].values():
            assert common.sha(item['path']) == item['sha256']
        for row in common.read(origin['source_paths']['input']['path']):
            assert not {'entities', 'association'} & row.keys()
            docs[str(row['pmc_id'])] = row
        for row in common.read(origin['source_paths']['base']['path']):
            base[str(row['pmc_id'])] = row
    pools, keys, rows, matrices = {}, {}, {}, {}
    for split in ['train', 'heldout']:
        grouped = defaultdict(list)
        rows[split] = common.read(a.base/f'{split}.prompts.jsonl.gz')
        for row in rows[split]: grouped[row['pmc_id']].extend(row['candidates'])
        pools[split] = dict(grouped)
        keys[split], xs = [], []
        for row in rows[split]:
            for c in row['candidates']:
                keys[split].append((row['pmc_id'], c['offset'], c['length'], c['identifier']))
                xs.append(legacy.features(docs[row['pmc_id']], c, base[row['pmc_id']]))
        assert len(set(keys[split])) == len(keys[split])
        matrices[split] = xs
    assert set(pools['train']) == set(freeze['train_ids'])
    assert set(pools['heldout']) == set(freeze['heldout_ids'])
    assert not set(pools['train']) & set(pools['heldout'])
    targets = common.read(a.base/'train.targets.jsonl.gz')
    assert [r['row_id'] for r in targets] == [r['row_id'] for r in rows['train']]
    y = np.array([t for r in targets for t in r['targets']])
    assert len(y) == len(keys['train'])
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(matrices['train'])
    model = HistGradientBoostingClassifier(**plan['estimator'])
    fit_start = time.monotonic()
    model.fit(x, y)
    fit_seconds = time.monotonic() - fit_start
    predictions = {}
    for split in matrices:
        predictions[split] = model.predict_proba(vectorizer.transform(matrices[split]))[:, 1]
        assert np.isfinite(predictions[split]).all()
    blob = pickle.dumps(dict(model=model, vectorizer=vectorizer), protocol=5)
    (a.output/'gate.pkl').write_bytes(blob)
    recovered = pickle.loads(blob)
    for split in matrices:
        assert np.array_equal(predictions[split], recovered['model'].predict_proba(
            recovered['vectorizer'].transform(matrices[split]))[:, 1])
    tokenizer = Tokenizer.from_file(str(Path(plan['tokenizer_directory'])/'tokenizer.json'))
    terms, aliases = common.ontology(plan['hpo'])
    stats = {}
    for split in pools:
        lookup = dict(zip(keys[split], map(float, predictions[split])))
        retained, records, encoded = {}, [], []
        scores = []
        for pmc, cs in pools[split].items():
            chosen = []
            for c in cs:
                key = (pmc, c['offset'], c['length'], c['identifier'])
                probability = lookup[key]
                scores.append(dict(pmc_id=pmc, offset=c['offset'], length=c['length'],
                                   identifier=c['identifier'], gate_probability=probability,
                                   retained=probability >= .1))
                if probability >= .1: chosen.append(c)
            retained[pmc] = chosen
            records.extend(common.grouped(docs[pmc], chosen, terms, aliases))
        for row in records:
            # common.grouped explicitly whitelists context/cid/span/HPO, not probabilities.
            record = dict(row_id=row['row_id'], pmc_id=row['pmc_id'],
                keys=[[c['offset'], c['length'], c['identifier']] for c in row['candidates']],
                **common.encode_slots(tokenizer, row['messages'],
                    [f'Question C{i}:\n' for i in range(len(row['candidates']))]))
            assert 'gate_probability' not in json.dumps(row['messages'])
            encoded.append(record)
        stats[split] = common.summary(records, encoded)
        stats[split]['documents_with_zero_retained'] = sum(not cs for cs in retained.values())
        for kind, output in [('prompts', records), ('encoded', encoded), ('gate-scores', scores)]:
            common.gzwrite(a.output/f'{split}.{kind}.jsonl.gz', output)
    kept = predictions['train'] >= .1
    diagnosis = dict(all_candidates=int(len(y)), all_positive=int(y.sum()),
                     retained_candidates=int(kept.sum()), retained_positive=int(y[kept].sum()),
                     retained_negative=int((1-y[kept]).sum()),
                     positive_coverage=float(y[kept].sum()/y.sum()),
                     interpretation='Training resubstitution gate selection only; not OOF/generalization/quality.')
    report = dict(completed_at_utc=common.now(), source=common.ref(__file__),
        common_source=common.ref(common.__file__), plan=common.ref(a.output/'plan.json'),
        base_input_freeze=common.ref(a.base/'input-only-freeze.json'),
        train=stats['train'], heldout=stats['heldout'], training_diagnostic=diagnosis,
        model=common.ref(a.output/'gate.pkl'), model_reloaded_all_probabilities_exact=True,
        feature_count=len(vectorizer.feature_names_), feature_names=vectorizer.feature_names_,
        estimator=model.get_params(), fit_seconds=fit_seconds, seconds=time.monotonic()-start,
        sklearn=sklearn.__version__, numpy=np.__version__,
        train_selection='resubstitution; candidate lower resources exclude F1 and own document',
        heldout_annotations_read=False, originaldev_read=False, neural_forward=False,
        probability_in_LLM_input=False, input_filter_cut=.1,
        limitations=['Legacy semantic+ontology HGB features, no CRF-only/public-source candidates.',
                     'Gate trained on 32 labels; retained training examples are not OOF-selected.',
                     'No post-filter quality evaluation or threshold search.'])
    report['artifacts'] = [common.ref(p) for p in sorted(a.output.glob('*.gz'))]
    common.write(a.output/'census.json', report)
    signal.alarm(0)
    print(json.dumps({k:report[k] for k in ['fit_seconds','seconds','feature_count','train','heldout','training_diagnostic']},ensure_ascii=False))


if __name__ == '__main__':
    main()
