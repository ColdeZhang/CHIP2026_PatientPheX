"""Load the frozen LightGBM selector and replace entities in a prediction template.

Starts from an existing pure-input candidate cache. It does not build candidates,
train any model, read labels, or evaluate quality. All paths are CLI parameters.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time

FEATURES = ['biobert_'+x for x in ('present','score','neural','dictionary','abbreviation','origin_unknown')]
OLD_FEATURES = [p+x for p in ('public_','bioformer_') for x in ('present','score','neural','dictionary','abbreviation','origin_unknown')]


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check(path, expected):
    actual = ref(path)
    assert all(actual[k] == expected[k] for k in ('bytes', 'sha256')), str(path)
    return actual


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def feature(multi, row, candidate):
    return multi.features(row['input'], candidate, row['base']) | {
        name: candidate[name] for name in OLD_FEATURES + FEATURES}


def field_span(line, wanted='entities'):
    """Original literal replacement helper; only a top-level value can change."""
    decoder = json.JSONDecoder()
    pos = 0
    def whitespace(p):
        while p < len(line) and line[p].isspace():
            p += 1
        return p
    pos = whitespace(pos)
    assert line[pos] == '{'
    pos += 1
    names, found = set(), None
    while True:
        pos = whitespace(pos)
        if line[pos] == '}':
            break
        name, pos = decoder.raw_decode(line, pos)
        assert isinstance(name, str) and name not in names
        names.add(name)
        pos = whitespace(pos)
        assert line[pos] == ':'
        pos = whitespace(pos + 1)
        start = pos
        _, pos = decoder.raw_decode(line, pos)
        if name == wanted:
            found = start, pos
        pos = whitespace(pos)
        if line[pos] == '}':
            break
        assert line[pos] == ','
        pos += 1
    assert not line[pos+1:].strip() and found is not None
    return found


def replace_entities(line, entities):
    old = json.loads(line)
    if old['entities'] == entities:
        return line
    a, b = field_span(line)
    result = line[:a] + json.dumps(entities, ensure_ascii=False) + line[b:]
    x, y = field_span(result)
    assert (line[:a].encode(), line[b:].encode()) == (result[:x].encode(), result[y:].encode())
    assert {k:v for k,v in old.items() if k != 'entities'} == {k:v for k,v in json.loads(result).items() if k != 'entities'}
    return result


def execute(args):
    start = time.monotonic()
    signal.alarm(30)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    contract = json.loads(args.contract.read_text())
    assert contract['cut'] == .3 and contract['columns'] == 103
    assert contract['CPU_cap_seconds'] == 30
    check(__file__, contract['entry_source'])
    bound = {}
    for name in ('model', 'input_pool', 'input', 'relation_template', 'hpo'):
        bound[name] = check(getattr(args, name), contract['inputs'][name])
    dependency_root = args.dependency_root.resolve()
    dependency_refs = []
    for source in contract['dependency_sources']:
        dependency_refs.append(check(dependency_root / source['relative_path'], source))
    runtime_refs = [check(args.runtime_root / source['relative_path'], source)
                    for source in contract['runtime_sources']]
    sys.path.insert(0, str(args.runtime_root.resolve()))
    import lightgbm
    import numpy as np
    assert lightgbm.__version__ == contract['runtime_version'] == '4.6.0'
    ontology_dir = dependency_root / 'solutions/ontology_rules'
    sequence_dir = dependency_root / 'solutions/sequence_crf'
    sys.path.insert(0, str(ontology_dir))
    multi = importlib.import_module('calibrate_multi')
    ontology_run = multi.single.baseline
    assert Path(ontology_run.__file__).resolve() == ontology_dir / 'run.py'
    with args.model.open('rb') as stream:
        bundle = pickle.load(stream)
    model, vectorizer = bundle['model'], bundle['vectorizer']
    assert isinstance(model, lightgbm.LGBMClassifier)
    assert bundle['features'] == vectorizer.feature_names_ == contract['features']
    assert vectorizer.sparse is False and len(vectorizer.vocabulary_) == 103
    assert all(vectorizer.vocabulary_[name] == i for i, name in enumerate(contract['features']))
    assert np.array_equal(model.classes_, [0, 1]) and model.n_features_in_ == 103
    assert model.n_iter_ == model.booster_.current_iteration() == 160
    inputs = rows(args.input)
    pools = rows(args.input_pool)
    byid = {row['pmc_id']: row for row in inputs}
    assert len(inputs) == len(byid) == len(pools) == contract['documents']
    assert all(set(row) == {'pmc_id', 'pmid', 'patient', 'full_text'} for row in inputs)
    assert len({row['pmc_id'] for row in pools}) == len(pools)
    assert set(byid) == {row['pmc_id'] for row in pools}
    scores = {}
    candidate_count = 0
    for row in pools:
        assert row['input'] == byid[row['pmc_id']]
        candidates = row['candidates']
        matrix = vectorizer.transform([feature(multi, row, c) for c in candidates])
        assert matrix.shape == (len(candidates), 103)
        probabilities = model.predict_proba(matrix)
        assert probabilities.shape == (len(candidates), 2) and np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        scores[row['pmc_id']] = [dict(c, selector_probability=float(p))
                                for c, p in zip(candidates, probabilities[:, 1])]
        candidate_count += len(candidates)
    assert candidate_count == contract['candidate_count']
    # The feature modules retain their ontology run object. The decoder needs
    # the distinct sequence module under the same historical import name.
    spec = importlib.util.spec_from_file_location('run', sequence_dir / 'run.py')
    sequence_run = importlib.util.module_from_spec(spec)
    sys.modules['run'] = sequence_run
    spec.loader.exec_module(sequence_run)
    sys.path.insert(0, str(sequence_dir))
    decoder = importlib.import_module('decode_nested_selector')
    assert decoder.prior.base is sequence_run and multi.single.baseline is ontology_run
    _, allowed, _, _ = sequence_run.parse_hpo(args.hpo)
    allowed.discard('HP:0000118')
    template = args.relation_template.read_bytes().decode('utf-8').splitlines(keepends=True)
    template_rows = [json.loads(line) for line in template]
    assert len(template_rows) == len(byid) and {x['pmc_id'] for x in template_rows} == set(byid)
    assert sum(len(a['phenotype']) for x in template_rows for a in x['association']) == contract['preserved_relations']
    result_lines = []
    for line, before in zip(template, template_rows):
        did = before['pmc_id']
        candidates = scores[did]
        entities, _ = decoder.decode(candidates, decoder.greedy(candidates, .3), byid[did], allowed, .3)
        result_lines.append(replace_entities(line, entities))
    payload = ''.join(result_lines).encode('utf-8')
    actual_sha = hashlib.sha256(payload).hexdigest()
    assert actual_sha == contract['expected_prediction_sha256']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction = args.output_dir / 'prediction.jsonl'
    prediction.write_bytes(payload)
    elapsed = time.monotonic() - start
    assert elapsed < 30
    report = dict(status='pure_load_byteexact_reproduction_pass',
                  finished_utc=datetime.now(timezone.utc).isoformat(), seconds=elapsed,
                  source=ref(__file__), contract=ref(args.contract), inputs=bound,
                  dependency_sources=dependency_refs, runtime_sources=runtime_refs,
                  model_class=type(model).__name__, actual_classes=model.classes_.tolist(),
                  features=vectorizer.feature_names_, model_iterations=int(model.n_iter_),
                  documents=len(inputs), candidate_count=candidate_count,
                  prediction=ref(prediction), expected_prediction_sha256=contract['expected_prediction_sha256'],
                  preserved_relations=contract['preserved_relations'], all_nonentity_literal_bytes_preserved=True,
                  fit_calls=0, gold_reads=0, quality_calls=0, GPU_calls=0,
                  scope='Starts from frozen pure-input candidates and an existing relation template; does not reproduce upstream candidate or relation generation.')
    with (args.output_dir / 'reproduction.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    signal.alarm(0)
    print(json.dumps({k:report[k] for k in ('status', 'seconds', 'documents', 'candidate_count', 'prediction')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'model', 'input-pool', 'input', 'relation-template', 'hpo',
                 'dependency-root', 'runtime-root', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    execute(parser.parse_args())
