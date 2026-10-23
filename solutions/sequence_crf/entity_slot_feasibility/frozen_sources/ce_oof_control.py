#!/usr/bin/env python3
"""Strict nested CE feature control; frozen legacy fitting recipe is imported.

GPU phases never read evaluation labels. Candidate training origins are checked
before the three CE fits. CPU tree assembly requires complete exact-key joins.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import context_calibrate as legacy


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def write(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def gzwrite(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False) + '\n').encode())


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def key(pmc, c):
    return str(pmc), int(c['offset']), int(c['length']), c['identifier']


def example(doc, c, terms):
    import enhance
    _, passage, start, end = enhance.location(doc, c['offset'], c['offset'] + c['length'])
    assert passage['text'][start:end] == c['text']
    names = '; '.join(terms.get(h, {}).get('name', 'unlinked phenotype') for h in c['identifier'].split(';'))
    return {'pmc_id': doc['pmc_id'], 'phrase': c['text'] + ' | ' + names,
            'context': passage['text'][max(0, start - 180):start] + ' [ ' + c['text'] + ' ] ' + passage['text'][end:end + 180],
            'candidate': c}


def score_map(path):
    out = {}
    for d in legacy.zipped(path):
        for c in d['candidates']:
            k = key(d['pmc_id'], c)
            assert k not in out and math.isfinite(c['probability']) and 0 <= c['probability'] <= 1
            out[k] = c
    return out


def prepare(a):
    import calibrate_multi as multi
    base = multi.single.baseline
    out = Path(a.work)
    reference = json.loads(Path(a.reference).read_text())
    config = reference['config']
    assert reference['features'] == 84 and config['threshold'] == .4
    assert all(sha(config[k]) == digest for k, digest in reference['inputs'].items())
    assert reference['source_sha256'] == sha(multi.__file__)
    assert all(sha(getattr(multi.single, name).__file__) == value
               for name, value in [('baseline', reference['dependencies']['run']),
                                   ('enhance', reference['dependencies']['enhance']),
                                   ('merge_semantic', reference['dependencies']['merge_semantic'])])
    assert sha(multi.single.__file__) == reference['dependencies']['calibrate']
    assert sha(legacy.__file__) == 'ce843f8244b8e8a63f72c98b78ac8dd1a32b08ab22c4c01cf32a26fecfabf9c7'
    specs = json.loads(Path(config['training_manifest']).read_text())
    assert len(specs) == 3
    training = multi.keyed(config['train'])
    inputs = multi.keyed(config['input'])
    assert len(training) == 48 and len(inputs) == 16 and not training.keys() & inputs.keys()
    assert all('entities' not in d and 'association' not in d for d in inputs.values())
    terms, _, _, _ = base.load_hpo(a.hpo)
    folds = {}
    provenance = []
    tree_rows = []
    for i, spec in enumerate(specs, 1):
        docs = multi.keyed(spec['gold'])
        fold_input = multi.keyed(spec['input'])
        assert len(docs) == 16 and set(docs) == set(fold_input)
        assert all(d == training[did] for did, d in docs.items())
        assert all({k: v for k, v in d.items() if k not in ('entities', 'association')} == fold_input[did] for did, d in docs.items())
        folds[i] = {'docs': docs, 'input': fold_input, 'spec': spec}
        write(out / f'fold{i}/gold16.jsonl', docs.values())
        write(out / f'fold{i}/input16.jsonl', fold_input.values())
        Path(out / f'fold{i}/train32.jsonl').write_bytes(Path(spec['generator_train']).read_bytes())
        bases = multi.keyed(spec['base'])
        raw = multi.single.read_candidates(spec['candidates'])
        semantic_manifest = json.loads(Path(spec['candidate_manifest']).read_text())
        assert semantic_manifest['inputs']['train'] == sha(spec['generator_train'])
        assert semantic_manifest['inputs']['input'] == sha(spec['input'])
        assert semantic_manifest['output']['sha256'] == sha(spec['candidates'])
        crfs = multi.checked_crf(spec['crf'], spec['crf_manifest'], spec['generator_train'], spec['input'], set(training))
        crfs = multi.with_confidence(crfs, spec['crf_confidence'], spec['crf'])
        examples = []
        for did, doc in docs.items():
            truth = {}
            for e in doc['entities']:
                if str(e.get('note', '')).upper() != 'NO':
                    truth.setdefault((e['offset'], e['length']), set()).update(e['identifier'].split(';'))
            for c in multi.pool(doc, bases[did], raw[did], crfs[did]):
                ids = truth.get((c['offset'], c['length']), set())
                y = bool(ids and ('-1' in ids or set(c['identifier'].split(';')) <= ids))
                tree_rows.append({'key': key(did, c), 'features': multi.features(doc, c, bases[did]), 'label': y, 'text': c['text']})
                examples.append(example(fold_input[did], c, terms))
        gzwrite(out / f'fold{i}/score-input.jsonl.gz', examples)
        provenance.append({'fold': i, 'ids': sorted(map(str, docs)), 'inputs': {k: sha(spec[k]) for k in spec}, 'score_rows': len(examples)})
    assert set().union(*(set(f['docs']) for f in folds.values())) == set(training)
    assert sum(len(f['docs']) for f in folds.values()) == 48
    # Three distinct generators. Each predicts both other groups without any
    # labels from the scored group or the future CE heldout group.
    matrix = []
    for g in range(1, 4):
        path = out / f'generator{g}'
        gtrain = list(folds[g]['docs'].values())
        gin = [d for j in range(1, 4) if j != g for d in folds[j]['input'].values()]
        write(path / 'train16.jsonl', gtrain)
        write(path / 'input32.jsonl', gin)
        gazetteer = base.Gazetteer(a.hpo, gtrain)
        predictions = [base.predict(d, gazetteer) for d in gin]
        write(path / 'baseline32.jsonl', predictions)
        for j in range(1, 4):
            if j == g:
                continue
            future = ({1, 2, 3} - {g, j}).pop()
            assert not set(folds[g]['docs']) & (set(folds[j]['docs']) | set(folds[future]['docs']))
            write(out / f'pool{g}-to{j}/baseline.jsonl', [d for d in predictions if d['pmc_id'] in folds[j]['docs']])
            matrix.append({'generator_fold': g, 'scored_fold': j, 'CE_heldout_fold': future,
                           'generator_train16_sha256': sha(path / 'train16.jsonl'),
                           'scored_input16_sha256': sha(out / f'fold{j}/input16.jsonl'),
                           'future_CE_heldout_ids_intersection': [], 'generator_target_intersection': []})
    # Current outer pool and legacy matching model: no supervised fit required.
    bases, raw = multi.keyed(config['base']), multi.single.read_candidates(config['candidates'])
    crfs = multi.checked_crf(config['crf'], config['crf_manifest'], config['train'], config['input'], set(training))
    crfs = multi.with_confidence(crfs, config['crf_confidence'], config['crf'])
    outer, outer_tree = [], []
    old = score_map(a.old_scores)
    for did, doc in inputs.items():
        cs = multi.pool(doc, bases[did], raw[did], crfs[did])
        for c in cs:
            outer.append(example(doc, c, terms))
            outer_tree.append({'key': key(did, c), 'features': multi.features(doc, c, bases[did]), 'candidate': c})
    missing = [r for r in outer if key(r['pmc_id'], r['candidate']) not in old]
    assert len(outer) == 14543 and len(old) == 14511 and len(missing) == 32
    assert set(old) < {key(r['pmc_id'], r['candidate']) for r in outer}
    assert all(old[key(r['pmc_id'], r['candidate'])]['text'] == r['candidate']['text'] for r in outer if key(r['pmc_id'], r['candidate']) in old)
    # Prove the reconstructed CE text formatter matches every legacy input.
    old_inputs = legacy.zipped(a.old_examples)
    old_examples = {key(r['pmc_id'], r['candidate']): r for r in old_inputs}
    assert set(old_examples) == set(old)
    for r in outer:
        if key(r['pmc_id'], r['candidate']) in old:
            prior = old_examples[key(r['pmc_id'], r['candidate'])]
            assert r['phrase'] == prior['phrase'] and r['context'] == prior['context']
    gzwrite(out / 'outer-missing.jsonl.gz', missing)
    gzwrite(out / 'outer-input.jsonl.gz', outer)
    gzwrite(out / 'tree-train.jsonl.gz', tree_rows)
    gzwrite(out / 'tree-input.jsonl.gz', outer_tree)
    write(out / 'outer-input.jsonl', inputs.values())
    Path(out / 'old-outer-scores.jsonl.gz').write_bytes(Path(a.old_scores).read_bytes())
    assert len(tree_rows) == 45113 and sum(r['label'] for r in tree_rows) == 2845
    save(out / 'preparation.json', {'source_sha256': sha(__file__), 'legacy_source_sha256': sha(legacy.__file__),
         'reference_manifest': a.reference, 'reference_sha256': sha(a.reference), 'reference_prediction_sha256': reference['output_sha256'],
         'tree_reference_config': config, 'tree_estimator': reference['estimator'], 'hpo_sha256': sha(a.hpo),
         'folds': provenance, 'six_pool_isolation': matrix, 'tree_train_rows': len(tree_rows),
         'tree_positive': sum(r['label'] for r in tree_rows), 'outer_rows': len(outer), 'outer_missing': len(missing),
         'legacy_CE_phrase_context_all_exact': True, 'originaldev_read': False,
         'files': {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                   if p.is_file() and (p.suffix == '.jsonl' or p.suffix == '.gz')
                   and 'assembled' not in p.relative_to(out).parts}})


def examples(a):
    """Executed after semantic generators; all train32 CE labels remain local."""
    import calibrate
    root = Path(a.work)
    preparation = json.loads((root / 'preparation.json').read_text())
    for relative, digest in preparation['files'].items():
        assert sha(root / relative) == digest
    manifest_rows = {}
    for g in range(1, 4):
        origin = root / f'generator{g}/scored'
        manifest = json.loads((origin / 'manifest.json').read_text())
        assert manifest['source_sha256'] == 'dc5cc0b85922f7dc45b0f4f9fd7d8a5828a0cc234020923b31ea67a2059c143f'
        assert manifest['inputs']['train'] == sha(root / f'generator{g}/train16.jsonl')
        assert manifest['inputs']['input'] == sha(root / f'generator{g}/input32.jsonl')
        assert manifest['inputs']['hpo'] == preparation['hpo_sha256']
        assert manifest['output']['sha256'] == sha(origin / 'scored-candidates.jsonl.gz')
        raw = legacy.zipped(origin / 'scored-candidates.jsonl.gz')
        assert len(raw) == 32 and len({r['pmc_id'] for r in raw}) == 32
        for j in range(1, 4):
            if j == g:
                continue
            ids = {d['pmc_id'] for d in read(root / f'fold{j}/input16.jsonl')}
            rows = [d for d in raw if d['pmc_id'] in ids]
            assert {r['pmc_id'] for r in rows} == ids
            path = root / f'pool{g}-to{j}'
            gzwrite(path / 'scored-candidates.jsonl.gz', rows)
            split = dict(manifest, inputs=dict(manifest['inputs'], input=sha(root / f'fold{j}/input16.jsonl')),
                         parent_actual_manifest_sha256=sha(origin / 'manifest.json'),
                         parent_actual_pool_sha256=sha(origin / 'scored-candidates.jsonl.gz'),
                         operation='Exact document subset of same generator, no scores changed')
            split['output'] = {'sha256': sha(path / 'scored-candidates.jsonl.gz')}
            save(path / 'manifest.json', split)
            manifest_rows[(g, j)] = {'gold': str(root / f'fold{j}/gold16.jsonl'), 'input': str(root / f'fold{j}/input16.jsonl'),
                 'generator_train': str(root / f'generator{g}/train16.jsonl'), 'base': str(path / 'baseline.jsonl'),
                 'candidates': str(path / 'scored-candidates.jsonl.gz'), 'candidate_manifest': str(path / 'manifest.json')}
    for i in range(1, 4):
        rows = [manifest_rows[(({1, 2, 3} - {i, j}).pop(), j)] for j in range(1, 4) if j != i]
        train_ids = {d['pmc_id'] for d in read(root / f'fold{i}/train32.jsonl')}
        heldout = {d['pmc_id'] for d in read(root / f'fold{i}/input16.jsonl')}
        for r in rows:
            gen = {d['pmc_id'] for d in read(r['generator_train'])}
            target = {d['pmc_id'] for d in read(r['gold'])}
            assert gen <= train_ids and target <= train_ids and not gen & target
            assert not (gen | target) & heldout
        save(root / f'fold{i}/training-origins.json', rows)
        # Import the unchanged old preparation recipe; its temporary input uses
        # an existing lower pool solely to satisfy formatting. Replace by the
        # fixed complete current candidate input after its fit rows are frozen.
        input_origin = manifest_rows[(({1, 2, 3} - {i}).pop(), i)]
        args = SimpleNamespace(training_manifest=str(root / f'fold{i}/training-origins.json'),
            train=str(root / f'fold{i}/train32.jsonl'), input=str(root / f'fold{i}/input16.jsonl'),
            hpo=a.hpo, base=input_origin['base'], candidates=input_origin['candidates'],
            output_dir=str(root / f'fold{i}/examples'))
        legacy.prepare(args)
        target = root / f'fold{i}/examples/input.jsonl.gz'
        target.write_bytes((root / f'fold{i}/score-input.jsonl.gz').read_bytes())
        mpath = target.parent / 'preparation.manifest.json'
        m = json.loads(mpath.read_text())
        m['inference_examples'] = len(legacy.zipped(target))
        m['output_hashes']['input.jsonl.gz'] = sha(target)
        m['input_override'] = {'policy': 'All frozen current CRF+Sap candidate keys, same phrase/context formatter; no labels',
                               'preparation_sha256': sha(root / 'preparation.json')}
        save(mpath, m)


def infer(a):
    """Only the 32 absent outer keys; identical old CE tokenization/precision."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(1)
    assert sha(Path(a.model) / 'model.safetensors') == 'f05d4fd321934b19276805f46cc3bf834f8814eaa7efb5f89b82ea7018409063'
    rows = legacy.zipped(a.input)
    assert len(rows) == 32 and all('label' not in r for r in rows)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, local_files_only=True).cuda().eval()
    tokens = tok([r['phrase'] for r in rows], [r['context'] for r in rows], padding='max_length', truncation='only_second', max_length=160, return_tensors='pt')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(**{k: v.cuda() for k, v in tokens.items()}).logits
    probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
    grouped = {}
    for r, p in zip(rows, probabilities):
        grouped.setdefault(r['pmc_id'], []).append(dict(r['candidate'], probability=p, previous_probability=r['candidate']['probability']))
    gzwrite(a.output, [{'pmc_id': did, 'candidates': cs} for did, cs in grouped.items()])
    save(Path(a.output).with_suffix('.manifest.json'), {'source_sha256': sha(__file__), 'input_sha256': sha(a.input),
        'model_sha256': sha(Path(a.model) / 'model.safetensors'), 'output_sha256': sha(a.output), 'rows': 32,
        'training': False, 'precision': 'same BF16 logits/FP32 softmax/max160 only_second as legacy'})


def tree(a):
    import calibrate_multi as multi
    root = Path(a.work)
    prep = json.loads((root / 'preparation.json').read_text())
    for relative, digest in prep['files'].items():
        assert sha(root / relative) == digest
    train, inputs = legacy.zipped(root / 'tree-train.jsonl.gz'), legacy.zipped(root / 'tree-input.jsonl.gz')
    vec = multi.single.DictVectorizer(sparse=False)
    matrix = vec.fit_transform([r['features'] for r in train])
    xt = vec.transform([r['features'] for r in inputs])
    assert matrix.shape == (45113, 84) and xt.shape == (14543, 84)
    feature_names = list(vec.feature_names_)
    if a.arm == 'ce':
        import numpy as np
        train_scores = {}
        for i in range(1, 4):
            fold = root / f'fold{i}'
            fitted = json.loads((fold / 'trained/training.manifest.json').read_text())
            assert fitted['source_sha256'] == sha(legacy.__file__)
            assert fitted['input_hashes']['train_examples'] == sha(fold / 'examples/train.jsonl.gz')
            assert fitted['input_hashes']['input_examples'] == sha(fold / 'examples/input.jsonl.gz')
            assert sha(fold / 'examples/input.jsonl.gz') == sha(fold / 'score-input.jsonl.gz')
            assert all(fitted['config'][k] == v for k, v in
                       {'epochs': 3, 'batch_size': 32, 'max_length': 160, 'learning_rate': 2e-5}.items())
            assert fitted['output_sha256'] == sha(fold / 'trained/scored-candidates.jsonl.gz')
            prepared = json.loads((fold / 'examples/preparation.manifest.json').read_text())
            assert prepared['source_sha256'] == sha(legacy.__file__)
            assert prepared['inputs']['train'] == sha(fold / 'train32.jsonl')
            assert prepared['inputs']['training_manifest'] == sha(fold / 'training-origins.json')
            assert prepared['output_hashes']['train.jsonl.gz'] == fitted['input_hashes']['train_examples']
            scores = score_map(fold / 'trained/scored-candidates.jsonl.gz')
            assert not set(train_scores) & set(scores)
            train_scores.update(scores)
        outer = score_map(root / 'old-outer-scores.jsonl.gz')
        missing_manifest = json.loads((root / 'outer-missing-scores.jsonl.manifest.json').read_text())
        assert missing_manifest['source_sha256'] == sha(__file__)
        assert missing_manifest['input_sha256'] == sha(root / 'outer-missing.jsonl.gz')
        assert missing_manifest['model_sha256'] == 'f05d4fd321934b19276805f46cc3bf834f8814eaa7efb5f89b82ea7018409063'
        assert missing_manifest['output_sha256'] == sha(root / 'outer-missing-scores.jsonl.gz')
        missing = score_map(root / 'outer-missing-scores.jsonl.gz')
        assert not set(outer) & set(missing)
        outer.update(missing)
        assert set(train_scores) == {tuple(r['key']) for r in train}
        assert set(outer) == {tuple(r['key']) for r in inputs}
        assert all(train_scores[tuple(r['key'])]['text'] == r['text'] for r in train)
        assert all(outer[tuple(r['key'])]['text'] == r['candidate']['text'] for r in inputs)
        matrix = np.column_stack([matrix, [train_scores[tuple(r['key'])]['probability'] for r in train]])
        xt = np.column_stack([xt, [outer[tuple(r['key'])]['probability'] for r in inputs]])
        feature_names.append('CE_probability')
    model = multi.single.HistGradientBoostingClassifier(**prep['tree_estimator']).fit(matrix, [r['label'] for r in train])
    prob = model.predict_proba(xt)[:, 1]
    grouped = {}
    for r, p in zip(inputs, prob):
        grouped.setdefault(r['key'][0], []).append((r['candidate'], float(p)))
    prediction = []
    for doc in read(root / 'outer-input.jsonl'):
        selected = []
        for c, p in sorted(grouped[str(doc['pmc_id'])], key=lambda cp: (-cp[1], -cp[0]['length'], cp[0]['offset'])):
            if p < .4 or any(c['offset'] < e['offset'] + e['length'] and e['offset'] < c['offset'] + c['length'] for e in selected):
                continue
            selected.append({k: c[k] for k in ['identifier', 'offset', 'length', 'text']} | {'type': 'Phenotype', 'note': None})
        selected.sort(key=lambda e: e['offset'])
        prediction.append({'pmc_id': doc['pmc_id'], 'pmid': doc.get('pmid'), 'entities': selected,
                           'association': multi.single.merge_semantic.assign(doc, selected)})
    output = root / f'assembled/{a.arm}.jsonl'
    write(output, prediction)
    if a.arm == 'base':
        assert sha(output) == prep['reference_prediction_sha256']
    save(output.with_suffix('.manifest.json'), {'source_sha256': sha(__file__), 'arm': a.arm,
        'preparation_sha256': sha(root / 'preparation.json'), 'features': feature_names,
        'feature_count': matrix.shape[1], 'estimator': model.get_params(), 'threshold': .4,
        'training_rows': len(train), 'input_rows': len(inputs), 'output_sha256': sha(output),
        'reference_prediction_sha256': prep['reference_prediction_sha256'], 'gold_evaluation_read': False})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    for k in ('work', 'reference', 'hpo', 'old-scores', 'old-examples'):
        p.add_argument('--' + k, required=True)
    p = sub.add_parser('examples')
    for k in ('work', 'hpo'):
        p.add_argument('--' + k, required=True)
    p = sub.add_parser('infer')
    for k in ('model', 'input', 'output'):
        p.add_argument('--' + k, required=True)
    p = sub.add_parser('tree')
    p.add_argument('--work', required=True)
    p.add_argument('--arm', choices=['base', 'ce'], required=True)
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
