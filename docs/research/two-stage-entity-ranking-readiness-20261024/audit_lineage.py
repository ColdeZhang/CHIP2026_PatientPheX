"""Read-only hash/identity audit. No labels, feature matrix, models or scores used."""
import gzip
import hashlib
import json
from pathlib import Path
import time

R = Path('/workspace/patientphex-runs/round2-20261023')
OUT = Path(__file__).parent
sources = {}

def ref(p):
    p = Path(p)
    v = dict(path=str(p), bytes=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    sources[str(p)] = v
    return v

def read(p):
    ref(p)
    return json.loads(Path(p).read_text())

def checked(r):
    actual = ref(r['path'])
    assert actual['sha256'] == r['sha256']
    if 'bytes' in r:
        assert actual['bytes'] == r['bytes']
    return Path(r['path'])

def ids(p):
    ref(p)
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt') as f:
        v = [json.loads(line)['pmc_id'] for line in f if line.strip()]
    assert len(v) == len(set(v))
    return set(v)

def digest(p):
    return ref(p)['sha256']

def main():
    start = time.monotonic()
    fit = read(R/'research/lightgbm-selector-readiness/formal-control/fold1/fitted/manifest.json')
    assert fit['outputs']['model-vectorizer.pkl']['sha256'] == 'a465e716fedab408a027f4c3af20f2e9870f88ac3a02209e7206d2ebf6b41227'
    checked(fit['outputs']['model-vectorizer.pkl'])  # hash only, no pickle load
    prep = read(checked(fit['prepared_manifest']))
    oldprep = read(checked(fit['old_prepared_manifest']))
    oldfit = read(checked(fit['old_fit_manifest']))
    reference = read(checked(fit['reference']))
    cfg = reference['config']
    assert digest(cfg['train']) == reference['inputs']['train']
    assert digest(cfg['input']) == reference['inputs']['input']
    train = ids(cfg['train'])
    held = ids(cfg['input'])  # pure input, not F1 gold
    assert len(train) == 48 and len(held) == 16 and not train & held
    assert digest(cfg['training_manifest']) == reference['inputs']['training_manifest']
    specs = read(cfg['training_manifest'])
    nested = []
    seen = set()
    for i, spec in enumerate(specs):
        origin = {k: digest(p) for k,p in spec.items()}
        assert origin == fit['training_origins'][i] == oldfit['training_origins'][i]
        gen = ids(spec['generator_train'])
        target = ids(spec['input'])
        assert len(gen) == 32 and len(target) == 16
        assert gen == train - target and target <= train
        assert not (gen | target) & held and not seen & target
        seen |= target
        assert target == ids(spec['base']) == ids(spec['crf']) == ids(spec['candidates'])
        sem = read(spec['candidate_manifest'])
        crf = read(spec['crf_manifest'])
        base = read(Path(spec['base']).parent/'baseline-work/manifest.json')
        assert sem['inputs']['train'] == base['inputs']['train']['sha256'] == crf['train_sha256'] == origin['generator_train']
        assert sem['inputs']['input'] == base['inputs']['input']['sha256'] == crf['input_sha256'] == origin['input']
        assert sem['output']['sha256'] == origin['candidates']
        assert base['output']['sha256'] == origin['base']
        assert crf['output_sha256'] == origin['crf']
        assert sem['input_annotations_used'] is False and crf['input_annotations_used'] is False
        assert sem['model_manifest']['model'] == 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
        assert sem['model_manifest']['derived_weights']['sha256'] == 'dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814'
        nested.append(dict(origin_index=i, generator_documents=len(gen), target_documents=len(target),
                           generator_train=ref(spec['generator_train']),
                           target_input=ref(spec['input']), sources_sha256=origin,
                           held_F1_intersection=0, target_generator_intersection=0,
                           baseline_manifest=ref(Path(spec['base']).parent/'baseline-work/manifest.json')))
    assert seen == train
    pf = prep['folds'][1]
    assert train == set(pf['train_ids']) and held == set(pf['input_ids'])
    assert train == ids(checked(pf['train']))
    assert pf['new_train_rows'] == fit['training_examples'] == 45403
    assert fit['actual_feature_count'] == 101 and fit['estimator']['objective'] == 'binary'
    for role in ['public', 'bioformer']:
        p = oldfit['arguments'][role+'_manifest']
        m = read(p)
        assert digest(p) == oldfit[role+'_manifest_sha256']
        assert m['public_model_task_fit'] is False and m['annotations_read'] is False
        assert digest(oldfit['arguments'][role]) == m['output_sha256'] == oldfit[role+'_sha256']
        assert ids(m['input_path']) == train | held
        assert digest(m['input_path']) == m['input_sha256']
    bm = read(checked(prep['cache']['manifest']))
    assert bm['public_model_training_on_task'] is False
    assert bm['model_weight_sha256'] == '1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573'
    for k in ['canonical', 'provenance']:
        checked(prep['cache'][k])
    source_dir = Path('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules')
    for name, h in fit['dependencies'].items():
        assert digest(source_dir/(name+'.py')) == h
    labels_source = source_dir/'public_phenotagger_biobert_selector/dependencies/fit_helpers.py'
    assert digest(labels_source) == fit['label_function_sha256']
    fit_source = Path('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/lightgbm_selector_control/fit_fold.py')
    assert digest(fit_source) == fit['source_sha256']
    result = dict(status='PASS_hash_and_document_lineage_no_new_census',
        source_sha256=digest(__file__), outer_train_documents=48, held_input_documents=16,
        nested=nested, train_pool=pf['train'], train_gold=ref(cfg['train']),
        candidate_rows_manifest_only=fit['training_examples'], positive_rows_manifest_only=fit['training_positive'],
        actual_features=101, objective='binary',
        H40_scope='All three task-trained baseline/SapBERT-detector/CRF sources match nested train32 excluding both held F1 and own target16. Public sources are frozen no-task-fit manifest evidence, not independent reproduction of upstream pretraining.',
        limitations=['No upstream training rerun or historical execution attestation.',
          'Train JSON objects decoded for document IDs; annotation values not used.',
          'No gold-label grouping/counts, feature computation, model load, prediction or quality evaluation.',
          'No original dev gold or held-F1 gold opened.'],
        seconds=time.monotonic()-start)
    with (OUT/'lineage-result.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    with (OUT/'lineage-sources.json').open('x') as f: json.dump(list(sources.values()),f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],sources=len(sources))))

if __name__ == '__main__':
    main()
