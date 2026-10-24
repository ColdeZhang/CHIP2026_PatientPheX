"""Fixed F1 document-grouped meta OOF development screen, never F0/GPU.

No model fit in prepare. Eight independent 12-document fits in fit; both full
OOF predictions freeze before evaluate reads the already-studied F1 gold file.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import warnings

BASE = ['old9B_selected', 'reverse_same_adapter_selected',
        'binary32_probability', 'oldLR_probability_max_occurrence',
        'rules_selected_mean_occurrence']
CONTEXT = ['pair_same_sentence_max', 'pair_log_dist_min',
           'pair_closer_margin_max', 'negated_for_target_max',
           'explicit_cohort_max', 'pair_patients_max']
R1 = ['r1_observed', 'r1_selected', 'r1_selected_patient_fraction']
FEATURES = {'without_r1': BASE + CONTEXT, 'with_r1': BASE + CONTEXT + R1}
PARAMETERS = dict(boosting_type='gbdt', objective='binary', n_estimators=100,
                  max_depth=3, num_leaves=7, min_child_samples=35,
                  learning_rate=.05, reg_lambda=10., reg_alpha=0.,
                  max_bin=255, min_data_in_bin=3, min_child_weight=.001,
                  min_split_gain=0., boost_from_average=True,
                  subsample_for_bin=200000, subsample=1., subsample_freq=0,
                  colsample_bytree=1., feature_fraction_bynode=1.,
                  class_weight=None, is_unbalance=False, scale_pos_weight=1.,
                  use_missing=True, zero_as_missing=False,
                  feature_pre_filter=True, enable_bundle=True,
                  random_state=20261024, data_random_seed=20261024,
                  feature_fraction_seed=20261024, bagging_seed=20261024,
                  deterministic=True, force_col_wise=True, n_jobs=1,
                  device_type='cpu', verbosity=-1)
SEED = 20261024


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(reference):
    path = Path(reference['path'])
    require(path.stat().st_size == reference['bytes'] and
            sha(path) == reference['sha256'], 'Changed artifact: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_rows(path, values):
    with Path(path).open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def key(row):
    result = tuple(row[k] for k in ('pmc_id', 'patient_id', 'concept'))
    require(all(type(x) is str for x in result), 'Nonliteral key')
    return result


def keyed(values):
    result = {key(row): row for row in values}
    require(len(result) == len(values), 'Duplicate pair key')
    return result


def split_documents(documents):
    ordered = sorted(documents, key=lambda d: (hashlib.sha256(
        f'{SEED}:{d}'.encode('utf-8')).hexdigest(), d))
    require(len(ordered) == len(set(ordered)) == 16, 'Need exactly 16 documents')
    return [ordered[i::4] for i in range(4)]


def numeric(value, unit=False):
    require(type(value) in (bool, float, int) and math.isfinite(value),
            'Invalid numeric feature')
    require(not unit or 0 <= value <= 1, 'Feature outside [0,1]')
    return float(value)


def r1_vector(mapping, cid, pid, patient_ids, valid):
    observed = valid and cid in mapping
    selected = mapping[cid] if observed else []
    require(len(selected) == len(set(selected)) and set(selected) <= set(patient_ids),
            'Invalid normalized patient identity')
    return [float(observed), float(pid in selected), len(selected) / len(patient_ids)]


def selected_pairs(documents):
    result = set()
    for doc in documents:
        for association in doc['association']:
            for concept in association['phenotype']:
                result.add((doc['pmc_id'], association['patient_id'], concept))
    return result


def association_span(line):
    """Locate one top-level association value; preserve every other literal byte."""
    decoder, position, names, found = json.JSONDecoder(), 0, set(), None
    def whitespace(index):
        while index < len(line) and line[index].isspace():
            index += 1
        return index
    position = whitespace(position)
    require(line[position:position + 1] == '{', 'Require JSON object line')
    position += 1
    while True:
        position = whitespace(position)
        if line[position:position + 1] == '}':
            position += 1
            break
        name, position = decoder.raw_decode(line, position)
        require(isinstance(name, str) and name not in names, 'Duplicate/nonstring key')
        names.add(name)
        position = whitespace(position)
        require(line[position:position + 1] == ':', 'Missing object colon')
        begin = whitespace(position + 1)
        value, end = decoder.raw_decode(line, begin)
        if name == 'association':
            require(isinstance(value, list), 'Association must be a list')
            found = (begin, end)
        position = whitespace(end)
        if line[position:position + 1] == ',':
            position += 1
            continue
        require(line[position:position + 1] == '}', 'Invalid object separator')
        position += 1
        break
    require(not line[position:].strip() and found is not None, 'Missing/trailing JSON')
    return found


def nonrelation_bytes(lines):
    result = []
    for line in lines:
        begin, end = association_span(line)
        result.append((line[:begin] + '[]' + line[end:]).encode('utf-8'))
    return b''.join(result)


def load_plan(path):
    plan = read(path)
    require(plan['execution_ready'] and plan['features'] == FEATURES and
            plan['parameters'] == PARAMETERS and plan['threshold'] == .5,
            'Fixed method changed')
    require(plan['source'] == ref(__file__), 'Source changed')
    # Source contract is checked before opening any feature/target bytes.
    lineage = read(checked(plan['refs']['source_inventory']))
    require(lineage['status'] == 'PASS_ALL_BASE_SOURCES_EXCLUDE_ENTIRE_F1',
            'Base source isolation not established')
    for reference in plan['refs'].values():
        checked(reference)
    return plan, ref(path), Path(plan['output_dir'])


def prepare(plan, planref, directory):
    refs = plan['refs']
    prep = read(refs['base_preparation']['path'])
    role = next(r for r in prep['roles'] if r['role'] == 'meta_F1')
    F1, base_train = set(role['input_ids']), set(role['train_ids'])
    require(len(F1) == 16 and len(base_train) == 32 and not F1 & base_train,
            'Entire F1 not excluded from base training')
    require(role['output_rows'] == refs['F1_continuous'] and role['positive_labels'] == 365,
            'Frozen labels/source changed')
    folds = split_documents(F1)
    require(folds == plan['heldout_document_groups'], 'Frozen document split differs')
    old, reverse = rows(refs['F1_old']['path']), rows(refs['F1_reverse']['path'])
    require([d['pmc_id'] for d in old] == [d['pmc_id'] for d in reverse] and
            {d['pmc_id'] for d in old} == F1, 'Baseline order/coverage differs')
    require(all({k:v for k,v in a.items() if k != 'association'} ==
                {k:v for k,v in b.items() if k != 'association'}
                for a,b in zip(old,reverse)), 'Reverse changed fixed entities')
    require(sum(len(d['entities']) for d in old) == 1189, 'Entity template changed')
    prompts = rows(refs['F1_prompts']['path'])
    require(len(prompts) == 80 and all(set(p) == {'row_id','pmc_id','code_map',
            'patient_ids','messages'} for p in prompts), 'Prompt schema/count changed')
    byrow = {p['row_id']: p for p in prompts}
    require(len(byrow) == 80, 'Duplicate prompt')
    ordered = [dict(row_id=p['row_id'], pmc_id=p['pmc_id'], CID=cid,
                    patient_id=pid, concept=concept)
               for p in prompts for cid,concept in p['code_map'].items()
               for pid in p['patient_ids']]
    universe = set(key(row) for row in ordered)
    require(len(ordered) == len(universe) == 1557, 'Fixed pair universe differs')
    oldset, revset = selected_pairs(old), selected_pairs(reverse)
    require(oldset <= universe and revset <= universe, 'Old selection outside pool')
    binary = {}
    probability_rows = rows(refs['F1_binary']['path'])
    require([r['row_id'] for r in probability_rows] == list(byrow), 'Binary row order')
    for row in probability_rows:
        p = byrow[row['row_id']]
        require(row['pmc_id'] == p['pmc_id'] and len(row['keys']) == len(row['probabilities']),
                'Binary source identity/length differs')
        for (cid,pid), probability in zip(row['keys'],row['probabilities']):
            require(cid in p['code_map'] and pid in p['patient_ids'], 'Binary key outside pool')
            k = (p['pmc_id'],pid,p['code_map'][cid])
            require(k not in binary, 'Duplicate binary key')
            binary[k] = numeric(probability, True)
    with gzip.open(refs['R1_raw']['path'], 'rt') as stream:
        raw = [json.loads(line) for line in stream]
    metadata = read(refs['R1_result']['path'])
    require([r['row_id'] for r in raw] == list(byrow), 'R1 row order differs')
    r1facts, coverage = {}, Counter()
    for row, receipt in zip(raw, metadata['rows']):
        canonical = json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        require(hashlib.sha256(canonical).hexdigest() == receipt['canonical_item_sha256'],
                'R1 saved row canonical receipt mismatch')
        p = byrow[row['row_id']]
        valid = row['normalized_valid']
        require(type(valid) is bool, 'R1 validity not boolean')
        mapping = row['normalized_evidence']['provided_mapping'] if valid else {}
        require(set(mapping) <= set(p['code_map']), 'R1 CID outside pool')
        for cid,concept in p['code_map'].items():
            for pid in p['patient_ids']:
                k = (p['pmc_id'],pid,concept)
                require(k not in r1facts, 'Duplicate R1 pair')
                r1facts[k] = r1_vector(mapping,cid,pid,p['patient_ids'],valid)
                state = ('invalid_unknown' if not valid else 'missing_unknown' if cid not in mapping
                         else 'selected' if pid in mapping[cid] else 'observed_not_selected')
                coverage[state] += 1
    require(coverage == dict(invalid_unknown=232, missing_unknown=50,
            selected=453, observed_not_selected=822), 'R1 fixed coverage differs')
    # Explicitly authorized F1 target table: only original target and whitelisted fields.
    continuous = keyed(rows(refs['F1_continuous']['path']))
    context = keyed(rows(refs['F1_rule']['path']))
    require(set(binary) == set(r1facts) == set(continuous) == set(context) == universe,
            'Feature source keys differ')
    require(all('target' not in row for row in context.values()), 'Context cache has targets')
    output = []
    for row in ordered:
        k = key(row)
        c, pure = continuous[k], context[k]
        require(c['row_id'] == row['row_id'] and c['CID'] == row['CID'], 'Target row identity differs')
        features = [float(k in oldset),float(k in revset),binary[k],
                    numeric(c['features']['relation_lr_probability'],True),
                    numeric(pure['features']['rule_selected_mean'],True)]
        require(pure['features']['rule_selected_mean'] == c['features']['rule_selected_mean'],
                'Rule feature changed between copies')
        for name in CONTEXT:
            require(pure['features'][name] == c['features'][name], 'Context copy differs: '+name)
            features.append(numeric(pure['features'][name]))
        require(type(c['target']) is int and c['target'] in (0,1), 'Original target not binary')
        output.append(dict(row, x=features + r1facts[k], y=c['target'],
                           heldout_fold=next(i for i,g in enumerate(folds) if row['pmc_id'] in g)))
    require(sum(r['y'] for r in output) == 365, 'Original literal label count changed')
    write_rows(directory/'F1.matrix.jsonl',output)
    save(directory/'preparation.json',dict(status='COMPLETE_FIXED_F1_FEATURES_AND_LABELS_NO_FIT',
        source=ref(__file__),plan=planref,matrix=ref(directory/'F1.matrix.jsonl'),
        heldout_document_groups=folds,features=FEATURES,rows=1557,positive_labels=365,
        R1_coverage=dict(coverage),base_train32=sorted(base_train),all_F1=sorted(F1),
        no_base_model_fit_on_entire_F1=True,no_new_gold_file_read=True,
        F1_mixed_training_target_table_read=True,no_F0_read=True,
        disclosure='Known F1 labels are meta-training targets. This is development OOF, not a new blind test.'))


def fit(plan, planref, directory):
    started = time.monotonic()
    require(not (directory/'fit-started.json').exists(), 'Unique eight-fit run already reserved')
    save(directory/'fit-started.json',dict(at_utc=now(),plan=planref,fit_calls_reserved=8,
                                          cpu_cap_seconds=60,no_retries=True))
    signal.signal(signal.SIGALRM, lambda signum,frame: (_ for _ in ()).throw(TimeoutError('60s fit budget')))
    signal.setitimer(signal.ITIMER_REAL,60)
    import numpy as np
    import lightgbm as lgb
    require(lgb.__version__ == '4.6.0', 'Runtime changed')
    prep = read(directory/'preparation.json')
    require(prep['plan'] == planref, 'Preparation changed')
    data = rows(checked(prep['matrix']))
    X = np.asarray([row['x'] for row in data],dtype=np.float64)
    y = np.asarray([row['y'] for row in data],dtype=np.int64)
    ids = np.asarray([row['pmc_id'] for row in data])
    fold_ids = np.asarray([row['heldout_fold'] for row in data])
    require(X.shape == (1557,14) and np.isfinite(X).all() and y.sum() == 365,
            'Fixed matrix shape/finite/label contract')
    probabilities = {arm:np.full(1557,np.nan) for arm in FEATURES}
    fitted, per_fold = [], []
    for fold, held_ids in enumerate(plan['heldout_document_groups']):
        train_mask, held_mask = fold_ids != fold, fold_ids == fold
        train_documents = set(ids[train_mask].tolist())
        held_documents = set(ids[held_mask].tolist())
        require(held_documents == set(held_ids) and len(train_documents) == 12 and
                not train_documents & held_documents and set(y[train_mask].tolist()) == {0,1},
                'Meta fold leaks documents or lacks two classes')
        per_fold.append(dict(fold=fold,train_ids=sorted(train_documents),held_ids=held_ids,
            train_rows=int(train_mask.sum()),held_rows=int(held_mask.sum()),
            train_positive_labels=int(y[train_mask].sum()),held_positive_labels=int(y[held_mask].sum())))
        for arm, columns in FEATURES.items():
            width = len(columns)
            model = lgb.LGBMClassifier(**PARAMETERS)
            tick = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(X[train_mask,:width],y[train_mask])
                values = model.predict_proba(X[held_mask,:width])[:,1]
                native = model.booster_.predict(X[held_mask,:width],num_threads=1)
            require(model.n_iter_ == model.booster_.current_iteration() == 100 and
                    model.n_features_in_ == width and model.classes_.tolist() == [0,1],
                    'Actual estimator differs')
            require(np.array_equal(native,values) and np.isfinite(values).all() and
                    ((values >= 0)&(values <= 1)).all(), 'Wrong positive probability contract')
            path = directory/f'fold{fold}.{arm}.pkl'
            with path.open('xb') as stream:
                pickle.dump(model,stream,protocol=5)
                stream.flush();os.fsync(stream.fileno())
            with path.open('rb') as stream:
                loaded = pickle.load(stream)
            with warnings.catch_warnings(record=True) as reload_warnings:
                warnings.simplefilter('always')
                recovered = loaded.predict_proba(X[held_mask,:width])[:,1]
            require(np.array_equal(values,recovered), 'Actual saved-model replay differs')
            require(np.isnan(probabilities[arm][held_mask]).all(), 'OOF row predicted twice')
            probabilities[arm][held_mask] = recovered
            fitted.append(dict(fold=fold,arm=arm,columns=columns,model=ref(path),
                fit_train_document_ids=sorted(train_documents),predict_held_document_ids=held_ids,
                train_matrix_sha256=hashlib.sha256(X[train_mask,:width].tobytes()).hexdigest(),
                train_target_sha256=hashlib.sha256(y[train_mask].tobytes()).hexdigest(),
                held_matrix_sha256=hashlib.sha256(X[held_mask,:width].tobytes()).hexdigest(),
                held_probability_sha256=hashlib.sha256(recovered.tobytes()).hexdigest(),
                rows_predicted=int(held_mask.sum()),iterations=100,actual_parameters=model.get_params(),
                native_positive_column_exact=True,reload_array_equal=True,
                seconds=time.monotonic()-tick,
                warnings=sorted(set(str(w.message) for w in list(caught)+list(reload_warnings)))))
    require(len(fitted) == 8 and all(np.isfinite(p).all() for p in probabilities.values()),
            'Incomplete eight-fit OOF')
    output = [dict(row_id=row['row_id'],pmc_id=row['pmc_id'],CID=row['CID'],
                   patient_id=row['patient_id'],concept=row['concept'],
                   heldout_fold=row['heldout_fold'],
                   probabilities={arm:float(p[i]) for arm,p in probabilities.items()})
              for i,row in enumerate(data)]
    write_rows(directory/'OOF.scores.jsonl',output)
    save(directory/'fit-state.json',dict(status='COMPLETE_EIGHT_FITS_OOF_RELOAD_NO_F0',
        source=ref(__file__),plan=planref,preparation=ref(directory/'preparation.json'),
        scores=ref(directory/'OOF.scores.jsonl'),fits=fitted,folds=per_fold,
        features=FEATURES,parameters=PARAMETERS,runtime=dict(lightgbm=lgb.__version__,numpy=np.__version__,python=sys.version),
        F1_meta_training_labels_read=True,F0_read=False,new_quality_calls=0,
        elapsed_seconds=time.monotonic()-started,cap_seconds=60))
    require(time.monotonic()-started < 60,'Fit budget exceeded')
    signal.setitimer(signal.ITIMER_REAL,0)


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def finalize(plan, planref, directory):
    started = time.monotonic()
    require(not (directory/'evaluation-started.json').exists() and
            not (directory/'complete-freeze.json').exists(), 'Unique endpoint already reserved/frozen')
    signal.signal(signal.SIGALRM, lambda signum,frame: (_ for _ in ()).throw(TimeoutError('30s endpoint budget')))
    signal.setitimer(signal.ITIMER_REAL,30)
    state = read(directory/'fit-state.json')
    require(state['plan'] == planref and len(state['fits']) == 8, 'Incomplete frozen models')
    for item in state['fits']:
        checked(item['model'])
    scores = rows(checked(state['scores']))
    raw = Path(plan['refs']['F1_old']['path']).read_bytes().decode('utf-8').splitlines(keepends=True)
    template = [json.loads(line) for line in raw]
    predictions = {}
    for arm in FEATURES:
        chosen = defaultdict(lambda:defaultdict(set))
        for row in scores:
            if row['probabilities'][arm] >= .5:
                chosen[row['pmc_id']][row['patient_id']].add(row['concept'])
        lines = []
        for line, doc in zip(raw,template):
            value = [dict(patient_id=a['patient_id'],phenotype=sorted(chosen[doc['pmc_id']][a['patient_id']]))
                     for a in doc['association']]
            begin,end = association_span(line)
            lines.append(line[:begin]+json.dumps(value,ensure_ascii=False)+line[end:])
        require(len(lines) == 16 and nonrelation_bytes(lines) == nonrelation_bytes(raw),
                'Incomplete endpoint or nonrelation literal drift')
        path = directory/f'{arm}.OOF.pred.jsonl'
        with path.open('xb') as stream:
            stream.write(''.join(lines).encode('utf-8'));stream.flush();os.fsync(stream.fileno())
        predictions[arm] = ref(path)
    save(directory/'complete-freeze.json',dict(status='BOTH_COMPLETE_F1_OOF_PREDICTIONS_FROZEN',
        at_utc=now(),source=ref(__file__),plan=planref,fit_state=ref(directory/'fit-state.json'),
        predictions=predictions,scores=state['scores'],new_quality_calls=0,
        F1_training_labels_previously_read=True,F1_full_gold_file_not_yet_opened_in_this_run=True,
        nonrelation_literal_sha256=hashlib.sha256(nonrelation_bytes(raw)).hexdigest()))
    save(directory/'evaluation-started.json',dict(at_utc=now(),freeze=ref(directory/'complete-freeze.json'),
        quality_calls_reserved=2,old_metrics_reused=True,F0_read=False))
    # H50: register validator under evaluator's canonical sibling import name first.
    validator = module('validate_predictions',plan['refs']['validator']['path'])
    evaluator = module('r1_meta_oof_evaluator',plan['refs']['evaluator']['path'])
    require(evaluator.VERSION == 'patientphex-public-approx-v2','Evaluator changed')
    inputs = rows(plan['refs']['F1_input']['path'])
    hpo = validator.read_hpo(plan['refs']['hpo']['path'])
    # Training has already consumed known F1 labels. Freeze is not a claim of blind gold.
    goldref = plan['F1_gold_after_complete_freeze']
    require(sha(goldref['path']) == goldref['sha256'],'F1 gold version changed')
    gold = rows(goldref['path'])
    baselines = {}
    for name,item in plan['comparators'].items():
        value = read(checked(item['ref']))
        for field in item['metric_path']:
            value = value[field]
        baselines[name] = value
    strongest_name = max(baselines,key=lambda name:baselines[name]['Score'])
    strongest = baselines[strongest_name]
    require(strongest_name == plan['strongest_comparator'] and
            strongest['Score'] == plan['required_strong_score'], 'Strong baseline changed')
    endpoints = {}
    for arm,predref in predictions.items():
        pred = rows(checked(predref))
        valid = validator.validate_predictions(inputs,pred,hpo)
        save(directory/f'{arm}.validation.json',valid)
        require(valid['valid'],'Invalid OOF output')
        result = evaluator.evaluate_report(gold,pred,'all','text')
        save(directory/f'{arm}.metrics.json',result)
        require(result['documents'] == 16 and result['patients'] == 39 and
                all(result['metrics'][k] == strongest[k] for k in ('menF1','docF1')),
                'Evaluation coverage/nonrelation metrics changed')
        fold_description = []
        for fold,docids in enumerate(plan['heldout_document_groups']):
            docs = [d for d in result['per_document'] if d['pmc_id'] in docids]
            counts = {k:sum(d['relation'][k] for d in docs) for k in ('tp','fp','fn')}
            fold_description.append(dict(fold=fold,held_ids=docids,
                relation_counts_from_same_full_evaluation=counts,
                relation_micro_from_aggregated_counts=evaluator.prf(**counts)['f1'],
                selected_pairs=sum(r['probabilities'][arm] >= .5 for r in scores if r['pmc_id'] in docids),
                no_additional_evaluator_call=True))
        endpoints[arm] = dict(metrics=result['metrics'],counts=result['counts'],
            prediction=predref,metrics_file=ref(directory/f'{arm}.metrics.json'),
            validation_file=ref(directory/f'{arm}.validation.json'),fold_description=fold_description)
    a,b = endpoints['with_r1']['metrics'],endpoints['without_r1']['metrics']
    keep = a['Score'] > max(b['Score'],strongest['Score']) and a['macF1'] >= strongest['macF1']
    save(directory/'result.json',dict(status='COMPLETE_FIXED_DOCUMENT_META_OOF_DEVELOPMENT_SCREEN',
        keep=keep,decision='report_for_root_decision_no_automatic_F0' if keep else 'defer_no_F0_no_parameter_search',
        endpoints=endpoints,old_baselines_reused=baselines,strongest_name=strongest_name,
        required_score_strictly_above=max(b['Score'],strongest['Score']),required_macro_at_least=strongest['macF1'],
        delta_with_minus_without={k:a[k]-b[k] for k in a},
        delta_with_minus_strongest={k:a[k]-strongest[k] for k in a},
        plan=planref,source=ref(__file__),fit_state=ref(directory/'fit-state.json'),
        freeze=ref(directory/'complete-freeze.json'),new_v2_calls=2,old_v2_calls=0,
        F0_read=False,GPU_used=False,automatic_migration=False,
        limitation='Posthoc method selection on known F1; grouped 12-document meta-fit OOF does not prove or disprove the distinct16-document-fit F1→F0 method; not champion five-fold reproduction.',
        finalize_seconds=time.monotonic()-started,finalize_cap_seconds=30))
    for prediction in predictions.values():
        checked(prediction)
    require(time.monotonic()-started < 30,'Endpoint budget exceeded')
    signal.setitimer(signal.ITIMER_REAL,0)


def selftest():
    require(r1_vector({},'C0','P',['P'],False) == [0.,0.,0.], 'Invalid must be unknown')
    require(r1_vector({},'C0','P',['P'],True) == [0.,0.,0.], 'Missing must be unknown')
    require(r1_vector({'C0':[]},'C0','P',['P'],True) == [1.,0.,0.], 'Explicit empty is observed')
    require(r1_vector({'C0':['P']},'C0','P',['P','Q'],True) == [1.,1.,.5], 'Selected/fraction')
    groups = split_documents([str(i) for i in range(16)])
    require(groups == split_documents(list(reversed([str(i) for i in range(16)]))) and
            all(len(g) == 4 for g in groups), 'Split depends on input order')
    raw = ' {"pmc_id":"S", "entities":[{"text":"é"}], "association" : [],"x":2}\r\n'
    begin,end = association_span(raw)
    changed = raw[:begin]+'[{"patient_id":"Q","phenotype":[]}]'+raw[end:]
    require(nonrelation_bytes([raw]) == nonrelation_bytes([changed]), 'Literal bytes differ')
    return dict(status='PASS_SYNTHETIC_UNKNOWN_SPLIT_AND_LITERAL_CONTRACT',fit_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan')
    parser.add_argument('--stage',choices=['prepare','fit','finalize'])
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.self_test:
        print(json.dumps(selftest()));return
    plan,planref,directory = load_plan(args.plan)
    {'prepare':prepare,'fit':fit,'finalize':finalize}[args.stage](plan,planref,directory)
    require(ref(args.plan) == planref,'Plan changed')
    print(json.dumps(dict(stage=args.stage,status='complete')))


if __name__ == '__main__':
    main()
