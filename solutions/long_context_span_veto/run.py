"""One frozen CPU endpoint: exact represented long-context GP span max<0 veto."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(spec):
    require(ref(spec['path']) == spec, 'File identity changed: ' + spec['path'])
    return Path(spec['path'])


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    # Candidate records are not document JSONL (H60).
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def put_bytes(path, raw):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return ref(path)


def put(path, value):
    return put_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def block_gold(path):
    forbidden = os.path.realpath(path)
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == forbidden:
                raise PermissionError('Held gold blocked before complete prediction freeze')
    sys.addaudithook(guard)


def source_refs():
    return {'run.py': ref(HERE / 'run.py')}


def veto(base, boundaries):
    result, removed = copy.deepcopy(base), []
    for before, after in zip(base, result):
        kept = []
        for index, entity in enumerate(before['entities']):
            key = before['pmc_id'], entity['offset'], entity['length'], entity['text']
            if key in boundaries:
                removed.append(dict(pmc_id=before['pmc_id'], entity_index=index, entity=copy.deepcopy(entity)))
            else:
                kept.append(copy.deepcopy(entity))
        after['entities'] = kept
        require({k: v for k, v in before.items() if k != 'entities'} ==
                {k: v for k, v in after.items() if k != 'entities'}, 'Nonentity object changed')
    return result, removed


def payload(base_bytes, prediction, removed):
    return base_bytes if not removed else ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in prediction).encode()


def negative_spans(base, scores, window_ids):
    require(scores['schema'] == 'bioclinical-baseline-span-scores-v1' and
            scores['status'] == 'COMPLETE_ALL_BASELINE_TARGET_SCORES', 'Wrong score interface')
    require(len(window_ids) == len(set(window_ids)), 'Duplicate held window')
    allowed = set(window_ids); targets = {}
    for doc in base:
        for index, entity in enumerate(doc['entities']):
            identity = (doc['pmc_id'], index)
            require(identity not in targets, 'Duplicate baseline target identity')
            targets[identity] = entity
    require(len(scores['records']) == len(targets), 'Incomplete baseline target coverage')
    seen, by_span, negative = set(), {}, set()
    for record in scores['records']:
        identity = (record['pmc_id'], record['entity_index'])
        require(type(record['entity_index']) is int and identity in targets and identity not in seen,
                'Missing duplicate or unknown entity record')
        seen.add(identity); entity = targets[identity]
        require(record['target_id'] == f'{identity[0]}:{identity[1]}' and
                record['offset'] == entity['offset'] and record['length'] == entity['length'] and
                record['text_sha256'] == hashlib.sha256(entity['text'].encode()).hexdigest(),
                'Exact entity identity changed')
        require(type(record['representable']) is bool and
                record['representable'] == bool(record['window_scores']), 'Representation flag disagrees')
        windows, values = set(), []
        for row in record['window_scores']:
            require(row['row_id'] in allowed, 'Score from unknown held window')
            require(all(type(row[k]) is int and row[k] >= 0 for k in ('block_index','i','j')) and
                    row['i'] <= row['j'], 'Invalid GP representation')
            key = (row['row_id'], row['block_index'], row['i'], row['j'])
            require(key not in windows, 'Duplicate representation')
            windows.add(key)
            require(type(row['score']) in (float,int) and math.isfinite(row['score']), 'Nonfinite raw score')
            values.append(row['score'])
        if values:
            require(record['reason'] == 'OK' and type(record['max_score']) in (float,int) and
                    math.isfinite(record['max_score']) and record['max_score'] == max(values),
                    'Wrong or nonfinite cross-window max')
        else:
            require(record['max_score'] is None and
                    record['reason'] in ('native_unaligned_endpoint','no_common_window'),
                    'Unrepresented span must have no score')
        key = identity[0], entity['offset'], entity['length'], entity['text']
        signature = (record['representable'], record['reason'], record['max_score'],
                     sorted((r['row_id'],r['block_index'],r['i'],r['j'],r['score']) for r in record['window_scores']))
        if key in by_span:
            require(by_span[key] == signature, 'Same exact span has conflicting entity scores')
        by_span[key] = signature
        if values and record['max_score'] < 0:
            negative.add(key)
    require(seen == set(targets), 'Baseline target set differs')
    return negative


def validate_supply(scores, target_map, window_ids):
    require(set(target_map['window_targets']) == set(window_ids), 'Target map window set differs')
    originals = {r['target_id']:r for r in target_map['records']}
    require(len(originals) == len(target_map['records']) == len(scores['records']), 'Target-map row coverage differs')
    expected = {key:set() for key in originals}
    for window, rows in target_map['window_targets'].items():
        for row in rows:
            key = (window,row['block_index'],row['i'],row['j'])
            require(key not in expected[row['target_id']], 'Duplicate target-map representation')
            expected[row['target_id']].add(key)
    for row in scores['records']:
        require(row['target_id'] in originals and all(row[k] == v for k,v in originals[row['target_id']].items()),
                'Score metadata differs from frozen input target map')
        require({(r['row_id'],r['block_index'],r['i'],r['j']) for r in row['window_scores']} == expected[row['target_id']],
                'Missing or extra scored representation')


def bind(decision_path, endpoint, output, plan_path):
    decision = read(decision_path); block_gold(decision['refs']['gold']['path'])
    refs = copy.deepcopy(decision['refs'])
    refs['interface'] = ref(HERE / 'evidence/interface.json')
    refs.update(read(checked(refs['interface']))['refs'])
    for name, item in refs.items():
        if name != 'gold': checked(item)
    root = Path(endpoint).resolve()
    refs['complete'] = ref(root / 'complete.json')
    complete = read(checked(refs['complete']))
    require(complete['status'] == 'COMPLETE_FRESH8EP256_AND_ALL17_HELD_RAW' and
            complete['actual_adam_updates'] == 256 and complete['held_docs'] == 16 and
            complete['held_patients'] == 39 and complete['held_windows'] == 17,
            'Wrong or partial parent endpoint')
    require(len(complete['held_window_ids']) == len(set(complete['held_window_ids'])) == 17,
            'Incomplete held window identities')
    refs['scores'] = ref(root / 'baseline-span-scores.json')
    require(all(refs['scores'][k] == complete['baseline_span_scores'][k] for k in ('bytes','sha256')),
            'Sidecar differs from complete parent endpoint')
    put(plan_path, dict(status='BOUND_COMPLETE17_LONG_CONTEXT_GP_VETO', bound_utc=now(),
        decision=ref(decision_path), source=source_refs(), refs=refs,
        output=str(Path(output).resolve()), endpoints=1, new_v2_max=1, old_v2=0,
        score_content_not_parsed_during_binding=True, no_gold_opened=True))


def load_plan(path):
    plan = read(path)
    require(plan['source'] == source_refs(), 'Executable source differs from plan')
    decision = read(checked(plan['decision']))
    require(decision['experiment'] == 'E-20261024-LONG-CONTEXT-SPAN-VETO-001', 'Wrong experiment')
    require(all(plan['refs'][k] == v for k,v in decision['refs'].items()), 'Decision refs changed')
    require(plan['endpoints'] == plan['new_v2_max'] == 1 and plan['old_v2'] == 0, 'Budget differs')
    return plan, decision


def validator(refs):
    path = checked(refs['validator'])
    spec = importlib.util.spec_from_file_location('validate_predictions', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def assemble(plan_path):
    plan, decision = load_plan(plan_path); refs = plan['refs']; block_gold(refs['gold']['path'])
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)
    put(out / 'started.json', dict(at_utc=now(), plan=ref(plan_path), no_gold=True))
    for name, item in refs.items():
        if name != 'gold': checked(item)
    complete = read(checked(refs['complete'])); scores = read(checked(refs['scores']))
    require(all(scores['source_baseline'][k] == refs['baseline'][k] for k in ('bytes','sha256')),
            'Scores target a different baseline')
    require(all(scores['target_map'][k] == refs['target_map'][k] for k in ('bytes','sha256')) and
            all(scores['model_checkpoint'][k] == complete['checkpoint'][k] for k in ('bytes','sha256')),
            'Target map or model checkpoint differs')
    validate_supply(scores, read(checked(refs['target_map'])), complete['held_window_ids'])
    va = validator(refs)
    base = va.load_jsonl(checked(refs['baseline'])); inputs = va.load_jsonl(checked(refs['input']))
    require(len(base) == len(inputs) == 16 and sum(len(d['entities']) for d in base) == 1313 and
            sum(len(d['patient']) for d in inputs) == 39 and
            sum(len(d['association']) for d in base) == 39, 'Baseline shell differs')
    negative = negative_spans(base, scores, complete['held_window_ids'])
    prediction, removed = veto(base, negative)
    report = va.validate_predictions(inputs, prediction, va.read_hpo(checked(refs['hpo'])))
    put(out / 'format.json', report); require(report['valid'], 'Prediction invalid')
    prediction_ref = put_bytes(out / 'prediction.jsonl', payload(checked(refs['baseline']).read_bytes(), prediction, removed))
    put(out / 'freeze.json', dict(at_utc=now(), plan=ref(plan_path), prediction=prediction_ref,
        format=ref(out / 'format.json'), removed=removed, removed_entities=len(removed),
        negative_exact_spans=len(negative), entities=1313-len(removed), documents=16, patients=39,
        nonentity_objects_exact=True, remaining_entity_objects_and_order_exact=True,
        no_gold=True, fit_calls=0, NN_calls=0, inherited_exact_baseline=not removed))


def quality(plan_path):
    plan, decision = load_plan(plan_path); refs = plan['refs']; out = Path(plan['output'])
    freeze = read(out / 'freeze.json')
    require(freeze['plan'] == ref(plan_path), 'Freeze bound to different plan')
    pred_path = checked(freeze['prediction']); checked(freeze['format'])
    require(read(out / 'format.json')['valid'] is True, 'Invalid frozen prediction')
    baseline = read(checked(refs['baseline_metrics']))
    require(baseline['metrics']['Score'] == decision['baseline_score'], 'Baseline metrics differ')
    # Exclusive intent prevents a second evaluation, including after a failed call.
    put(out / 'quality-intent.json', dict(at_utc=now(), freeze=ref(out / 'freeze.json'),
        new_v2=int(not freeze['inherited_exact_baseline']), old_v2=0))
    if freeze['inherited_exact_baseline']:
        require(ref(pred_path)['sha256'] == refs['baseline']['sha256'] and freeze['removed_entities'] == 0,
                'Zero-change endpoint is not byte-exact baseline')
        report, calls = baseline, 0
    else:
        va = validator(refs); ev_path = checked(refs['evaluator'])
        spec = importlib.util.spec_from_file_location('boundary_v2', ev_path)
        ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
        require(ev.VERSION == 'patientphex-public-approx-v2', 'Evaluator version differs')
        gold = va.load_jsonl(checked(refs['gold'])); pred = va.load_jsonl(pred_path)
        report = ev.evaluate(gold, pred, body_scope='all', no_id_doc='text', wildcard_no_id=True, validate=False)
        report.update(evaluator_version=ev.VERSION, configuration={'body_scope':'all','no_id_doc':'text','wildcard_no_id':True},
                      prediction_sha256=freeze['prediction']['sha256'], gold_sha256=refs['gold']['sha256'])
        calls = 1
    require(all(report['metrics'][k] == baseline['metrics'][k] for k in ('micF1', 'macF1')), 'Relation metric changed')
    put(out / 'metrics.json', report)
    put(out / 'result.json', dict(at_utc=now(), metrics=report['metrics'], baseline=baseline['metrics'],
        delta={k:v-baseline['metrics'][k] for k,v in report['metrics'].items()},
        removed_entities=freeze['removed_entities'],
        removed_mention_TP=baseline['counts']['mention']['tp']-report['counts']['mention']['tp'],
        removed_mention_FP=baseline['counts']['mention']['fp']-report['counts']['mention']['fp'],
        retain=report['metrics']['Score'] > baseline['metrics']['Score'],
        new_v2=calls, old_v2=0, NN_calls=0, fit_calls=0, development_not_blind=True,
        freeze=ref(out / 'freeze.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    binding = sub.add_parser('plan')
    binding.add_argument('--decision', default=str(HERE / 'decision.json'))
    for name in ('endpoint', 'output', 'plan-output'):
        binding.add_argument('--' + name, required=True)
    for mode in ('assemble', 'quality'):
        sub.add_parser(mode).add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.mode == 'plan': bind(args.decision, args.endpoint, args.output, args.plan_output)
    elif args.mode == 'assemble': assemble(args.plan)
    else: quality(args.plan)


if __name__ == '__main__': main()
