"""One frozen CPU endpoint: delete only an exact, complete BOUNDARY span."""
import argparse
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from schema import strict_parse

HERE = Path(__file__).resolve().parent
RAW_KEYS = {'row_id', 'index', 'input_tokens', 'generated_ids',
            'raw_text_with_special', 'seconds', 'max_new_tokens', 'eos_token_id'}
RESULT_KEYS = RAW_KEYS | {'text', 'eos', 'capped', 'parsed', 'rejection'}


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
    return {name: ref(HERE / name) for name in ('run.py', 'schema.py')}


def span(row, text_key):
    return row['pmc_id'], row['offset'], row['length'], row[text_key]


def validate_rows(mentions, encoded, inputs):
    require(len(mentions) == len(encoded), 'Missing encoded rows')
    documents = {d['pmc_id']: d for d in inputs}
    require(len(documents) == len(inputs), 'Duplicate input document')
    ids, spans = set(), set()
    for row, tokens in zip(mentions, encoded):
        key = span(row, 'quote')
        require(row['row_id'] not in ids and key not in spans, 'Ambiguous candidate identity')
        ids.add(row['row_id']); spans.add(key)
        require(type(row['offset']) is int and type(row['length']) is int,
                'Offsets and lengths must be integers')
        require(row['offset'] >= 0 and row['length'] == len(row['quote']) > 0,
                'Invalid candidate span')
        require(all(tokens[k] == row[k] for k in ('row_id', 'pmc_id', 'offset', 'length', 'quote', 'candidates')),
                'Encoded row differs from held pool')
        require('answer' not in row and 'target_status' not in row, 'Held row contains target')
        passages = documents[row['pmc_id']]['full_text']
        matches = [p for p in passages if p['offset'] <= row['offset'] and
                   row['offset'] + row['length'] <= p['offset'] + len(p['text'])]
        require(len(matches) == 1, 'Candidate passage is absent or ambiguous')
        p = matches[0]; start = row['offset'] - p['offset']
        require(p['text'][start:start + row['length']] == row['quote'], 'Candidate quote differs from input')
    require({r['pmc_id'] for r in mentions} == set(documents), 'Candidate document coverage differs')


def boundary_keys(mentions, responses, raw_rows, parsed_rows, encoded):
    require(len(mentions) == len(responses) == len(raw_rows) == len(parsed_rows) == len(encoded),
            'Incomplete response or per-row evidence coverage')
    ids, spans, boundary = set(), set(), set()
    for index, (m, r, raw, parsed, tokens) in enumerate(zip(mentions, responses, raw_rows, parsed_rows, encoded)):
        key = span(m, 'quote')
        require(m['row_id'] not in ids and key not in spans, 'Duplicate row or exact candidate span')
        ids.add(m['row_id']); spans.add(key)
        require(set(r) == RESULT_KEYS and set(raw) == RAW_KEYS, 'Unknown or missing actual response fields')
        require(type(r['index']) is int and r['index'] == index and r['row_id'] == m['row_id'],
                'Response row identity/order changed')
        require(parsed == r and raw == {k: r[k] for k in RAW_KEYS}, 'Raw/parsed/results disagree')
        require(type(r['input_tokens']) is int and r['input_tokens'] == len(tokens['input_ids']),
                'Actual input token count differs')
        require(type(r['max_new_tokens']) is int and r['max_new_tokens'] == 128,
                'Generation limit changed')
        require(type(r['eos_token_id']) is int and r['eos_token_id'] == 151645, 'EOS identity changed')
        generated = r['generated_ids']
        require(type(generated) is list and all(type(t) is int and t >= 0 for t in generated),
                'Invalid raw token IDs')
        require(len(generated) <= r['max_new_tokens'], 'Output exceeds fixed cap')
        eos = bool(generated and generated[-1] == r['eos_token_id'])
        capped = len(generated) >= r['max_new_tokens']
        require(type(r['eos']) is bool and type(r['capped']) is bool and
                r['eos'] == eos and r['capped'] == capped, 'EOS/cap metadata differs from token evidence')
        try:
            expected = strict_parse(r['text'], m['candidates'], eos=eos, capped=capped)
            rejection = None
        except (ValueError, TypeError) as error:
            expected, rejection = None, str(error)
        require(r['parsed'] == expected and r['rejection'] == rejection, 'Strict parse/rejection differs')
        if expected is not None and expected['status'] == 'BOUNDARY':
            require(expected == {'status': 'BOUNDARY', 'selected': [], 'ids': []}, 'Invalid BOUNDARY mode')
            boundary.add(key)
    return boundary


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


def bind(decision_path, inference, output, plan_path):
    decision = read(decision_path); block_gold(decision['refs']['gold']['path'])
    refs = copy.deepcopy(decision['refs'])
    for key, item in refs.items():
        if key != 'gold':
            checked(item)
    require(ref(HERE / 'schema.py')['sha256'] == refs['parent_schema']['sha256'],
            'Copied parser differs from actual inference source')
    infer = Path(inference).resolve()
    refs['inference_complete'] = ref(infer / 'complete.json')
    complete = read(checked(refs['inference_complete']))
    require(complete['status'] == 'COMPLETE_FIXED_MENTION_CANDIDATE_INFERENCE' and
            complete['processed_rows'] == 1416 and complete['training_epoch'] == 2 and
            complete['documents'] == 16 and complete['patients'] == 39 and
            complete['no_quality_gold_read'] is True, 'Incomplete or different inference endpoint')
    refs['responses'] = ref(infer / 'results.json')
    require(refs['responses']['sha256'] == complete['results_sha256'], 'Results differ from terminal')
    members = [{kind: ref(infer / 'generation' / f'{i:04d}.{kind}.json')
                for kind in ('raw', 'parsed')} for i in range(1416)]
    # Hash every original response; do not inspect output counts to choose a policy.
    put(plan_path, dict(status='BOUND_COMPLETE_1416_SINGLE_BOUNDARY_VETO', bound_utc=now(),
        decision=ref(decision_path), source=source_refs(), refs=refs, generation=members,
        output=str(Path(output).resolve()), endpoints=1, new_v2_max=1, old_v2=0,
        no_gold_opened=True, actual_response_content_not_parsed_during_binding=True))


def load_plan(path):
    plan = read(path)
    require(plan['source'] == source_refs(), 'Executable source differs from bound plan')
    decision = read(checked(plan['decision']))
    require(decision['experiment'] == 'E-20261024-BOUNDARY-VETO-001', 'Wrong experiment')
    require(all(plan['refs'][k] == v for k, v in decision['refs'].items()), 'Decision source refs changed')
    require(ref(HERE / 'schema.py')['sha256'] == plan['refs']['parent_schema']['sha256'],
            'Parser differs from frozen parent schema')
    require(plan['endpoints'] == 1 and plan['new_v2_max'] == 1 and plan['old_v2'] == 0, 'Endpoint budget differs')
    return plan, decision


def validator(refs):
    path = checked(refs['validator'])
    spec = importlib.util.spec_from_file_location('validate_predictions', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def assemble(plan_path):
    plan, decision = load_plan(plan_path); refs = plan['refs']
    block_gold(refs['gold']['path'])
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)
    put(out / 'started.json', dict(at_utc=now(), plan=ref(plan_path), no_gold=True))
    for name, item in refs.items():
        if name != 'gold':
            checked(item)
    complete = read(checked(refs['inference_complete']))
    require(complete['results_sha256'] == refs['responses']['sha256'] and
            complete['processed_rows'] == 1416 and complete['training_epoch'] == 2,
            'Inference completion changed')
    pool = read(checked(refs['pool_manifest']))
    require(pool['files']['held'] == refs['mentions'] and pool['held_rows'] == 1416,
            'Pool manifest differs')
    cpu = read(checked(refs['cpu_complete']))
    require(cpu['counts']['dev']['encoded']['sha256'] == refs['encoded']['sha256'], 'Encoded pool differs')
    mentions = jsonl(checked(refs['mentions'])); responses = read(checked(refs['responses']))
    with gzip.open(checked(refs['encoded']), 'rt') as stream:
        encoded = [json.loads(line) for line in stream]
    raw_rows, parsed_rows = [], []
    require(len(plan['generation']) == 1416, 'Missing per-row manifest')
    for item in plan['generation']:
        raw_rows.append(read(checked(item['raw'])))
        parsed_rows.append(read(checked(item['parsed'])))
    va = validator(refs)
    base = va.load_jsonl(checked(refs['baseline'])); inputs = va.load_jsonl(checked(refs['input']))
    require(len(mentions) == len(responses) == 1416 and len(base) == len(inputs) == 16, 'Coverage differs')
    require(sum(len(d['entities']) for d in base) == 1313 and
            sum(len(d['patient']) for d in inputs) == 39 and
            sum(len(d['association']) for d in base) == 39, 'Baseline shell differs')
    validate_rows(mentions, encoded, inputs)
    boundaries = boundary_keys(mentions, responses, raw_rows, parsed_rows, encoded)
    prediction, removed = veto(base, boundaries)
    report = va.validate_predictions(inputs, prediction, va.read_hpo(checked(refs['hpo'])))
    put(out / 'format.json', report); require(report['valid'], 'Prediction format invalid')
    data = payload(checked(refs['baseline']).read_bytes(), prediction, removed)
    prediction_ref = put_bytes(out / 'prediction.jsonl', data)
    put(out / 'freeze.json', dict(at_utc=now(), plan=ref(plan_path), prediction=prediction_ref,
        format=ref(out / 'format.json'), removed=removed, removed_entities=len(removed),
        complete_BOUNDARY_mentions=len(boundaries), entities=1313-len(removed), documents=16, patients=39,
        nonentity_objects_exact=True, remaining_entity_objects_and_order_exact=True,
        compound_and_multiple_HPO_objects_follow_same_span_rule=True,
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
    for name in ('inference', 'output', 'plan-output'):
        binding.add_argument('--' + name, required=True)
    for mode in ('assemble', 'quality'):
        sub.add_parser(mode).add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.mode == 'plan':
        bind(args.decision, args.inference, args.output, args.plan_output)
    elif args.mode == 'assemble':
        assemble(args.plan)
    else:
        quality(args.plan)


if __name__ == '__main__':
    main()
