"""One predeclared Task1 endpoint: complete, explicit NO removes an exact span."""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ref(path):
    p = Path(path)
    b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def checked(spec):
    assert ref(spec['path']) == spec, spec['path']
    return Path(spec['path'])


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def veto(base, mentions, responses):
    assert len(mentions) == len(responses)
    by_id, by_span, negative = {}, {}, set()
    for i, (m, r) in enumerate(zip(mentions, responses)):
        key = (m['pmc_id'], m['offset'], m['length'], m['quote'])
        assert m['row_id'] not in by_id and key not in by_span
        assert r['row_id'] == m['row_id'] and r['index'] == i
        assert type(m['offset']) is int and type(m['length']) is int
        assert m['offset'] >= 0 and m['length'] == len(m['quote']) > 0
        by_id[m['row_id']] = r
        by_span[key] = r
        p = r['parsed']
        if p is None:
            continue  # An abstention never authorizes deletion.
        assert p['status'] in ('NO', 'NIL', 'HPO')
        if p['status'] != 'NO':
            continue
        assert p['ids'] == []
        if r['eos'] is True and r['capped'] is False and r['rejection'] is None:
            assert json.loads(r['text']) == []
            negative.add(key)
    result = copy.deepcopy(base)
    removed = []
    for before, after in zip(base, result):
        kept = []
        for e in before['entities']:
            key = (before['pmc_id'], e['offset'], e['length'], e['text'])
            if key in negative:
                removed.append(dict(pmc_id=before['pmc_id'], entity=copy.deepcopy(e)))
            else:
                kept.append(copy.deepcopy(e))
        after['entities'] = kept
        assert {k: v for k, v in before.items() if k != 'entities'} == {
            k: v for k, v in after.items() if k != 'entities'}
    return result, removed, len(negative)


def main(plan_path, mode):
    plan = json.loads(Path(plan_path).read_text())
    refs, out = plan['refs'], Path(plan['output'])
    assert refs['source'] == ref(__file__)
    decision = json.loads(checked(refs['decision']).read_text())
    assert decision['experiment'] == 'E-20261024-MENTION-NO-VETO-001'
    assert refs['baseline']['sha256'] == decision['baseline_sha256']
    evaluator = checked(refs['evaluator']); checked(refs['validator'])
    assert refs['evaluator']['sha256'] == 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    assert refs['validator']['sha256'] == 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f'
    sys.path.insert(0, str(evaluator.parent))
    ev = importlib.import_module('evaluate')
    va = importlib.import_module('validate_predictions')
    assert ev.VERSION == 'patientphex-public-approx-v2'
    if mode == 'assemble':
        forbidden = os.path.realpath(refs['gold']['path'])
        def guard(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                if os.path.realpath(os.fsdecode(args[0])) == forbidden:
                    raise PermissionError('Held gold blocked before full prediction freeze')
        sys.addaudithook(guard)
        out.mkdir(parents=True, exist_ok=False)
        put(out/'started.json', dict(at_utc=now(), plan=ref(plan_path)))
        complete = json.loads(checked(refs['inference_complete']).read_text())
        assert complete['status'] == 'COMPLETE_FIXED_MENTION_CANONICAL_INFERENCE'
        assert complete['processed_rows'] == 1416 and complete['training_epoch'] == 2
        assert complete['results_sha256'] == refs['responses']['sha256']
        mentions = [json.loads(x) for x in checked(refs['mentions']).read_text().splitlines()]
        responses = json.loads(checked(refs['responses']).read_text())
        assert len(mentions) == len(responses) == 1416
        base = va.load_jsonl(checked(refs['baseline']))
        inputs = va.load_jsonl(checked(refs['input']))
        assert len(base) == len(inputs) == 16
        assert sum(len(d['entities']) for d in base) == 1313
        assert sum(len(d['association']) for d in base) == 39
        assert {d['pmc_id'] for d in base} == {m['pmc_id'] for m in mentions}
        prediction, removed, no_count = veto(base, mentions, responses)
        report = va.validate_predictions(inputs, prediction, va.read_hpo(checked(refs['hpo'])))
        put(out/'format.json', report)
        assert report['valid']
        payload = checked(refs['baseline']).read_bytes() if not removed else ''.join(
            json.dumps(d, ensure_ascii=False)+'\n' for d in prediction).encode()
        with (out/'prediction.jsonl').open('xb') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        put(out/'freeze.json', dict(at_utc=now(), plan=ref(plan_path), prediction=ref(out/'prediction.jsonl'),
            format=ref(out/'format.json'), removed=removed, removed_entities=len(removed),
            complete_NO_mentions=no_count, entities=1313-len(removed),
            nonentity_objects_exact=True, remaining_entity_objects_and_order_exact=True,
            no_gold=True, fit_calls=0, NN_calls=0, inherited_exact_baseline=not removed))
    else:
        freeze = json.loads((out/'freeze.json').read_text())
        assert freeze['plan'] == ref(plan_path)
        checked(freeze['prediction']); checked(freeze['format'])
        baseline = json.loads(checked(refs['baseline_metrics']).read_text())
        assert baseline['metrics']['Score'] == decision['baseline_score']
        if freeze['inherited_exact_baseline']:
            report, calls = baseline, 0
        else:
            put(out/'evaluation-intent.json', dict(at_utc=now(), freeze=ref(out/'freeze.json'), new_v2=1, old_v2=0))
            gold = va.load_jsonl(checked(refs['gold']))
            pred = va.load_jsonl(checked(freeze['prediction']))
            report = ev.evaluate_report(gold, pred, body_scope='all', no_id_doc='text')
            calls = 1
        assert all(report['metrics'][k] == baseline['metrics'][k] for k in ('micF1', 'macF1'))
        put(out/'metrics.json', report)
        put(out/'result.json', dict(at_utc=now(), metrics=report['metrics'], baseline=baseline['metrics'],
            delta={k: v-baseline['metrics'][k] for k,v in report['metrics'].items()},
            removed_entities=freeze['removed_entities'],
            removed_mention_TP=baseline['counts']['mention']['tp']-report['counts']['mention']['tp'],
            removed_mention_FP=baseline['counts']['mention']['fp']-report['counts']['mention']['fp'],
            retain=report['metrics']['Score']>baseline['metrics']['Score'],
            new_v2=calls, old_v2=0, NN_calls=0, fit_calls=0, development_not_blind=True,
            freeze=ref(out/'freeze.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('mode', choices=('assemble','evaluate'))
    args = parser.parse_args()
    main(args.plan, args.mode)
