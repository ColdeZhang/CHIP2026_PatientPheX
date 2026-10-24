"""Fixed complete-F1 control: add all accepted grounded mentions to strong Task1."""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def assemble(plan_path, p, ev, va):
    start = time.monotonic()
    r = p['refs']
    forbidden = os.path.realpath(r['gold']['path'])
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == forbidden:
                raise PermissionError('No held gold during assembly')
    sys.addaudithook(audit)
    out = Path(p['output'])
    out.mkdir(parents=True, exist_ok=False)
    put(out/'started.json', dict(at_utc=now(), plan=ref(plan_path)))
    inputs = va.load_jsonl(checked(r['input']))
    base = va.load_jsonl(checked(r['baseline']))
    docs = {d['pmc_id']: d for d in inputs}
    assert len(docs) == len(base) == 16 and set(docs) == {d['pmc_id'] for d in base}
    assert sum(len(d['patient']) for d in inputs) == 39
    assert sum(len(d['entities']) for d in base) == 1313
    quotes = [json.loads(line) for line in checked(r['normalized']).read_text().splitlines()]
    assert len(quotes) == 573
    manifest = json.loads(checked(r['normalization_manifest']).read_text())
    assert manifest['normalizations'] == r['normalized']
    assert manifest['input_sha256'] == r['input']['sha256']
    assert manifest['policy_sha256'] == r['policy']['sha256']
    checked(r['policy'])
    predicted = copy.deepcopy(base)
    by = {d['pmc_id']: d for d in predicted}
    known = {d['pmc_id']: {(e['offset'], e['length'], hid)
             for e in d['entities'] if e['note'] != 'NO'
             for hid in e['identifier'].split(';')} for d in base}
    accepted = 0
    additions = []
    for q in quotes:
        assert q['pmc_id'] in docs
        assert q['policy_sha256'] == r['policy']['sha256']
        assert q['status'] in {'mapped', 'nil', 'abstain'}
        if q['status'] == 'abstain':
            continue
        accepted += 1
        doc = docs[q['pmc_id']]
        assert q['target_patient_id'] in {a['patient_id'] for a in doc['patient']}
        passage = doc['full_text'][q['passage_index']]
        offset, length = q['offset'], q['length']
        assert type(offset) is int and type(length) is int and length == len(q['quote']) > 0
        begin = offset - passage['offset']
        assert begin >= 0 and passage['text'][begin:begin+length] == q['quote']
        ids = q['identifier'].split(';')
        assert set(ids) == set(q['ids']) and len(ids) == len(set(ids))
        assert (ids == ['-1']) == (q['status'] == 'nil')
        for hid in ids:
            key = (offset, length, hid)
            if key in known[q['pmc_id']]:
                continue
            known[q['pmc_id']].add(key)
            entity = dict(identifier=hid, offset=offset, length=length,
                          text=q['quote'], type='Phenotype', note=None)
            by[q['pmc_id']]['entities'].append(entity)
            additions.append(dict(pmc_id=q['pmc_id'], entity=entity))
    assert accepted == 328
    for old, new in zip(base, predicted):
        assert new['entities'][:len(old['entities'])] == old['entities']
        assert {k:v for k,v in old.items() if k != 'entities'} == {k:v for k,v in new.items() if k != 'entities'}
    result = va.validate_predictions(inputs, predicted, va.read_hpo(checked(r['hpo'])))
    put(out/'format.json', result)
    assert result['valid']
    path = out/'prediction.jsonl'
    with path.open('x') as f:
        for d in predicted:
            f.write(json.dumps(d, ensure_ascii=False, separators=(',', ':'))+'\n')
        f.flush()
        os.fsync(f.fileno())
    put(out/'additions.json', additions)
    put(out/'freeze.json', dict(at_utc=now(), plan=ref(plan_path), prediction=ref(path),
        format=ref(out/'format.json'), additions=ref(out/'additions.json'), added=len(additions),
        documents=16, patients=39, base_entities=1313, final_entities=1313+len(additions),
        accepted_anchor_records=accepted, old_entities_exact=True, nonentity_objects_exact=True,
        no_gold_read=True, NN_calls=0, fit_calls=0, seconds=time.monotonic()-start))


def evaluate(plan_path, p, ev, va):
    start = time.monotonic()
    out, r = Path(p['output']), p['refs']
    frozen = json.loads((out/'freeze.json').read_text())
    assert frozen['plan'] == ref(plan_path)
    pred_path = checked(frozen['prediction'])
    if frozen['added'] == 0:
        put(out/'result.json', dict(status='CLOSED_NO_SUPPLY', new_v2=0))
        return
    put(out/'evaluation-intent.json', dict(at_utc=now(), freeze=ref(out/'freeze.json')))
    gold = va.load_jsonl(checked(r['gold']))
    pred = va.load_jsonl(pred_path)
    metrics = ev.evaluate_report(gold, pred, body_scope='all', no_id_doc='text')
    put(out/'metrics.json', metrics)
    old = json.loads(checked(r['baseline_metrics']).read_text())
    assert old['metrics']['Score'] == p['baseline_score']
    delta = {k: metrics['metrics'][k]-v for k,v in old['metrics'].items()}
    assert delta['micF1'] == delta['macF1'] == 0
    put(out/'result.json', dict(status='COMPLETE_FIXED_TASK1_CONTROL', at_utc=now(),
        baseline=old['metrics'], metrics=metrics['metrics'], delta=delta,
        added=frozen['added'], count_delta={k:{x:metrics['counts'][k][x]-old['counts'][k][x]
            for x in ['tp','fp','fn']} for k in ['mention','document','relation']},
        new_v2=1, old_v2=0, NN_calls=0, fit_calls=0, retain=delta['Score']>0,
        not_blind=True, original_dev_read=False, freeze=ref(out/'freeze.json'),
        seconds=time.monotonic()-start))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('mode', choices=['assemble', 'evaluate'])
    a = ap.parse_args()
    p = json.loads(Path(a.plan).read_text())
    assert p['refs']['source'] == ref(__file__)
    evpath = checked(p['refs']['evaluator'])
    checked(p['refs']['validator'])
    sys.path.insert(0, str(evpath.parent))
    ev, va = importlib.import_module('evaluate'), importlib.import_module('validate_predictions')
    (assemble if a.mode == 'assemble' else evaluate)(a.plan, p, ev, va)
