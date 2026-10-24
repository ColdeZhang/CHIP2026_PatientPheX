"""One frozen CPU endpoint; no model, fit, threshold search or parser repair."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

S = Path(__file__).resolve().parent


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]


def publish(path, raw):
    path = Path(path)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    parent = path.parent.resolve()
    while True:
        fd = os.open(parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if parent == parent.parent: break
        parent = parent.parent


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def load(name, item):
    spec = importlib.util.spec_from_file_location(name, checked(item))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pairs(documents, ev):
    return {(str(d['pmc_id']), str(pid), h) for d in documents
            for pid, hs in ev.associations(d).items() for h in hs if h != 'NO'}


def primary(base, long_positive, support):
    return base | (long_positive & support)


def compatible(documents, base, ev):
    by = {str(d['pmc_id']): d for d in base}
    assert len(documents) == len(by) == 16
    assert {str(d['pmc_id']) for d in documents} == set(by)
    for d in documents:
        old = by[str(d['pmc_id'])]
        assert {k:v for k,v in d.items() if k not in ('entities','association')} == {
            k:v for k,v in old.items() if k not in ('entities','association')}
        assert ev.associations(d).keys() == ev.associations(old).keys()
        assert len(d['association']) == len(ev.associations(d))


def contract(plan):
    ev = load('consensus_contract_eval', plan['refs']['evaluator'])
    support = load('consensus_byte_support', plan['refs']['byte_support'])
    spans, mask = support.byte_helpers(plan['refs']['byte_helper'])
    assert primary({'b'}, {'l','x'}, {'s','x'}) == {'b','x'}
    assert primary({'b'}, {'b'}, set()) == {'b'}
    assert primary({'b'}, {'l'}, {'s'}) == {'b'}
    # Removing inherited B from S cannot change the protected endpoint.
    assert primary({'b'}, {'b','x'}, {'b','x'}) == primary({'b'}, {'b','x'}, {'x'})
    literal = {'pmc_id':'D','association':[{'patient_id':'P','phenotype':[
        'HP:0000001;HP:0000002', 'literal;NIL', 'NO']} ]}
    assert pairs([literal], ev) == {('D','P','HP:0000001'),('D','P','HP:0000002'),('D','P','literal;NIL')}
    raw = '{"pmc_id":"D", "entities":[], "association":[{"patient_id":"P", "phenotype":["HP:0000001;HP:0000002", "literal;NIL"], "extra":7}]}\n'
    values = ['HP:0000001;HP:0000002','literal;NIL','HP:0000003']
    rendered = support.replace_phenotypes(raw, {'P':values}, spans, mask)
    assert mask(raw) == mask(rendered)
    assert json.loads(rendered)['association'][0]['phenotype'] == values
    assert support.replace_phenotypes(raw, {'P':values[:2]}, spans, mask) == raw
    # An intersection on literal v2 atoms is not a substring/NIL match.
    assert ('D','P','literal;NIL') not in primary(set(), {('D','P','literal;NIL')}, {('D','P','literal')})
    tree = ast.parse(Path(__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ev'
             and n.func.attr.startswith('evaluate')]
    assert len(calls) == 1 and calls[0].func.attr == 'evaluate'
    config = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    assert config == dict(body_scope='all', no_id_doc='text', wildcard_no_id=True, validate=True)
    return dict(status='PASS_SYNTHETIC_NO_ACTUAL_INTERSECTION_NO_GOLD', checks=8,
                evaluation_call_sites=1, report_call_sites=0, actual_evaluate_calls=0,
                source=ref(__file__), evaluator=plan['refs']['evaluator'])


def run(plan_path):
    started = time.time(); signal.alarm(120)
    plan = read(plan_path); out = Path(plan['output'])
    guard = {'gold_allowed':False}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == os.path.realpath(plan['gold']['path']) and not guard['gold_allowed']:
                raise PermissionError('Held gold forbidden until complete fixed prediction is durable')
    sys.addaudithook(audit)
    assert plan['formula'] == 'B4 UNION (L_logit_gt_0 INTERSECT (G_old UNION S1_intersect_S2 UNION Q_native UNION C_strict))'
    assert plan['source'] == ref(__file__) and plan['cut'] == 0
    for item in plan['refs'].values(): checked(item)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'started.json', dict(at_unix=started, plan=ref(plan_path), source=ref(__file__), budget_seconds=120))
    r = plan['refs']; sys.path.insert(0,str(Path(r['evaluator']['path']).parent))
    ev = load('consensus_eval', r['evaluator']); va = load('consensus_validate', r['validator'])
    # Guard against accidentally calling the report's extra sensitivity evaluations.
    def forbidden_report(*args, **kwargs): raise AssertionError('Report/sensitivity evaluations prohibited')
    ev.evaluate_report = forbidden_report
    helper = load('consensus_bytes', r['byte_support']); spans, mask = helper.byte_helpers(r['byte_helper'])
    base_raw = checked(r['baseline']).read_bytes(); lines = base_raw.decode().splitlines(keepends=True)
    baseline = [json.loads(s) for s in lines]; inputs = rows(r['input']['path'])
    old = pairs(baseline, ev); prior = read(r['baseline_metrics']['path'])
    assert len(old) == 524 and prior['metrics']['Score'] == plan['baseline_score'] == 0.701325377212172
    assert len(baseline) == len(inputs) == 16 and sum(len(d['association']) for d in baseline) == 39
    assert sum(len(d['entities']) for d in baseline) == 1189
    available = {(str(d['pmc_id']),str(a['patient_id'])) for d in baseline for a in d['association']}
    views = {}
    for name in ('G','S','Q','C','inherited_baseline'):
        source = rows(r[name]['path']); compatible(source, baseline, ev); views[name] = pairs(source, ev)
    inherited = views.pop('inherited_baseline')
    assert inherited <= old and inherited <= views['S']
    # Stored S is B_old union S1 intersection S2. Outside B4 this subtraction is exact.
    views['S'] -= inherited
    support_keys = set().union(*views.values())
    complete = read(r['parent_complete']['path']); ended = read(r['parent_ended']['path'])
    assert ended['returncode'] == 0 and not ended['timed_out']
    assert complete['status'] == 'COMPLETE_FRESH30EP240_AND_ALL16_HELD_RELATIONS'
    assert complete['actual_adam_updates'] == 240 and complete['held_pairs'] == 2874 and complete['held_windows'] == 17
    assert complete['held_scores']['sha256'] == r['L_scores']['sha256']
    geometry = json.loads(gzip.decompress(checked(r['held_geometry']).read_bytes()))
    assert all('labels' not in d for d in geometry)
    expected = {tuple(k) for d in geometry for k in d['keys']}; scores = rows(r['L_scores']['path'])
    assert len(scores) == len(expected) == 2874 and {tuple(x['key']) for x in scores} == expected
    positive = set()
    for x in scores:
        k = tuple(x['key']); v = float(x['logit']); prob = float(x['probability'])
        assert k[:2] in available and re.fullmatch(r'HP:\d{7}',k[2])
        assert math.isfinite(v) and math.isfinite(prob) and 0 <= prob <= 1
        assert abs(prob-1/(1+math.exp(-max(-80.,min(80.,v))))) < 1e-7
        if v > 0: positive.add(k)
    final = primary(old, positive, support_keys); additions = final-old
    # Literal complete-source support only; no patient or label-specific selection.
    rendered = []
    for line, doc in zip(lines, baseline):
        values = {a['patient_id']:a['phenotype']+sorted(h for pmc,pid,h in additions
                  if pmc == str(doc['pmc_id']) and pid == str(a['patient_id'])) for a in doc['association']}
        rendered.append(helper.replace_phenotypes(line, values, spans, mask))
    payload = ''.join(rendered).encode(); predicted = [json.loads(s) for s in rendered]
    assert pairs(predicted,ev) == final and all(mask(a)==mask(b) for a,b in zip(lines,rendered))
    for b,d in zip(baseline,predicted):
        assert {k:v for k,v in b.items() if k!='association'} == {k:v for k,v in d.items() if k!='association'}
        for a,z in zip(b['association'],d['association']):
            assert z['phenotype'][:len(a['phenotype'])] == a['phenotype']
            assert {k:v for k,v in a.items() if k!='phenotype'} == {k:v for k,v in z.items() if k!='phenotype'}
    fmt = va.validate_predictions(inputs,predicted,va.read_hpo(r['hpo']['path'])); assert fmt['valid']
    save(out/'format.json',fmt); publish(out/'prediction.jsonl',payload)
    save(out/'freeze.json',dict(at_unix=time.time(), plan=ref(plan_path), source=ref(__file__),
        prediction=ref(out/'prediction.jsonl'), format=ref(out/'format.json'), all2874_scored=True,
        positive_L=len(positive), support_counts={k:len(v) for k,v in views.items()},
        support_union=len(support_keys), support_outside_B4=len(support_keys-old),
        intersection=len(positive&support_keys), added_pairs=sorted(additions), added=len(additions),
        baseline=524, final=len(final), documents=16, patients=39, entities=1189,
        inherited_S_removed=len(inherited), S_outside_B4_equivalence_proved=True,
        nonphenotype_bytes_exact=True, original_phenotype_prefixes_exact=True,
        gold_opened=False, byteexact=payload==base_raw, new_v2=0, old_v2=0))
    if not additions:
        assert payload == base_raw
        metrics = prior; calls=0; tp=fp=0
    else:
        save(out/'evaluation-intent.json',dict(at_unix=time.time(),freeze=ref(out/'freeze.json'),
            underlying_evaluate_calls=1, report_calls=0, sensitivity_calls=0, old_calls=0))
        guard['gold_allowed']=True
        gold=rows(checked(plan['gold'])); calls=1
        metrics=ev.evaluate(gold,predicted,body_scope='all',no_id_doc='text',wildcard_no_id=True,validate=True)
        metrics.update(evaluator_version=ev.VERSION, configuration=dict(body_scope='all',unlinked_document_mode='text'),
                       evaluation_calls=dict(underlying_evaluate=1,report=0,sensitivity=0))
        tp=metrics['counts']['relation']['tp']-prior['counts']['relation']['tp']
        fp=metrics['counts']['relation']['fp']-prior['counts']['relation']['fp']
        assert tp+fp==len(additions)
    save(out/'metrics.json',metrics)
    delta={k:metrics['metrics'][k]-v for k,v in prior['metrics'].items()}
    assert delta['menF1']==delta['docF1']==0
    result=dict(status='COMPLETE_SINGLE_FIXED_F1_CONSENSUS',baseline=prior['metrics'],new=metrics['metrics'],
        delta=delta,added=len(additions),added_TP=tp,added_FP=fp,strict_positive=delta['Score']>0,
        new_v2=calls,old_v2=0,report_calls=0,sensitivity_calls=0,NN_calls=0,fit_calls=0,remote_calls=0,
        seconds=time.time()-started,freeze=ref(out/'freeze.json'),not_blind=True,
        post_parent_negative_hypothesis=True,no_F0_executed=True)
    save(out/'result.json',result); signal.alarm(0)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['contract','run']); ap.add_argument('--plan',default=str(S/'plan-F1.json'))
    a=ap.parse_args()
    if a.mode=='contract':
        p=read(a.plan); p['refs']['evaluator']['path']=str(Path(p['refs']['evaluator']['path']))
        sys.path.insert(0,str(Path(p['refs']['evaluator']['path']).parent))
        save(S/'evidence/contract.json',contract(p)); print('8 synthetic contracts PASS; zero actual intersections/gold/evaluate')
    else: run(a.plan)
