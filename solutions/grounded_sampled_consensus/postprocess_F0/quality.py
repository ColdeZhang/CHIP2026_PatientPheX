"""One prespecified B4 UNION (S1 INTERSECT S2), with a unique bounded v2 call."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from common import (BASE_SCORE, BASE_SHA, GOLD, POLICY_SHA, SEEDS, all_patient_keys,
                    checked, exact_intersection, guard, jsonl, load_plan, modules,
                    publish, read, ref, save)


def now(): return datetime.now(timezone.utc).isoformat()


def verify_normalized(plan):
    refs = plan['inputs']; norm = read(checked(refs['normalization']))
    assert norm['schema'] == 'grounded-sampled-complete94-normalization-v1'
    assert norm['complete_all94'] and norm['rows'] == 94 and norm['seeds'] == list(SEEDS)
    assert norm['no_policy_or_threshold_changes'] and norm['no_gold_fit_or_quality']
    assert norm['sources'] == plan['sources']
    final_plan = load_plan(checked(norm['plan']), 'finalize')
    prep = read(checked(norm['preparation'])); ex = read(checked(norm['extraction']))
    assert final_plan['inputs']['preparation'] == norm['preparation']
    assert prep['extraction'] == norm['extraction'] and prep['sources'] == plan['sources']
    assert ex['sources'] == plan['sources']
    prep_plan = load_plan(checked(prep['plan']), 'prepare')
    extract_plan = load_plan(checked(ex['plan']), 'extract')
    assert prep_plan['inputs']['extraction'] == norm['extraction']
    assert extract_plan['inputs']['collection'] == ex['collection']
    assert extract_plan['inputs']['F0_input'] == refs['F0_input']
    assert prep['complete_all94'] and ex['complete_all94'] and ex['rows'] == 94
    assert norm['input'] == prep['input'] == ex['F0_input'] == refs['F0_input']
    assert norm['policy'] == prep['policy'] == refs['policy'] and refs['policy']['sha256'] == POLICY_SHA
    assert norm['train32'] == prep['train32'] == refs['train32']
    collection = read(checked(ex['collection'])); assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
    for item in collection['files'].values(): checked(item)
    assert collection['wrapper']['complete94_actual_members_verified']
    quotes = jsonl(checked(ex['quotes'])); decisions = read(checked(ex['decisions']))
    rows = jsonl(checked(norm['normalizations'])); assert len(quotes) == len(rows) == norm['records']
    reconstructed = []
    for d in decisions:
        for i, q in enumerate(d['decision']['anchors']):
            reconstructed.append(dict(seed=d['seed'], row_id=d['row_id'], pmc_id=d['pmc_id'],
                target_patient_id=d['target_patient_id'], anchor_index=i, **q))
    assert quotes == reconstructed and len(decisions) == 94
    for q, row in zip(quotes, rows):
        assert all(row[k] == value for k, value in q.items())
        assert row['policy_sha256'] == POLICY_SHA and row['normalization_policy'] == 'grounded-public-sap-conservative-v1'
    if prep['necessary_public_query_count']:
        c = read(checked(norm['query_collection'])); q = read(checked(norm['query_complete']))
        for name in ['query_collection','query_complete','query_scores']:
            assert final_plan['inputs'][name] == norm[name]
        for item in c['files'].values(): checked(item)
        assert c['all_actual_SHA_verified'] and c['query_reported_success']
        assert not c['wrapper']['timed_out'] and 'endpoint_error' not in c['wrapper']
        checked(norm['query_scores']); checked(norm['semantic_evidence'])
        assert q['queries']['sha256'] == prep['queries']['sha256'] and q['scores']['sha256'] == norm['query_scores']['sha256']
        assert q['policy']['sha256'] == POLICY_SHA and q['query_count'] == prep['necessary_public_query_count']
        assert q['public_bank_rows'] == 41099 and q['no_bank_reencoding'] and q['no_fit']
    else: assert all(norm[k] is None for k in ['query_collection','query_complete','query_scores'])
    return rows, decisions


def assemble(path):
    state = guard(); plan = load_plan(path, 'quality'); refs = plan['inputs']
    assert plan['formula'] == 'B4 UNION (S1 INTERSECT S2)' and plan['new_v2_max'] == 1 and plan['old_v2'] == 0
    assert plan['baseline_score'] == BASE_SCORE and plan['expected_baseline_relations'] == 343
    H, A, _, schema, strict, _ = modules()
    for name, item in refs.items():
        if name != 'gold': checked(item)
    assert refs['baseline']['sha256'] == BASE_SHA
    lines, baseline, ids = H.rows(refs['baseline']['path'])
    _, inputs, input_ids = H.rows(refs['F0_input']['path'])
    assert ids == input_ids and len(ids) == 16
    assert sum(len(d['entities']) for d in baseline) == 947
    patients = all_patient_keys(inputs)
    assert sum(len(d['association']) for d in baseline) == 47
    for doc, inp in zip(baseline, inputs):
        assert [a['patient_id'] for a in doc['association']] == [p['patient_id'] for p in inp['patient']]
    normalized, decisions = verify_normalized(plan)
    hpo = strict.HPO(refs['hpo']['path'])
    class BoundStrict:
        HPO = hpo
        verify_span = staticmethod(strict.verify_span)
    selected = {}; consumption = {}
    for seed in SEEDS:
        subset = [r for r in normalized if r['seed'] == seed]
        ds = [d for d in decisions if d['seed'] == seed]
        assert len(ds) == 47 and {(d['pmc_id'], d['target_patient_id']) for d in ds} == patients
        selected[seed], records, counts = A.new_relations(subset, inputs, BoundStrict, schema, POLICY_SHA)
        assert set(selected[seed]) == patients
        consumption[str(seed)] = dict(records=records, counts=counts, all47_decisions=True)
    both = exact_intersection(selected[SEEDS[0]], selected[SEEDS[1]])
    both_keys = {(did,pid,*key) for (did,pid), values in both.items() for key in values}
    old = H.relation_keys(baseline, hpo); assert len(old) == 343
    added = both_keys - old
    spans, mask = H.byte_helpers(refs['byte_helper']); rendered = []
    for line, doc in zip(lines, baseline):
        values = {}
        for assoc in doc['association']:
            pid = assoc['patient_id']; extras = sorted((kind,value) for did,p,kind,value in added if did == str(doc['pmc_id']) and p == pid)
            values[pid] = assoc['phenotype'] + [value for _, value in extras]
        changed = H.replace_phenotypes(line, values, spans, mask)
        assert mask(changed) == mask(line)
        rendered.append(changed)
    pred = [json.loads(line) for line in rendered]
    assert H.relation_keys(pred, hpo) == old | both_keys
    for base, current in zip(baseline, pred):
        assert {k:v for k,v in base.items() if k != 'association'} == {k:v for k,v in current.items() if k != 'association'}
        for before, after in zip(base['association'], current['association']):
            assert after['phenotype'][:len(before['phenotype'])] == before['phenotype']
            assert {k:v for k,v in before.items() if k != 'phenotype'} == {k:v for k,v in after.items() if k != 'phenotype'}
    if not added: assert ''.join(rendered).encode() == Path(refs['baseline']['path']).read_bytes()
    validator = H.load_module('validate_predictions', refs['validator'])
    validation = validator.validate_predictions(inputs, pred, validator.read_hpo(refs['hpo']['path']))
    assert validation['valid']
    out = Path(plan['output_directory']); out.mkdir(parents=True, exist_ok=False)
    publish(out/'prediction.jsonl', ''.join(rendered).encode())
    save(out/'format.json', validation); save(out/'consumption.json', consumption)
    save(out/'all-seed-keys.json', {str(s):sorted((did,pid,*key) for (did,pid), ks in selected[s].items() for key in ks) for s in SEEDS})
    save(out/'intersection-and-additions.json', dict(intersection=sorted(both_keys), added=sorted(added)))
    assert not state['F0_quality_enabled']
    save(out/'freeze.json', dict(schema='grounded-sampled-one-complete-endpoint-freeze-v1', at_utc=now(),
        plan=ref(path), sources=plan['sources'], formula=plan['formula'], prediction=ref(out/'prediction.jsonl'),
        format=ref(out/'format.json'), consumption=ref(out/'consumption.json'), all_seed_keys=ref(out/'all-seed-keys.json'),
        contributions=ref(out/'intersection-and-additions.json'), normalization=refs['normalization'],
        source_relations={str(s):sum(map(len, selected[s].values())) for s in SEEDS},
        intersection_relations=len(both_keys), added=len(added), final_relations=len(old|both_keys),
        baseline_relations=343, documents=16, patients=47, entities=947, complete_all94=True,
        nonphenotype_bytes_exact=True, baseline_relation_lists_prefix_exact=True,
        no_added_prediction_byteexact=not added, gold_opened_or_hashed=False, new_v2=0, old_v2=0))
    print(json.dumps(ref(out/'freeze.json')))


def check_freeze(path, plan):
    out = Path(plan['output_directory']); f = read(out/'freeze.json')
    assert f['plan'] == ref(path) and f['sources'] == plan['sources'] and f['formula'] == plan['formula']
    assert f['complete_all94'] and f['nonphenotype_bytes_exact'] and f['baseline_relation_lists_prefix_exact']
    assert not f['gold_opened_or_hashed'] and f['new_v2'] == f['old_v2'] == 0
    for k in ['prediction','format','consumption','all_seed_keys','contributions','normalization']: checked(f[k])
    assert read(f['format']['path'])['valid']
    return out, f


def worker(path):
    state = guard(); p = load_plan(path, 'quality'); out, frozen = check_freeze(path, p); refs=p['inputs']
    assert read(out/'evaluation-launch.json')['freeze'] == ref(out/'freeze.json')
    H, _, _, _, _, _ = modules()
    prior = read(checked(refs['baseline_metrics']))
    baseline_freeze = read(checked(refs['baseline_freeze']))
    baseline_plan = read(checked(refs['baseline_plan']))
    assert prior['prediction_sha256'] == refs['baseline']['sha256'] == BASE_SHA
    assert prior['gold_sha256'] == GOLD['sha256'] and prior['metrics']['Score'] == BASE_SCORE
    assert baseline_freeze['prediction'] == refs['baseline'] and baseline_freeze['plan'] == refs['baseline_plan']
    assert baseline_plan['refs']['gold'] == GOLD and baseline_plan['refs']['input'] == refs['F0_input']
    if not frozen['added']:
        assert checked(frozen['prediction']).read_bytes() == checked(refs['baseline']).read_bytes()
        save(out/'result.json', dict(status='NO_ADDED_KEYS_BASE_BYTEEXACT_ZERO_V2', baseline=prior['metrics'],
            keep=False, new_v2=0, old_v2=0, freeze=ref(out/'freeze.json'), no_gold_read=True))
        return
    validator = H.load_module('validate_predictions', refs['validator'])
    evaluator = H.load_module('_sampled_grounded_public_v2', refs['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions and evaluator.VERSION == 'patientphex-public-approx-v2'
    assert not (out/'evaluation-intent.json').exists() and not (out/'metrics.json').exists()
    save(out/'evaluation-intent.json', dict(at_utc=now(), plan=ref(path), freeze=ref(out/'freeze.json'),
        prediction=frozen['prediction'], gold=GOLD, new_v2=1, old_v2=0))
    state['F0_quality_enabled'] = True
    checked(GOLD); gold=validator.load_jsonl(GOLD['path']); prediction=validator.load_jsonl(frozen['prediction']['path'])
    metrics=evaluator.evaluate_report(gold,prediction,body_scope='all',no_id_doc='text')
    metrics.update(prediction_sha256=frozen['prediction']['sha256'], gold_sha256=GOLD['sha256'])
    save(out/'metrics.json',metrics)
    delta={k:metrics['metrics'][k]-prior['metrics'][k] for k in ('menF1','docF1','micF1','macF1','Score')}
    # Gold duplicates are legal: use the evaluator's set semantics.
    truth={(str(d['pmc_id']),pid,value) for d in gold for pid,values in evaluator.associations(d).items() for value in values}
    added={(did,pid,value) for did,pid,kind,value in read(frozen['contributions']['path'])['added']}
    result=dict(status='COMPLETE_ONE_FIXED_ENDPOINT', baseline=prior['metrics'], metrics=metrics['metrics'], delta=delta,
        added=len(added), added_TP=len(added&truth), added_FP=len(added-truth), keep=delta['Score']>0,
        macro_has_no_extra_veto=True, new_v2=1, old_v2=0, freeze=ref(out/'freeze.json'),
        Task1_unchanged=delta['menF1']==delta['docF1']==0, development_not_blind=True, originaldev_read=False,
        no_single_seed_quality=True, no_other_arm_or_seed_scan=True)
    save(out/'result.json',result)
    assert result['Task1_unchanged']


def evaluate(path):
    guard(); p=load_plan(path,'quality'); out,f=check_freeze(path,p)
    assert p['score_budget_seconds']==30 and p['term_grace_seconds']==1
    assert not any((out/name).exists() for name in ['evaluation-launch.json','evaluation-intent.json','metrics.json','result.json'])
    command=[sys.executable,str(Path(__file__).resolve()),'worker','--plan',str(Path(path).resolve())]
    start=time.monotonic(); utc=time.time()
    save(out/'evaluation-launch.json',dict(at_utc=now(),command=command,plan=ref(path),freeze=ref(out/'freeze.json'),
        deadline_TERM_unix=utc+30,deadline_KILL_unix=utc+31,no_automatic_retry=True))
    with (out/'evaluation.stdout').open('xb') as stdout,(out/'evaluation.stderr').open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True); timed_out=False
        try: rc=child.wait(timeout=max(.001,start+30-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=max(.001,start+31-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        for stream in (stdout,stderr):stream.flush();os.fsync(stream.fileno())
    save(out/'evaluation-terminal.json',dict(returncode=rc,timed_out=timed_out,seconds=time.monotonic()-start,
        unique_worker=True,budget_seconds=30,grace_seconds=1,plan=ref(path),freeze=ref(out/'freeze.json')))
    assert rc==0 and not timed_out
    print(json.dumps(ref(out/'result.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['assemble','evaluate','worker']);ap.add_argument('--plan',required=True)
    args=ap.parse_args();{'assemble':assemble,'evaluate':evaluate,'worker':worker}[args.mode](args.plan)
