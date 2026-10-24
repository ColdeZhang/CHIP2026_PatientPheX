"""One prespecified B4 UNION all completion relations, with a unique bounded v2 call."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from common import (BASE_SCORE, BASE_SHA, GOLD, POLICY_SHA, all_patient_keys,
                    checked, guard, jsonl, load_plan, modules,
                    publish, read, ref, save)


def now(): return datetime.now(timezone.utc).isoformat()


def verify_normalized(plan):
    refs = plan['inputs']; norm = read(checked(refs['normalization']))
    assert norm['schema'] == 'grounded-held-normalization-complete-v1'
    assert norm['documents'] == 16 and norm['patients'] == 39
    assert norm['no_policy_or_threshold_changes'] and norm['no_quality'] and norm['no_fit']
    for item in norm['sources'].values(): checked(item)
    assert norm['policy_sha256'] == POLICY_SHA
    prep = read(checked(norm['preparation'])); ex = read(checked(norm['extraction_manifest']))
    assert prep['extraction_manifest'] == norm['extraction_manifest']
    assert prep['schema'] == 'grounded-held-normalization-preparation-v1' and prep['no_gold_fit_or_score']
    assert prep['input'] == refs['F1_input'] and prep['train32'] == refs['train32']
    assert prep['policy'] == refs['policy'] and prep['policy']['sha256'] == POLICY_SHA
    assert norm['input_sha256'] == refs['F1_input']['sha256']
    ex_plan = read(checked(ex['plan'])); assert ex_plan['inputs']['F1_input'] == refs['F1_input']
    assert ex_plan['inputs']['prompts'] == refs['prompts']
    for item in ex['sources'].values(): checked(item)
    assert ex['sources'] == ex_plan['sources']
    collection = read(checked(refs['collection']))
    assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
    for item in collection['files'].values(): checked(item)
    for key, name in [('complete','output/complete.json'),('responses','output/responses.jsonl'),('wrapper','wrapper-complete.json')]:
        assert ex_plan['inputs'][key] == collection['files'][name]
    assert collection['wrapper']['complete39_actual_members_verified']
    complete = read(checked(collection['files']['output/complete.json']))
    assert complete['rows'] == 39
    assert complete['status'] == 'COMPLETE_FRESH39_RAW_GENERATION_NO_QUALITY'
    assert complete['source']['sha256'] == ex['sources']['run.py']['sha256']
    assert ex['complete_original39'] and ex['documents'] == 16 and ex['patients'] == 39
    assert ex['EOS_rows'] == complete['EOS_rows'] and ex['cap_rows'] == complete['cap_rows']
    quotes = jsonl(checked(ex['quotes'])); decisions = read(checked(ex['decisions']))
    rows = jsonl(checked(norm['normalizations'])); assert len(quotes) == len(rows) == norm['records']
    reconstructed = []
    for d in decisions:
        for i, q in enumerate(d['decision']['anchors']):
            reconstructed.append(dict(row_id=d['row_id'], pmc_id=d['pmc_id'],
                target_patient_id=d['target_patient_id'], anchor_index=i, **q))
    assert quotes == reconstructed and len(decisions) == 39
    assert norm['quotes_sha256'] == ex['quotes']['sha256']
    for q, row in zip(quotes, rows):
        assert all(row[k] == value for k, value in q.items())
        assert row['policy_sha256'] == POLICY_SHA and row['normalization_policy'] == 'grounded-public-sap-conservative-v1'
    if prep['necessary_public_query_count']:
        c = read(checked(norm['query_collection'])); query = read(checked(norm['query_complete']))
        for item in c['files'].values(): checked(item)
        assert c['all_actual_SHA_verified'] and c['wrapper']['returncode'] == 0 and not c['wrapper']['timed_out']
        checked(norm['query_scores']); checked(norm['semantic_evidence'])
        assert query['queries']['sha256'] == prep['queries']['sha256'] and query['scores']['sha256'] == norm['query_scores']['sha256']
        assert query['policy']['sha256'] == POLICY_SHA and query['query_count'] == prep['necessary_public_query_count']
        assert query['public_bank_rows'] == 41099 and query['no_bank_reencoding'] and query['no_fit']
    else: assert all(norm[k] is None for k in ['query_collection','query_complete','query_scores'])
    return rows, decisions


def assemble(path):
    state = guard(); plan = load_plan(path, 'quality'); refs = plan['inputs']
    assert plan['formula'] == 'B4 UNION all completion relations' and plan['new_v2_max'] == 1 and plan['old_v2'] == 0
    assert plan['baseline_score'] == BASE_SCORE and plan['expected_baseline_relations'] == 524
    H, A, _, schema, strict, _ = modules()
    for name, item in refs.items():
        if name != 'gold': checked(item)
    assert refs['baseline']['sha256'] == BASE_SHA
    lines, baseline, ids = H.rows(refs['baseline']['path'])
    _, inputs, input_ids = H.rows(refs['F1_input']['path'])
    assert ids == input_ids and len(ids) == 16
    assert sum(len(d['entities']) for d in baseline) == 1189
    patients = all_patient_keys(inputs)
    assert sum(len(d['association']) for d in baseline) == 39
    for doc, inp in zip(baseline, inputs):
        assert [a['patient_id'] for a in doc['association']] == [p['patient_id'] for p in inp['patient']]
    normalized, decisions = verify_normalized(plan)
    hpo = strict.HPO(refs['hpo']['path'])
    class BoundStrict:
        HPO = hpo
        verify_span = staticmethod(strict.verify_span)
    assert {(d['pmc_id'], d['target_patient_id']) for d in decisions} == patients
    selected, records, counts = A.new_relations(normalized, inputs, BoundStrict, schema, POLICY_SHA)
    assert set(selected) == patients
    consumption = dict(records=records, counts=counts, all39_decisions=True)
    completion_keys = {(did,pid,*key) for (did,pid), values in selected.items() for key in values}
    old = H.relation_keys(baseline, hpo); assert len(old) == 524
    added = completion_keys - old
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
    assert H.relation_keys(pred, hpo) == old | completion_keys
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
    save(out/'all-completion-keys.json', sorted(completion_keys))
    save(out/'all-and-additions.json', dict(all_completion=sorted(completion_keys), added=sorted(added)))
    assert not state['F1_quality_enabled']
    save(out/'freeze.json', dict(schema='quote-completion-one-complete-endpoint-freeze-v1', at_utc=now(),
        plan=ref(path), sources=plan['sources'], formula=plan['formula'], prediction=ref(out/'prediction.jsonl'),
        format=ref(out/'format.json'), consumption=ref(out/'consumption.json'), all_completion_keys=ref(out/'all-completion-keys.json'),
        contributions=ref(out/'all-and-additions.json'), normalization=refs['normalization'],
        source_relations=len(completion_keys), added=len(added), final_relations=len(old|completion_keys),
        baseline_relations=524, documents=16, patients=39, entities=1189, complete_all39=True,
        nonphenotype_bytes_exact=True, baseline_relation_lists_prefix_exact=True,
        no_added_prediction_byteexact=not added, gold_opened_or_hashed=False, new_v2=0, old_v2=0))
    print(json.dumps(ref(out/'freeze.json')))


def check_freeze(path, plan):
    out = Path(plan['output_directory']); f = read(out/'freeze.json')
    assert f['plan'] == ref(path) and f['sources'] == plan['sources'] and f['formula'] == plan['formula']
    assert f['complete_all39'] and f['nonphenotype_bytes_exact'] and f['baseline_relation_lists_prefix_exact']
    assert not f['gold_opened_or_hashed'] and f['new_v2'] == f['old_v2'] == 0
    for k in ['prediction','format','consumption','all_completion_keys','contributions','normalization']: checked(f[k])
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
    assert baseline_plan['refs']['gold'] == GOLD and baseline_plan['refs']['input'] == refs['F1_input']
    if not frozen['added']:
        assert checked(frozen['prediction']).read_bytes() == checked(refs['baseline']).read_bytes()
        save(out/'result.json', dict(status='NO_ADDED_KEYS_BASE_BYTEEXACT_ZERO_V2', baseline=prior['metrics'],
            keep=False, new_v2=0, old_v2=0, freeze=ref(out/'freeze.json'), no_gold_read=True))
        return
    validator = H.load_module('validate_predictions', refs['validator'])
    evaluator = H.load_module('_quote_completion_public_v2', refs['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions and evaluator.VERSION == 'patientphex-public-approx-v2'
    assert not (out/'evaluation-intent.json').exists() and not (out/'metrics.json').exists()
    save(out/'evaluation-intent.json', dict(at_utc=now(), plan=ref(path), freeze=ref(out/'freeze.json'),
        prediction=frozen['prediction'], gold=GOLD, new_v2=1, old_v2=0))
    state['F1_quality_enabled'] = True
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
        all_completion_relations_used=True, no_other_arm_or_filter_scan=True)
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
