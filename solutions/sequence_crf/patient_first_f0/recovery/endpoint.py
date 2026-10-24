#!/usr/bin/env python3
"""One 23+35 recovery endpoint: explicit facts, common freeze, then two v2 calls."""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from limited_codec import classify_final, EOS_IDS
from patient_first import validate_input

POLICY_SHA = '8bd22643d1193a0641ac169bc986d304e8c695215d6ad891b3b752ca93c1ea1d'
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')


def ref(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def check_ref(item, local=None):
    actual=ref(local or item['path'])
    assert all(actual[k]==item[k] for k in ('bytes','sha256')), item
    return actual


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    publish(path,raw)


def publish(path,raw):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    os.link(tmp,path);tmp.unlink()
    for directory in (path.parent,path.parent.parent):
        descriptor=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def relations(doc):
    result = {}
    for association in doc['association']:
        pid = association['patient_id']; assert pid not in result
        result[pid] = {part for value in association['phenotype']
                       for part in (value.split(';') if COMPOUND.fullmatch(value) else [value])}
    return result


def build_union(old, new, prompts):
    pool = {(str(p['pmc_id']), str(pid), hpo) for p in prompts
            for pid in p['patient_ids'] for hpo in p['code_map'].values()}
    assert len(old) == len(new)
    result = copy.deepcopy(old)
    old_pairs, new_pairs, result_pairs = set(), set(), set()
    for before, current, output in zip(old, new, result):
        assert {k:v for k,v in before.items() if k != 'association'} == {
            k:v for k,v in current.items() if k != 'association'}
        assert len(before['association']) == len(current['association'])
        for a, b, c in zip(before['association'], current['association'], output['association']):
            assert {k:v for k,v in a.items() if k != 'phenotype'} == {
                k:v for k,v in b.items() if k != 'phenotype'}
            pmc, pid = str(before['pmc_id']), str(a['patient_id'])
            left, right = set(a['phenotype']), set(b['phenotype'])
            assert len(left) == len(a['phenotype']) and len(right) == len(b['phenotype'])
            c['phenotype'] = sorted(left | right)
            old_pairs.update((pmc,pid,h) for h in left)
            new_pairs.update((pmc,pid,h) for h in right)
            result_pairs.update((pmc,pid,h) for h in c['phenotype'])
    assert old_pairs <= pool and new_pairs <= pool
    assert result_pairs == old_pairs | new_pairs
    return result, dict(pool_pairs=len(pool), old_relations=len(old_pairs),
                       primary_relations=len(new_pairs), union_relations=len(result_pairs),
                       added_over_old=len(result_pairs-old_pairs), removed_from_old=0,
                       all_nonassociation_exact=True, all_patient_metadata_exact=True)


def load_plan(b):
    plan=read(b/'endpoint-plan.json');policy=read(b/'policy.json')
    assert ref(b/'policy.json')['sha256']==POLICY_SHA
    assert plan['source_sha256']==ref(__file__)['sha256']
    assert plan['policy']==ref(b/'policy.json')
    assert policy['schema']=='patient-first-F0-abstain-recovery-two-arms-v1'
    assert not policy['new_quality_read'] and policy['original_strict_failed_unscored']
    assert policy['contract']==dict(rows=58,documents=16,patients=47,entities=947,candidates=415,pairs=1166)
    assert plan['keep_rule']==policy['keep_rule']=='delta(micF1+macF1)>0 and delta(macF1)>=0'
    refs=plan['references']
    assert set(refs)==set(policy['reference_sha256'])|{'runner','raw_store','driver','operations'}
    for key,digest in policy['reference_sha256'].items():assert refs[key]['sha256']==digest
    for item in refs.values():check_ref(item)  # Explicit nongold list only.
    assert refs['codec']['sha256']==ref(sys.modules[classify_final.__module__].__file__)['sha256']
    assert refs['schema']['sha256']==ref(sys.modules[validate_input.__module__].__file__)['sha256']
    for name,filename in [('recovery_protocol','protocol.json'),('inference_job','infer-job.json')]:
        check_ref(plan[name]);assert Path(plan[name]['path']).resolve()==b/filename
    protocol=read(plan['recovery_protocol']['path']);job=read(plan['inference_job']['path'])
    assert protocol['status']=='frozen_pre_new_F0_quality' and not protocol['new_F0_quality_read']
    assert protocol['cached_rows']==23 and protocol['new_rows']==35
    assert protocol['old_endpoint_plan']==refs['original_endpoint_plan'] and protocol['old_policy']==refs['original_policy']
    assert protocol['raw_cache']==refs['cached_raw23'] and protocol['codec_source']==refs['codec']
    assert protocol['keep_rule']==plan['keep_rule'] and protocol['full58_raw_and_common_freeze_before_exactly_two_v2']
    assert job['result']['id'] and job['cap_seconds']==600 and job['grace_seconds']==30
    check_ref(plan['inference_plan'])
    assert Path(plan['inference_plan']['path']).resolve()==b/'infer-plan.json'
    run=read(b/'infer-plan.json')
    assert run['phase']=='infer' and run['cap_seconds']==600 and run['grace_seconds']==30
    for key,filename in [('runner','recover_infer.py'),('codec','limited_codec.py'),('raw_store','raw_store.py'),('driver','driver.py'),('operations','operations.py')]:
        matches=[v for k,v in run['files'].items() if Path(k).name==filename]
        assert len(matches)==1 and all(matches[0][k]==refs[key][k] for k in ('bytes','sha256'))
    original=read(refs['original_endpoint_plan']['path'])
    assert plan['gold']==original['gold']  # Locator only; gold bytes remain unopened.
    assert all(original['references'][k]==refs[k] for k in original['references'])
    assert read(refs['original_inference_collection']['path'])['wrapper']['returncode']!=0
    return plan,policy,run


def verify_terminal(b,plan,run):
    terminal=b/'inference-terminal';collection=read(b/'inference-collection.json')
    assert collection['phase']=='infer' and collection['direct_existing_files_only']
    wrapper=read(terminal/'wrapper-complete.json')
    assert wrapper==collection['wrapper'] and wrapper['returncode']==0 and not wrapper.get('endpoint_error')
    assert wrapper['gpu_release']['target_has_no_process'] and wrapper['gpu_release']['target']==run['gpu_uuid']
    assert wrapper['output_dir']==run['output_dir']
    assert wrapper['phase']=='infer' and wrapper['cap_seconds']==600 and wrapper['grace_seconds']==30
    refs={}
    for name,item in collection['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        refs[name]=check_ref(item,terminal/path)
    required={'driver-started.json','wrapper-complete.json','stdout.log','stderr.log',
        'responses.jsonl','responses.manifest.json','responses.checkpoint.json','responses.loading.json',
        'responses.decisions.json','responses.raw-index.json','complete.json'}
    assert required<=set(refs) and 'generation-failure.json' not in refs
    complete=read(terminal/'complete.json');manifest=read(terminal/'responses.manifest.json')
    assert complete['rows']==manifest['expected_rows']==58 and complete['complete_coverage'] and complete['no_quality']
    assert complete['cached_rows']==manifest['cached_rows']==23 and complete['new_rows']==manifest['new_rows']==35
    assert complete['responses_sha256']==manifest['output_sha256']==refs['responses.jsonl']['sha256']
    assert complete['manifest_sha256']==refs['responses.manifest.json']['sha256']
    assert complete['inference_plan_sha256']==plan['inference_plan']['sha256']
    assert wrapper['inference_complete_sha256']==refs['complete.json']['sha256']
    assert read(terminal/'driver-started.json')['plan_sha256']==plan['inference_plan']['sha256']
    assert manifest['all58_raw_saved_before_codec'] and manifest['original_strict_failure_unchanged']
    assert not manifest['model_answer_repair'] and manifest['no_baseline_fallback']
    assert set(manifest['eos_token_ids'])==set(EOS_IDS)
    fp=manifest['fingerprint'];assert fp==read(terminal/'responses.checkpoint.json')
    for field,key in [('source_sha256','runner'),('limited_codec_sha256','codec'),('raw_store_sha256','raw_store'),
        ('cached_responses_sha256','cached_raw23'),('cached_fingerprint_sha256','cached_fingerprint'),
        ('prompts_sha256','prompts'),('schema_sha256','schema'),('token_audit_sha256','input_token_audit')]:
        assert fp[field]==plan['references'][key]['sha256']
    original_fp=read(plan['references']['cached_fingerprint']['path'])
    for key in ['prompts_sha256','training_complete_sha256','training_checkpoint_sha256','adapter_files','model_files']:
        assert fp[key]==original_fp[key]
    assert fp['configuration']['max_input_tokens']==14000 and fp['configuration']['max_new_tokens']==1600
    assert fp['batch_size']==1 and fp['thinking'] is False
    remote=Path(run['output_dir'])
    def local_remote(item,expected):
        relative=Path(item['path']).relative_to(remote)
        assert '..' not in relative.parts and relative.as_posix()==expected
        actual=refs[expected];assert all(actual[k]==item[k] for k in ('bytes','sha256'))
        return terminal/relative
    for key,name in [('decisions','responses.decisions.json'),('raw_index','responses.raw-index.json')]:
        local_remote(manifest[key],name)
    raw=(terminal/'responses.jsonl').read_bytes();cached=Path(plan['references']['cached_raw23']['path']).read_bytes()
    assert len(cached)==8853 and cached.endswith(b'\n') and raw.startswith(cached) and raw.endswith(b'\n')
    records=[json.loads(line) for line in raw.splitlines()];oldrecords=[json.loads(line) for line in cached.splitlines()]
    assert len(records)==len({r['row_id'] for r in records})==58 and len(oldrecords)==23 and records[:23]==oldrecords
    decisions=read(terminal/'responses.decisions.json');index=read(terminal/'responses.raw-index.json')
    assert len(decisions)==len(index)==58
    assert [d['row_id'] for d in decisions]==[r['row_id'] for r in records]
    for position,(record,decision,item) in enumerate(zip(records,decisions,index)):
        basename=hashlib.sha256(record['row_id'].encode()).hexdigest()+'.json'
        raw_name='raw-rows/'+basename;decision_name='decision-rows/'+basename
        payload=read(local_remote(item,raw_name));assert payload['record']==record and payload['raw_before_codec']
        assert payload['origin']==('exact_original23_cache' if position<23 else 'new_generation_of_remaining35')
        dp=read(terminal/decision_name)
        assert dp['row_id']==record['row_id'] and dp['decision']=={k:v for k,v in decision.items() if k!='row_id'}
        assert dp['raw']==item
        required|={raw_name,decision_name}
        if position>=23:
            name='generation-rows/'+basename;required.add(name);generation=read(terminal/name)
            for key in ['row_id','pmc_id','generated_token_ids','input_tokens','output_tokens','hit_token_limit']:
                assert generation[key]==record[key]
            actual_eos=bool(record['generated_token_ids'] and record['generated_token_ids'][-1] in EOS_IDS)
            assert generation['actual_last_token_eos']==actual_eos and generation['max_new_tokens']==1600
            assert generation['actual_terminal_eos_id']==(record['generated_token_ids'][-1] if actual_eos else None)
            assert generation['decoded_or_parsed'] is False
    assert set(refs)==required, 'No missing/extra collected raw/decision/generation member'
    assert complete['codec_status_counts']==manifest['codec_status_counts']
    return records,decisions,manifest,refs


def assemble(b,plan,policy,run,started):
    assert not (b/'assembled').exists() and not (b/'evaluation').exists()
    records,stored,manifest,terminal_refs=verify_terminal(b,plan,run)
    refs=plan['references'];prompts=rows(refs['prompts']['path']);by_id={p['row_id']:p for p in prompts}
    assert len(prompts)==len(by_id)==58 and set(by_id)=={r['row_id'] for r in records}
    audit={r['row_id']:r['patient_first'] for r in read(refs['input_token_audit']['path'])}
    assert set(audit)==set(by_id)
    order=sorted(prompts,key=lambda p:(audit[p['row_id']]['input_tokens'],p['row_id']))
    assert [r['row_id'] for r in records]==[p['row_id'] for p in order]
    sys.path.insert(0,plan['lightweight_libraries'])
    import tokenizers
    assert tokenizers.__version__=='0.23.1'
    tokenizer=tokenizers.Tokenizer.from_file(refs['tokenizer']['path'])
    old=rows(refs['baseline_pred']['path']);inputs=rows(refs['input']['path'])
    assert len(old)==len(inputs)==16 and sum(len(d['entities']) for d in old)==947
    assert sum(len(d['association']) for d in old)==47
    assert [d['pmc_id'] for d in old]==[d['pmc_id'] for d in inputs]
    primary=copy.deepcopy(old)
    selected={str(d['pmc_id']):{str(a['patient_id']):set() for a in d['association']} for d in old}
    stats=Counter();replayed=[];pool=set();candidates=0;pairs=0
    for position,(record,saved) in enumerate(zip(records,stored)):
        prompt=by_id[record['row_id']];validate_input(prompt);assert 'answer' not in prompt
        assert str(record['pmc_id'])==str(prompt['pmc_id']) and record['input_tokens']==audit[record['row_id']]['input_tokens']
        ids=record['generated_token_ids'];assert isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids)
        assert record['output_tokens']==len(ids)
        text_ids=ids[:-1] if ids and ids[-1] in EOS_IDS else ids
        assert tokenizer.decode(text_ids,skip_special_tokens=False)==record['response'], 'Complete raw ID decode differs'
        decision=classify_final(record,prompt);assert saved==dict(row_id=record['row_id'],**decision)
        if position<23:assert decision['status']=='complete'
        observed=set(decision['observed_key_subset']);assert observed<=set(prompt['patient_ids'])
        selected_pairs=decision['selected_pairs'];assert len(selected_pairs)==len({tuple(x) for x in selected_pairs})
        stats[decision['status']+'_rows']+=1;stats['patient_key_slots']+=len(prompt['patient_ids'])
        missing=len(prompt['patient_ids'])-len(observed)
        stats['abstained_patient_key_slots']+=missing;stats['abstained_candidate_pairs']+=missing*len(prompt['code_map'])
        stats['explicit_empty_patient_key_slots']+=sum(not any(pid==pair[0] for pair in selected_pairs) for pid in observed)
        stats['explicit_positive_pairs']+=len(selected_pairs)
        for pid,cid in selected_pairs:
            assert pid in observed and cid in prompt['code_map']
            selected[str(prompt['pmc_id'])][pid].add(prompt['code_map'][cid])
        candidates+=len(prompt['code_map']);pairs+=len(prompt['code_map'])*len(prompt['patient_ids'])
        pool.update((str(prompt['pmc_id']),pid,concept) for pid in prompt['patient_ids'] for concept in prompt['code_map'].values())
        replayed.append(dict(row_id=prompt['row_id'],pmc_id=prompt['pmc_id'],**decision))
    assert candidates==415 and pairs==len(pool)==1166
    assert {status:stats[status+'_rows'] for status in ['complete','partial','abstain']}==manifest['codec_status_counts']
    for before,after in zip(old,primary):
        for association in after['association']:
            association['phenotype']=sorted(selected[str(after['pmc_id'])][str(association['patient_id'])])
        assert {k:v for k,v in before.items() if k!='association'}=={k:v for k,v in after.items() if k!='association'}
        assert [{k:v for k,v in a.items() if k!='phenotype'} for a in before['association']]==[{k:v for k,v in a.items() if k!='phenotype'} for a in after['association']]
    secondary,union_contract=build_union(old,primary,prompts)
    assert union_contract['pool_pairs']==1166
    out=b/'assembled';out.mkdir()
    for arm,docs in [('primary',primary),('secondary',secondary)]:
        publish(out/(arm+'.jsonl'),''.join(json.dumps(doc,ensure_ascii=False)+'\n' for doc in docs).encode())
    save(out/'codec-replay.json',replayed)
    save(out/'assembly-report.json',dict(at_utc=now(),rows=58,documents=16,patients=47,entities=947,candidates=415,pairs=1166,
        abstention_counts=dict(stats),union=union_contract,all58_complete_raw_decode=True,codec_replay_exact=True,
        original23_exact=True,new_raw35=True,model_answer_lists_invented=False,no_baseline_fallback_primary=True,
        projection='Only positive selected_pairs; empty output phenotype means no selected relation, not a fabricated model []',gold_read=False))
    artifacts={name:ref(out/name) for name in ['primary.jsonl','secondary.jsonl','codec-replay.json','assembly-report.json']}
    save(out/'common-freeze.json',dict(frozen_at_utc=now(),source=ref(__file__),plan=ref(b/'endpoint-plan.json'),
        policy=ref(b/'policy.json'),inference_plan=plan['inference_plan'],recovery_protocol=plan['recovery_protocol'],
        inference_job=plan['inference_job'],references=refs,
        inference_collection=ref(b/'inference-collection.json'),inference_terminal=terminal_refs,
        predictions={'primary':artifacts['primary.jsonl'],'secondary':artifacts['secondary.jsonl']},artifacts=artifacts,
        both_fullpred_complete=True,gold_read=False,new_quality_calls_before_freeze=0,
        original_strict_failure_unscored_unchanged=True,seconds=time.monotonic()-started))
    print(json.dumps(ref(out/'common-freeze.json')))


def changes(gold,prediction,old):
    goldmap={d['pmc_id']:relations(d) for d in gold};oldmap={d['pmc_id']:relations(d) for d in old}
    change=dict(added_tp=0,added_fp=0,removed_tp=0,removed_fp=0)
    for doc in prediction:
        current=relations(doc);prior=oldmap[doc['pmc_id']];truth=goldmap[doc['pmc_id']]
        assert set(current)==set(prior)
        for pid in current:
            added=current[pid]-prior[pid];removed=prior[pid]-current[pid]
            change['added_tp']+=len(added&truth[pid]);change['added_fp']+=len(added-truth[pid])
            change['removed_tp']+=len(removed&truth[pid]);change['removed_fp']+=len(removed-truth[pid])
    return change


def evaluate(b,plan,policy,run,started):
    common_path=b/'assembled/common-freeze.json';common=read(common_path)
    assert common['source']==ref(__file__) and common['plan']==ref(b/'endpoint-plan.json')
    assert common['policy']==ref(b/'policy.json') and common['inference_plan']==plan['inference_plan']
    assert common['recovery_protocol']==plan['recovery_protocol'] and common['inference_job']==plan['inference_job']
    assert common['references']==plan['references'] and common['both_fullpred_complete']
    assert not common['gold_read'] and common['new_quality_calls_before_freeze']==0
    for item in [common['inference_collection'],*common['inference_terminal'].values(),*common['artifacts'].values()]:check_ref(item)
    for arm in ['primary','secondary']:
        assert common['predictions'][arm]==common['artifacts'][arm+'.jsonl']
        assert Path(common['predictions'][arm]['path']).resolve()==b/'assembled'/(arm+'.jsonl')
    out=b/'evaluation';out.mkdir(exist_ok=False)
    save(out/'before-gold.json',dict(at_utc=now(),common_freeze=ref(common_path),source=ref(__file__),
        plan=ref(b/'endpoint-plan.json'),policy=ref(b/'policy.json'),both_fullpred_complete=True,
        gold_not_yet_read=True,planned_new_calls=2,baseline_calls=0))
    refs=plan['references'];sys.path.insert(0,str(Path(refs['evaluator']['path']).parent))
    import evaluate as evaluator
    import validate_predictions as validator
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    inputs=validator.load_jsonl(refs['input']['path']);hpo=validator.read_hpo(refs['hpo']['path'])
    assert hpo['version']=='hp/releases/2026-06-23'
    predictions={arm:validator.load_jsonl(common['predictions'][arm]['path']) for arm in ['primary','secondary']}
    for arm,prediction in predictions.items():
        validation=validator.validate_predictions(inputs,prediction,hpo)
        save(out/(arm+'.validation.json'),validation);assert validation['valid']
    # First gold read/hash in this command, after both predictions and marker.
    check_ref(plan['gold']);gold=validator.load_jsonl(plan['gold']['path'])
    assert {d['pmc_id'] for d in gold}=={d['pmc_id'] for d in inputs}
    baseline=read(refs['baseline_metrics']['path']);old=rows(refs['baseline_pred']['path']);results={}
    for arm,prediction in predictions.items():
        metrics=evaluator.evaluate_report(gold,prediction,'all','text')
        save(out/(arm+'.metrics.json'),metrics)
        delta={key:metrics['metrics'][key]-value for key,value in baseline['metrics'].items()}
        assert delta['menF1']==delta['docF1']==0
        result=dict(completed_at_utc=now(),metrics=metrics['metrics'],baseline=baseline['metrics'],
            baseline_reused_not_rescored=True,delta=delta,changes=changes(gold,prediction,old),
            keep=delta['micF1']+delta['macF1']>0 and delta['macF1']>=0,keep_rule=plan['keep_rule'],
            prediction=common['predictions'][arm],common_freeze=ref(common_path),
            scope='One recovered F0 explicit-only primary or predeclared oldSFT48 union; original strict run remains failed and unscored')
        save(out/(arm+'.results.json'),result);results[arm]=result
    save(out/'results.json',dict(completed_at_utc=now(),arms=results,common_freeze=ref(common_path),
        before_gold=ref(out/'before-gold.json'),policy=ref(b/'policy.json'),source=ref(__file__),
        new_evaluator_calls=2,baseline_evaluator_calls=0,no_extra_combination_or_threshold=True,
        original_strict_unscored_failure_preserved=True,seconds=time.monotonic()-started))
    print(json.dumps(read(out/'results.json')))


def main():
    if not __debug__:raise RuntimeError('Do not disable contract assertions with Python -O')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['assemble','evaluate'])
    parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    b=args.directory.resolve();started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery CPU endpoint budget exceeded')))
    signal.alarm(30 if args.mode=='assemble' else 60)
    plan,policy,run=load_plan(b)
    {'assemble':assemble,'evaluate':evaluate}[args.mode](b,plan,policy,run,started)


if __name__=='__main__':main()
