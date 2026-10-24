#!/usr/bin/env python3
"""One originaldev64 a179 literal-union endpoint. Never score patient-alone."""
import argparse
import ast
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
from final_parser import parse_final

POLICY_SHA = 'faf149351c65e4f6272bd07ce2a2a5a632eeb18bf50d06c926a9e34321b38419'
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
METRICS = ('menF1','docF1','micF1','macF1','Score')

def ref(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def check_ref(item,local=None):
    actual=ref(local or item['path'])
    assert all(actual[k]==item[k] for k in ('bytes','sha256')),item
    return actual

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def now():return datetime.now(timezone.utc).isoformat()

def publish(path,raw):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    os.link(tmp,path);tmp.unlink()
    for directory in (path.parent,path.parent.parent):
        descriptor=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

def save(path,value):publish(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())

# Byte-identical preservation helpers from audited reverse-dev finish39c3.
def values(v):
 assert isinstance(v,str)
 return v.split(';') if COMPOUND.fullmatch(v) else [v]

def relations(doc):
 out={}
 for a in doc['association']:
  assert a['patient_id'] not in out
  out[a['patient_id']]={v for x in a['phenotype'] for v in values(x)}
 return out

def union(old,new):
 assert old['pmc_id']==new['pmc_id']
 assert {k:v for k,v in old.items() if k!='association'}=={k:v for k,v in new.items() if k!='association'}
 before,extra=relations(old),relations(new);assert set(before)==set(extra)
 result=copy.deepcopy(old);log=[]
 for a in result['association']:
  pid=a['patient_id'];added=sorted(extra[pid]-before[pid]);a['phenotype'].extend(added)
  log.extend(dict(pmc_id=old['pmc_id'],patient_id=pid,phenotype=v) for v in added)
 assert relations(result)=={p:before[p]|extra[p] for p in before}
 for a,z in zip(old['association'],result['association']):
  assert z['phenotype'][:len(a['phenotype'])]==a['phenotype']
  assert {k:v for k,v in a.items() if k!='phenotype'}=={k:v for k,v in z.items() if k!='phenotype'}
 return result,log

def pair_set(docs):
    return {(str(d['pmc_id']),str(pid),h) for d in docs for pid,hs in relations(d).items() for h in hs}

def bound_runner_contract(plan):
    """Only bind actual receipt constants, never silently alter generation math."""
    refs=plan['references'];replacement={
        'TRAINING_COMPLETE_SHA':refs['training_complete']['sha256'],
        'TRAINING_CHECKPOINT_SHA':refs['training_checkpoint']['sha256'],
        'TRAINING_DURABILITY_SHA':refs['training_durability']['sha256']}
    def normalized(path,actual):
        tree=ast.parse(Path(path).read_text());seen={}
        if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):tree.body.pop(0)
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                key=node.targets[0].id
                if key in replacement:
                    value=ast.literal_eval(node.value);seen[key]=value
                    if actual:assert value==replacement[key]
                    node.value=ast.Constant(value='__BOUND_RECEIPT__')
        assert set(seen)==set(replacement)
        return ast.dump(tree,include_attributes=False)
    assert normalized(refs['runner']['path'],True)==normalized(refs['runner_template']['path'],False)


def load_plan(b):
    plan=read(b/'endpoint-plan.json');policy=read(plan['policy']['path'])
    assert plan['status']=='BOUND_ACTUAL_TRAINING_AND_INFERENCE_PRE_QUALITY', 'Draft/pending endpoint cannot run'
    assert ref(__file__)['sha256']==plan['source_sha256']
    check_ref(plan['policy']);assert plan['policy']['sha256']==POLICY_SHA
    assert policy['schema']=='patient-first-full64-originaldev64-single-union-v1'
    assert policy['new_patient_first_dev_quality_read'] is False and policy['quality_calls']==1
    assert policy['candidate_rule']=='delta(Score)>0'
    refs=plan['references'];dynamic={'runner','training_complete','training_checkpoint','training_durability','training_collection'}
    assert set(refs)==set(policy['reference_sha256'])|dynamic
    for key,digest in policy['reference_sha256'].items():assert refs[key]['sha256']==digest
    # Explicit nongold list; no model weights in this metadata inventory.
    for item in refs.values():
        assert isinstance(item.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',item['sha256'])
        assert Path(item['path']).resolve()!=Path(plan['gold']['path']).resolve()
        check_ref(item)
    for key,fn in [('codec',classify_final),('schema',validate_input),('strict_parser',parse_final)]:
        assert ref(sys.modules[fn.__module__].__file__)['sha256']==refs[key]['sha256']
    for key in ['inference_plan','inference_job']:check_ref(plan[key])
    run=read(plan['inference_plan']['path']);job=read(plan['inference_job']['path'])
    assert run['phase']=='infer' and run['cap_seconds']==900 and run['grace_seconds']==30
    assert job['result']['id'] and job['cap_seconds']==900 and job['grace_seconds']==30
    for key,name in [('runner','infer.py'),('codec','limited_codec.py'),('schema','patient_first.py'),('raw_store','raw_store.py'),('driver','formal_driver.py'),('operations','formal_operations.py')]:
        matches=[v for k,v in run['files'].items() if Path(k).name==name]
        assert len(matches)==1 and all(matches[0][k]==refs[key][k] for k in ('bytes','sha256'))
    train=read(refs['training_complete']['path']);checkpoint=read(refs['training_checkpoint']['path']);durable=read(refs['training_durability']['path'])
    assert train['epochs']==2 and train['global_step']==68 and not train['smoke_only']
    assert train['source_sha256']==checkpoint['fingerprint']['source_sha256']==refs['trainer']['sha256']
    assert checkpoint['completed_epochs']==2 and checkpoint['global_step']==68
    assert checkpoint['fingerprint']['train_sha256']=='4fee5f9c3817730c08673bdd26bf5127d7cd7856d37d3f9082554f5db755c882'
    assert durable['status']=='PASS_file_and_directory_fsync_rename_and_reopened_SHA'
    assert durable['completed_epochs']==2 and durable['global_step']==68 and durable['source_sha256']==refs['trainer']['sha256']
    assert durable['checkpoint_metadata_sha256']==refs['training_checkpoint']['sha256']
    assert durable['file_directory_fsync'] and durable['reopened_SHA_all_exact'] and durable['all_members_reopened_and_SHA_verified']
    assert set(durable['files'])==set(checkpoint['files'])|{'checkpoint.json'}
    assert {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(checkpoint['files'])
    for name,digest in checkpoint['files'].items():assert durable['files'][name]['sha256']==digest
    assert durable['files']['checkpoint.json']['sha256']==refs['training_checkpoint']['sha256']
    tc=read(refs['training_collection']['path']);tw=tc['wrapper']
    assert tc['phase']=='train' and tw['returncode']==0 and not tw.get('endpoint_error')
    assert tw['all_epochs_persistent_and_readback_verified'] and tw['no_SHM_or_pending_export']
    for name,key in [('training_complete.json','training_complete'),('epoch-002/checkpoint.json','training_checkpoint'),('durability-epoch-002.json','training_durability')]:
        assert all(tc['files'][name][k]==refs[key][k] for k in ('bytes','sha256'))
    assert tw['training_complete_sha256']==refs['training_complete']['sha256']
    assert tw['final_checkpoint_metadata_sha256']==refs['training_checkpoint']['sha256']
    assert tw['durability']['2']['receipt_sha256']==refs['training_durability']['sha256']
    assert tw['durability']['2']['receipt']==durable
    oldmetrics=read(refs['baseline_metrics']['path'])
    assert oldmetrics['metrics']['Score']==0.7460819649861977
    assert plan['gold']==dict(path='/workspace/patientphex-data/day1/dev16.gold.jsonl',bytes=443056,sha256='c1f0f24a58b62bb898eb672550dc58247ddb9a0e8768c6f899812cfc327cc050')
    bound_runner_contract(plan)
    return plan,policy,run


def verify_terminal(b,plan,run):
    terminal=b/'inference-terminal';collection=read(b/'inference-collection.json')
    assert collection['phase']=='infer' and collection['direct_existing_files_only']
    wrapper=read(terminal/'wrapper-complete.json')
    assert wrapper==collection['wrapper'] and wrapper['returncode']==0 and not wrapper.get('endpoint_error')
    assert wrapper['gpu_release']['target_has_no_process'] and wrapper['gpu_release']['target']==run['gpu_uuid']
    assert wrapper['output_dir']==run['output_dir'] and wrapper['no_retry']
    assert wrapper['phase']=='infer' and wrapper['cap_seconds']==900 and wrapper['grace_seconds']==30
    refs={}
    for name,item in collection['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        refs[name]=check_ref(item,terminal/path)
    required={'driver-started.json','wrapper-complete.json','stdout.log','stderr.log','responses.jsonl','responses.manifest.json','responses.checkpoint.json','responses.loading.json','responses.decisions.json','responses.raw-index.json','complete.json'}
    assert required<=set(refs) and 'generation-failure.json' not in refs and len(refs)==203
    complete=read(terminal/'complete.json');manifest=read(terminal/'responses.manifest.json')
    assert complete['rows']==manifest['expected_rows']==64 and complete['complete_coverage'] and complete['no_quality']
    assert complete['cached_rows']==manifest['cached_rows']==0 and complete['new_rows']==manifest['new_rows']==64
    assert complete['responses_sha256']==manifest['output_sha256']==refs['responses.jsonl']['sha256']
    assert complete['manifest_sha256']==refs['responses.manifest.json']['sha256']
    assert complete['inference_plan_sha256']==plan['inference_plan']['sha256']
    assert wrapper['inference_complete_sha256']==refs['complete.json']['sha256']
    assert read(terminal/'driver-started.json')['plan_sha256']==plan['inference_plan']['sha256']
    assert manifest['all64_raw_saved_before_codec'] and manifest['all_rows_fresh'] and manifest['no_cache_or_resume']
    assert not manifest['model_answer_repair'] and manifest['no_baseline_fallback']
    assert set(manifest['eos_token_ids'])==set(EOS_IDS)
    fp=manifest['fingerprint'];assert fp==read(terminal/'responses.checkpoint.json')
    for field,key in [('source_sha256','runner'),('limited_codec_sha256','codec'),('raw_store_sha256','raw_store'),('prompts_sha256','prompts'),('schema_sha256','schema'),('token_audit_sha256','input_token_audit'),('training_complete_sha256','training_complete'),('training_checkpoint_sha256','training_checkpoint'),('training_durability_sha256','training_durability'),('source_manifest_sha256','model_source_manifest')]:
        assert fp[field]==plan['references'][key]['sha256']
    checkpoint=read(plan['references']['training_checkpoint']['path'])
    for name in ['adapter_config.json','adapter_model.safetensors']:assert fp['adapter_files'][name]==checkpoint['files']['adapter/'+name]
    assert fp['configuration']['max_input_tokens']==14000 and fp['configuration']['max_new_tokens']==1600
    assert fp['batch_size']==1 and fp['thinking'] is False
    remote=Path(run['output_dir'])
    def local_remote(item,expected):
        relative=Path(item['path']).relative_to(remote)
        assert '..' not in relative.parts and relative.as_posix()==expected
        actual=refs[expected];assert all(actual[k]==item[k] for k in ('bytes','sha256'))
        return terminal/relative
    for key,name in [('decisions','responses.decisions.json'),('raw_index','responses.raw-index.json')]:local_remote(manifest[key],name)
    raw=(terminal/'responses.jsonl').read_bytes();assert raw.endswith(b'\n')
    records=[json.loads(line) for line in raw.splitlines()];decisions=read(terminal/'responses.decisions.json');index=read(terminal/'responses.raw-index.json')
    assert len(records)==len({r['row_id'] for r in records})==len(decisions)==len(index)==64
    assert [d['row_id'] for d in decisions]==[r['row_id'] for r in records]
    for record,decision,item in zip(records,decisions,index):
        basename=hashlib.sha256(record['row_id'].encode()).hexdigest()+'.json'
        raw_name='raw-rows/'+basename;decision_name='decision-rows/'+basename;generation_name='generation-rows/'+basename
        payload=read(local_remote(item,raw_name));assert payload['record']==record and payload['raw_before_codec']
        assert payload['origin']=='new_generation_all64'
        dp=read(terminal/decision_name)
        assert dp['row_id']==record['row_id'] and dp['decision']=={k:v for k,v in decision.items() if k!='row_id'} and dp['raw']==item
        generation=read(terminal/generation_name)
        for key in ['row_id','pmc_id','generated_token_ids','input_tokens','output_tokens','hit_token_limit']:assert generation[key]==record[key]
        actual_eos=bool(record['generated_token_ids'] and record['generated_token_ids'][-1] in EOS_IDS)
        assert generation['actual_last_token_eos']==actual_eos and generation['max_new_tokens']==1600
        assert generation['actual_terminal_eos_id']==(record['generated_token_ids'][-1] if actual_eos else None)
        assert generation['decoded_or_parsed'] is False
        required|={raw_name,decision_name,generation_name}
    assert set(refs)==required and len(required)==203
    assert complete['codec_status_counts']==manifest['codec_status_counts']
    return records,decisions,manifest,refs


def assemble(b,plan,policy,run,started):
    assert not (b/'assembled').exists() and not (b/'evaluation').exists()
    records,stored,manifest,terminal_refs=verify_terminal(b,plan,run)
    refs=plan['references'];prompts=rows(refs['prompts']['path']);by_id={p['row_id']:p for p in prompts}
    assert len(prompts)==len(by_id)==64 and set(by_id)=={r['row_id'] for r in records}
    audit={r['row_id']:r['patient_first'] for r in read(refs['input_token_audit']['path'])}
    assert set(audit)==set(by_id)
    assert [r['row_id'] for r in records]==[p['row_id'] for p in sorted(prompts,key=lambda p:(audit[p['row_id']]['input_tokens'],p['row_id']))]
    sys.path.insert(0,plan['lightweight_libraries'])
    import tokenizers
    assert tokenizers.__version__=='0.23.1'
    tokenizer=tokenizers.Tokenizer.from_file(refs['tokenizer']['path'])
    old=rows(refs['baseline_pred']['path']);inputs=rows(refs['input']['path'])
    assert len(old)==len(inputs)==16 and sum(len(d['entities']) for d in old)==1072 and sum(len(d['association']) for d in old)==36
    assert [d['pmc_id'] for d in old]==[d['pmc_id'] for d in inputs]
    assert {str(d['pmc_id']) for d in old}=={str(p['pmc_id']) for p in prompts}
    selected={str(d['pmc_id']):{str(a['patient_id']):set() for a in d['association']} for d in old}
    stats=Counter();replayed=[];pool=set();candidates=0;pairs=0
    for record,saved in zip(records,stored):
        prompt=by_id[record['row_id']];validate_input(prompt);assert 'answer' not in prompt
        assert set(prompt['patient_ids'])==set(selected[str(prompt['pmc_id'])])
        assert str(record['pmc_id'])==str(prompt['pmc_id']) and record['input_tokens']==audit[record['row_id']]['input_tokens']
        ids=record['generated_token_ids'];assert isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids)
        assert record['output_tokens']==len(ids)
        text_ids=ids[:-1] if ids and ids[-1] in EOS_IDS else ids
        assert tokenizer.decode(text_ids,skip_special_tokens=False)==record['response'], 'Complete raw ID decode differs'
        decision=classify_final(record,prompt);assert saved==dict(row_id=record['row_id'],**decision)
        try:
            strict=parse_final(record,prompt);strict_ok=True;strict_reason='complete_all_patient_keys'
        except (ValueError,TypeError,KeyError,AssertionError) as error:
            strict=None;strict_ok=False;strict_reason=type(error).__name__+': '+str(error)
        assert strict_ok==(decision['status']=='complete')
        if strict_ok:assert {(pid,cid) for cid,pids in strict.items() for pid in pids}=={tuple(x) for x in decision['selected_pairs']}
        stats['strict_valid_rows' if strict_ok else 'strict_invalid_rows']+=1
        observed=set(decision['observed_key_subset']);assert observed<=set(prompt['patient_ids'])
        chosen=decision['selected_pairs'];assert len(chosen)==len({tuple(x) for x in chosen})
        stats[decision['status']+'_rows']+=1;stats['patient_key_slots']+=len(prompt['patient_ids'])
        missing=len(prompt['patient_ids'])-len(observed)
        stats['abstained_patient_key_slots']+=missing;stats['abstained_candidate_pairs']+=missing*len(prompt['code_map'])
        stats['explicit_empty_patient_key_slots']+=sum(not any(pid==pair[0] for pair in chosen) for pid in observed)
        stats['explicit_positive_pairs']+=len(chosen)
        for pid,cid in chosen:
            assert pid in observed and cid in prompt['code_map']
            selected[str(prompt['pmc_id'])][pid].add(prompt['code_map'][cid])
        candidates+=len(prompt['code_map']);pairs+=len(prompt['code_map'])*len(prompt['patient_ids'])
        pool.update((str(prompt['pmc_id']),pid,concept) for pid in prompt['patient_ids'] for concept in prompt['code_map'].values())
        replayed.append(dict(row_id=prompt['row_id'],pmc_id=prompt['pmc_id'],strict_valid=strict_ok,strict_reason=strict_reason,**decision))
    assert candidates==466 and pairs==len(pool)==1006
    assert {status:stats[status+'_rows'] for status in ['complete','partial','abstain']}==manifest['codec_status_counts']
    full=[];added=[];explicit=set()
    for before in old:
        # Temporary projection is never saved or evaluated as a separate arm.
        fresh=copy.deepcopy(before)
        for association in fresh['association']:association['phenotype']=sorted(selected[str(fresh['pmc_id'])][str(association['patient_id'])])
        explicit|=pair_set([fresh]);combined,log=union(before,fresh);full.append(combined);added.extend(log)
    old_pairs=pair_set(old);output_pairs=pair_set(full)
    assert len(old_pairs)==391 and explicit<=pool and output_pairs==old_pairs|explicit
    assert output_pairs-old_pairs<=pool
    public={tuple(x) for x in read(refs['preservation_contract']['path'])['additional_relations']}
    assert len(public)==35 and public<=old_pairs and public<=output_pairs
    for before,after in zip(old,full):
        assert json.dumps({k:v for k,v in before.items() if k!='association'},ensure_ascii=False)==json.dumps({k:v for k,v in after.items() if k!='association'},ensure_ascii=False)
    out=b/'assembled';out.mkdir()
    publish(out/'union.jsonl',''.join(json.dumps(doc,ensure_ascii=False)+'\n' for doc in full).encode())
    save(out/'codec-replay.json',replayed);save(out/'added-relations.json',added)
    save(out/'assembly-report.json',dict(at_utc=now(),rows=64,documents=16,patients=36,entities=1072,candidates=466,pairs=1006,abstention_counts=dict(stats),all64_complete_raw_ID_decode=True,codec_replay_exact=True,strict_complete_rows=stats['strict_valid_rows'],all_rows_fresh=True,cached_rows=0,baseline_relations=391,union_relations=len(output_pairs),added_relations=len(added),removed_relations=0,preserved_public_new_relations=35,old_phenotype_list_prefix_and_association_metadata_exact=True,all_nonassociation_fields_serialized_bytes_exact=True,patient_alone_saved_or_scored=False,gold_read=False))
    artifacts={name:ref(out/name) for name in ['union.jsonl','codec-replay.json','added-relations.json','assembly-report.json']}
    save(out/'freeze.json',dict(frozen_at_utc=now(),source=ref(__file__),plan=ref(b/'endpoint-plan.json'),policy=plan['policy'],inference_plan=plan['inference_plan'],inference_job=plan['inference_job'],references=refs,inference_collection=ref(b/'inference-collection.json'),inference_terminal=terminal_refs,prediction=artifacts['union.jsonl'],artifacts=artifacts,only_quality_arm='a179_union_patient_first',full_prediction_complete=True,gold_read=False,new_quality_calls_before_freeze=0,seconds=time.monotonic()-started))
    print(json.dumps(ref(out/'freeze.json')))


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
    assert change['removed_tp']==change['removed_fp']==0
    return change


def evaluate(b,plan,policy,run,started):
    freeze_path=b/'assembled/freeze.json';frozen=read(freeze_path)
    assert frozen['source']==ref(__file__) and frozen['plan']==ref(b/'endpoint-plan.json')
    assert frozen['policy']==plan['policy'] and frozen['inference_plan']==plan['inference_plan'] and frozen['inference_job']==plan['inference_job']
    assert frozen['references']==plan['references'] and frozen['full_prediction_complete']
    assert frozen['only_quality_arm']=='a179_union_patient_first' and not frozen['gold_read'] and frozen['new_quality_calls_before_freeze']==0
    for item in [frozen['inference_collection'],*frozen['inference_terminal'].values(),*frozen['artifacts'].values()]:check_ref(item)
    assert frozen['prediction']==frozen['artifacts']['union.jsonl'] and Path(frozen['prediction']['path']).resolve()==b/'assembled/union.jsonl'
    out=b/'evaluation';out.mkdir(exist_ok=False)
    save(out/'before-gold.json',dict(at_utc=now(),freeze=ref(freeze_path),source=ref(__file__),plan=ref(b/'endpoint-plan.json'),policy=plan['policy'],full_prediction_complete=True,gold_not_yet_read=True,planned_new_calls=1,baseline_calls=0,quality_arm='a179_union_patient_first'))
    refs=plan['references'];sys.path.insert(0,str(Path(refs['evaluator']['path']).parent))
    import evaluate as evaluator
    import validate_predictions as validator
    assert ref(evaluator.__file__)['sha256']==refs['evaluator']['sha256']
    assert ref(validator.__file__)['sha256']==refs['validator']['sha256']
    assert evaluator.VERSION=='patientphex-public-approx-v2' and evaluator.COMPOUND_HPO.pattern==COMPOUND.pattern
    inputs=validator.load_jsonl(refs['input']['path']);hpo=validator.read_hpo(refs['hpo']['path'])
    assert hpo['version']=='hp/releases/2026-06-23'
    prediction=validator.load_jsonl(frozen['prediction']['path']);validation=validator.validate_predictions(inputs,prediction,hpo)
    save(out/'union.validation.json',validation);assert validation['valid']
    # First gold bytes read/hash, only after complete raw/pred freeze and marker.
    check_ref(plan['gold']);gold=validator.load_jsonl(plan['gold']['path'])
    assert {d['pmc_id'] for d in gold}=={d['pmc_id'] for d in inputs}
    baseline=read(refs['baseline_metrics']['path']);old=rows(refs['baseline_pred']['path'])
    metrics=evaluator.evaluate_report(gold,prediction,'all','text')
    save(out/'union.metrics.json',metrics)
    delta={key:metrics['metrics'][key]-baseline['metrics'][key] for key in METRICS}
    assert delta['menF1']==delta['docF1']==0
    candidate_best=delta['Score']>0;numeric_target_met=metrics['metrics']['Score']>=0.8
    result=dict(completed_at_utc=now(),quality_arm='a179_union_patient_first',metrics=metrics['metrics'],baseline_metrics=baseline['metrics'],baseline_reused_not_rescored=True,delta=delta,changes=changes(gold,prediction,old),candidate_best=candidate_best,candidate_rule=policy['candidate_rule'],macro_decreased=delta['macF1']<0,macro_acceptance_gate=False,numeric_target_met=numeric_target_met,target_accepted=False,target_status='PENDING_independent_format_provenance_reproducibility_review' if numeric_target_met else 'NOT_REACHED_0.8',target_rule=policy['target_rule'],prediction=frozen['prediction'],freeze=ref(freeze_path),before_gold=ref(out/'before-gold.json'),policy=plan['policy'],source=ref(__file__),new_evaluator_calls=1,baseline_evaluator_calls=0,patient_alone_quality_calls=0,no_extra_combination_or_threshold=True,seconds=time.monotonic()-started,scope='Previously used originaldev16, fixed original64 prompts and fresh full64 patient adapter; one pre-quality a179 literal union with missing-key abstention. Not blind validation.')
    save(out/'results.json',result);print(json.dumps(result))


def main():
    if not __debug__:raise RuntimeError('Do not disable contract assertions with Python -O')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['assemble','evaluate']);parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    b=args.directory.resolve();started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Single-union CPU endpoint budget exceeded')))
    signal.alarm(30 if args.mode=='assemble' else 60)
    plan,policy,run=load_plan(b)
    {'assemble':assemble,'evaluate':evaluate}[args.mode](b,plan,policy,run,started)

if __name__=='__main__':main()
