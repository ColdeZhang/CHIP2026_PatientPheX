#!/usr/bin/env python3
"""Exactly one literal three-predictor union per fold; joint freeze before two v2 calls."""
import argparse,copy,hashlib,json,os,re,signal,sys,time
from datetime import datetime,timezone
from pathlib import Path
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def check(item):
 actual=ref(item['path']);assert actual==item;return actual
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]
def publish(p,raw):
 p=Path(p);tmp=p.with_name(p.name+'.pending')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink()
 for directory in [p.parent,p.parent.parent]:
  fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
def save(p,obj):publish(p,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())

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

def relations(doc):
    result = {}
    for association in doc['association']:
        pid = association['patient_id']; assert pid not in result
        result[pid] = {part for value in association['phenotype']
                       for part in (value.split(';') if COMPOUND.fullmatch(value) else [value])}
    return result

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

def main():
 if not __debug__:raise RuntimeError('Assertions must stay enabled')
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);args=ap.parse_args();b=args.directory.resolve()
 started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60 second CPU budget exceeded')));signal.alarm(60)
 plan=read(b/'plan.json');entry=read(b/'source-plan-freeze.json')
 assert entry['source']==ref(__file__) and entry['plan']==ref(b/'plan.json') and plan['source']==ref(__file__)
 assert plan['schema']=='fixed-old-reverse-patient-triple-union-v1' and plan['fold_order']==['F1','F0']
 assert plan['cpu_cap_seconds']==60 and plan['exactly_two_new_evaluator_calls']
 assert plan['keep_rule']=='both folds Score > old_union Score and both folds macF1 >= old_union macF1'
 assert plan['posthoc_after_existing_component_results'] and not plan['original_prequality_registration']
 for item in plan['common_references'].values():check(item)
 for fold in plan['fold_order']:
  for item in plan['folds'][fold]['references'].values():check(item)
 assert not (b/'assembled').exists() and not (b/'evaluation').exists()
 save(b/'started.json',dict(at_utc=now(),source=ref(__file__),plan=ref(b/'plan.json'),entry=ref(b/'source-plan-freeze.json'),gold_read=False))
 predictions={};reports={};prior={}
 (b/'assembled').mkdir()
 for fold in plan['fold_order']:
  spec=plan['folds'][fold];refs=spec['references'];expected=spec['expected']
  original_plan=read(refs['patient_endpoint_plan']['path']);assert original_plan['gold']==spec['gold']
  for name,original_name in [('old','baseline_pred'),('input','input'),('prompts','prompts')]:assert refs[name]==original_plan['references'][original_name]
  reverse_plan=read(refs['reverse_plan']['path']);assert reverse_plan['local_references']['baseline']==refs['old'] and reverse_plan['local_references']['input']==refs['input']
  reverse_summary=read(refs['reverse_summary']['path']);assert reverse_summary['recovery']['assembled/union.jsonl']==refs['old_union']
  oldmetrics=read(refs['old_union_metrics']['path']);assert oldmetrics['metrics']==reverse_summary['result']['arms']['union']['metrics']
  rulesmetrics=read(refs['rules_metrics']['path']);assert rulesmetrics['prediction_sha256']==refs['rules']['sha256'] and rulesmetrics['gold_sha256']==spec['gold']['sha256']
  for metric in [oldmetrics,rulesmetrics]:assert metric['evaluator_version']=='patientphex-public-approx-v2' and metric['configuration']['body_scope']=='all' and metric['configuration']['unlinked_document_mode']=='text'
  patientfreeze=read(refs['patient_freeze']['path'])
  assert (patientfreeze['prediction'] if fold=='F1' else patientfreeze['predictions']['primary'])==refs['patient']
  summary=read(refs['patient_data_summary']['path']);train=set(summary['training']['document_ids']);held=set(summary[fold+'_input_only']['document_ids'])
  assert len(train)==expected['train_documents'] and len(held)==16 and not train&held
  assert not summary[fold+'_input_only']['targets_read'] and summary[fold+'_input_only']['answer_fields']==0
  prompts=rows(refs['prompts']['path']);inputs=rows(refs['input']['path'])
  assert len(prompts)==expected['rows'] and all('answer' not in p for p in prompts)
  assert {str(p['pmc_id']) for p in prompts}=={str(d['pmc_id']) for d in inputs}==held
  assert sum(len(p['code_map']) for p in prompts)==expected['candidates']
  pool={(str(p['pmc_id']),str(pid),concept) for p in prompts for pid in p['patient_ids'] for concept in p['code_map'].values()}
  assert len(pool)==sum(len(p['code_map'])*len(p['patient_ids']) for p in prompts)==expected['pairs']
  docs={name:rows(refs[name]['path']) for name in ['old','reverse','patient','old_union','rules']}
  for name,value in docs.items():
   assert len(value)==16 and [d['pmc_id'] for d in value]==[d['pmc_id'] for d in inputs]
   assert sum(len(d['entities']) for d in value)==expected['entities'] and sum(len(d['association']) for d in value)==expected['patients']
   for baseline,current in zip(docs['old'],value):
    assert {k:v for k,v in baseline.items() if k!='association'}=={k:v for k,v in current.items() if k!='association'}
    assert [{k:v for k,v in a.items() if k!='phenotype'} for a in baseline['association']]==[{k:v for k,v in a.items() if k!='phenotype'} for a in current['association']]
   pairs={(str(d['pmc_id']),str(a['patient_id']),h) for d in value for a in d['association'] for h in a['phenotype']}
   assert pairs<=pool
  oldreverse,first=build_union(docs['old'],docs['reverse'],prompts)
  assert all({pid:values for pid,values in relations(a).items()}=={pid:values for pid,values in relations(b).items()} for a,b in zip(oldreverse,docs['old_union'])), 'Existing baseline relation sets must equal exact first two members; historical list order is preserved only in its frozen file'
  triple,second=build_union(oldreverse,docs['patient'],prompts)
  assert second['removed_from_old']==0 and second['pool_pairs']==expected['pairs']
  path=b/'assembled'/(fold+'.triple.jsonl');publish(path,''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in triple).encode())
  predictions[fold]=ref(path);prior[fold]=docs['old_union']
  reports[fold]=dict(contract=expected,old_reverse=first,patient_added_to_old_reverse=second,training_vs_heldout_overlap=0,metadata_exact=True,all_positive_keys_in_pool=True,old_union_pred_exact=True,patient_F0_abstention_disclosure=plan['patient_F0_disclosure'] if fold=='F0' else None)
 save(b/'assembled/contract.json',reports)
 save(b/'assembled/common-freeze.json',dict(at_utc=now(),source=ref(__file__),plan=ref(b/'plan.json'),entry=ref(b/'source-plan-freeze.json'),predictions=predictions,contract=ref(b/'assembled/contract.json'),both_folds_complete=True,new_triple_quality_calls=0,gold_opened_or_hashed_in_control=False,all_plan_nongold_references_verified=True))
 freeze=ref(b/'assembled/common-freeze.json')
 for item in predictions.values():check(item)
 (b/'evaluation').mkdir()
 save(b/'evaluation/before-gold.json',dict(at_utc=now(),common_freeze=freeze,predictions=predictions,both_folds_complete=True,gold_not_yet_read=True,exactly_two_calls_order=plan['fold_order'],baseline_evaluator_calls=0))
 common=plan['common_references'];sys.path.insert(0,str(Path(common['evaluator']['path']).parent))
 import evaluate as evaluator
 import validate_predictions as validator
 assert ref(evaluator.__file__)==common['evaluator'] and ref(validator.__file__)==common['validator']
 assert evaluator.VERSION=='patientphex-public-approx-v2'
 hpo=validator.read_hpo(common['hpo']['path']);assert hpo['version']=='hp/releases/2026-06-23'
 results={}
 for fold in plan['fold_order']:
  spec=plan['folds'][fold];refs=spec['references'];prediction=validator.load_jsonl(predictions[fold]['path']);inputs=validator.load_jsonl(refs['input']['path'])
  validation=validator.validate_predictions(inputs,prediction,hpo);save(b/'evaluation'/(fold+'.validation.json'),validation);assert validation['valid']
  check(spec['gold']);gold=validator.load_jsonl(spec['gold']['path'])
  metrics=evaluator.evaluate_report(gold,prediction,'all','text')
  metrics['gold_sha256']=spec['gold']['sha256'];metrics['prediction_sha256']=predictions[fold]['sha256'];save(b/'evaluation'/(fold+'.metrics.json'),metrics)
  old=read(refs['old_union_metrics']['path'])['metrics'];rules=read(refs['rules_metrics']['path'])['metrics']
  delta={k:v-old[k] for k,v in metrics['metrics'].items()};rulesdelta={k:v-rules[k] for k,v in metrics['metrics'].items()}
  assert delta['menF1']==delta['docF1']==rulesdelta['menF1']==rulesdelta['docF1']==0
  results[fold]=dict(metrics=metrics['metrics'],old_union=old,rules=rules,delta_vs_old_union=delta,delta_vs_rules=rulesdelta,changes_vs_old_union=changes(gold,prediction,prior[fold]),fold_keep=delta['Score']>0 and delta['macF1']>=0,old_metrics_reused_not_rescored=True)
 keep=all(results[f]['fold_keep'] for f in plan['fold_order'])
 save(b/'evaluation/results.json',dict(at_utc=now(),folds=results,keep=keep,keep_rule=plan['keep_rule'],status='pass_fixed_twofold_gate_pending_review_no_dev' if keep else 'negative_closed_no_scan',common_freeze=freeze,before_gold=ref(b/'evaluation/before-gold.json'),source=ref(__file__),plan=ref(b/'plan.json'),seconds=time.perf_counter()-started,new_evaluator_calls=2,old_evaluator_calls=0,new_training_or_GPU=False,posthoc_after_existing_component_results=True,original_strict_F0_failure_preserved=True,patient_F0_disclosure=plan['patient_F0_disclosure']))
 print(json.dumps(read(b/'evaluation/results.json'),ensure_ascii=False))
if __name__=='__main__':main()
