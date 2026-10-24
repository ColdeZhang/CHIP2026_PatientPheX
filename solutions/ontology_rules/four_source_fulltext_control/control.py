"""One fixed two-fold/two-arm literal control; all four freeze before any gold."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
sys.dont_write_bytecode = True
ARMS = ['primary-four-source', 'diagnostic-without-old']
FOLDS = ['F1', 'F0']
KEEP_RULE = 'Each arm: both folds Score strictly above triple AND macF1 nondecreasing; among passing arms maximize summed two-fold Score delta; exact tie primary.'

def now(): return datetime.now(timezone.utc).isoformat()
def ref(path):
    p=Path(path);raw=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def verify(record):
    assert set(record)=={'path','bytes','sha256'} and ref(record['path'])==record
    return record
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]
def publish(path,raw):
    p=Path(path);temporary=p.with_name(p.name+'.pending')
    with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(temporary,p);temporary.unlink()
    for parent in [p.parent,p.parent.parent]:
        fd=os.open(str(parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def save(path,value):publish(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

def literal_map(documents):
    result={}
    for document in documents:
        pmc=str(document['pmc_id']);assert pmc not in result
        result[pmc]={}
        for association in document['association']:
            pid=str(association['patient_id']);assert pid not in result[pmc]
            values=association['phenotype']
            assert isinstance(values,list) and all(isinstance(x,str) for x in values)
            assert len(values)==len(set(values))
            result[pmc][pid]=set(values)
    return result
def pairs(documents):
    return {(pmc,pid,value) for pmc,patients in literal_map(documents).items()
            for pid,values in patients.items() for value in values}
def pool_from_prompts(prompts,inputs,expected):
    assert len(prompts)==expected['rows'] and len({p['row_id'] for p in prompts})==len(prompts)
    assert all('answer' not in p and [m['role'] for m in p['messages']]==['system','user'] for p in prompts)
    source_ids=[str(d['pmc_id']) for d in inputs]
    assert len(source_ids)==expected['documents']==len(set(source_ids))
    assert all(not {'entities','association'} & d.keys() for d in inputs)
    assert {str(p['pmc_id']) for p in prompts}==set(source_ids)
    patient_ids={str(d['pmc_id']):[str(p['patient_id']) for p in d['patient']] for d in inputs}
    assert sum(len(v) for v in patient_ids.values())==expected['patients']
    for values in patient_ids.values():assert len(values)==len(set(values))
    for prompt in prompts:
        assert [str(x) for x in prompt['patient_ids']]==patient_ids[str(prompt['pmc_id'])]
        assert isinstance(prompt['code_map'],dict) and prompt['code_map']
        assert all(isinstance(k,str) and isinstance(v,str) for k,v in prompt['code_map'].items())
    pool={(str(p['pmc_id']),str(pid),concept) for p in prompts for pid in p['patient_ids'] for concept in p['code_map'].values()}
    assert sum(len(p['code_map']) for p in prompts)==expected['candidates']
    assert len(pool)==sum(len(p['code_map'])*len(p['patient_ids']) for p in prompts)==expected['pairs']
    return pool,patient_ids
def validate_components(components,inputs,pool,patient_ids,expected):
    template=components['triple'];literal_map(template)
    assert len(template)==expected['documents']
    for name,documents in components.items():
        literal_map(documents)
        assert len(documents)==expected['documents']
        assert [str(d['pmc_id']) for d in documents]==[str(d['pmc_id']) for d in inputs]
        assert sum(len(d['entities']) for d in documents)==expected['entities']
        assert sum(len(d['association']) for d in documents)==expected['patients']
        for old,current in zip(template,documents):
            assert {k:v for k,v in old.items() if k!='association'}=={k:v for k,v in current.items() if k!='association'}
            assert [str(a['patient_id']) for a in current['association']]==patient_ids[str(current['pmc_id'])]
            assert [{k:v for k,v in a.items() if k!='phenotype'} for a in current['association']]==[{k:v for k,v in a.items() if k!='phenotype'} for a in old['association']]
        assert pairs(documents)<=pool, name+' outside frozen candidate pool'

def compose(components,prompts,template_lines,helper):
    old_reverse,_=helper.build_union(components['old'],components['reverse'],prompts)
    rebuilt,_=helper.build_union(old_reverse,components['patient'],prompts)
    # Compare exact concept strings, not normalized/split HPO strings or array order.
    assert literal_map(rebuilt)==literal_map(components['triple']), 'Three-source baseline reconstruction differs'
    primary,primary_counts=helper.build_union(rebuilt,components['fulltext'],prompts)
    reverse_patient,_=helper.build_union(components['reverse'],components['patient'],prompts)
    diagnostic,diagnostic_counts=helper.build_union(reverse_patient,components['fulltext'],prompts)
    documents={ARMS[0]:primary,ARMS[1]:diagnostic};outputs={}
    assert pairs(primary)==pairs(components['triple'])|pairs(components['fulltext'])
    assert pairs(diagnostic)==pairs(components['reverse'])|pairs(components['patient'])|pairs(components['fulltext'])
    assert pairs(diagnostic)<=pairs(primary) and pairs(primary)-pairs(diagnostic)<=pairs(components['old'])
    original_bytes=helper.nonrelation_bytes(template_lines)
    for arm,data in documents.items():
        lines=[]
        for raw,document in zip(template_lines,data):
            start,end=helper.association_span(raw)
            lines.append(raw[:start]+json.dumps(document['association'],ensure_ascii=False)+raw[end:])
        assert helper.nonrelation_bytes(lines)==original_bytes
        assert literal_map([json.loads(line) for line in lines])==literal_map(data)
        outputs[arm]=''.join(lines).encode('utf-8')
    return outputs,dict(rebuilt_triple_literal_relation_sets_exact=True,template_nonrelation_bytes_exact=True,
        template_masked_nonrelation_sha256=hashlib.sha256(original_bytes).hexdigest(),
        primary_counts=primary_counts,diagnostic_counts=diagnostic_counts,
        literal_relations={name:len(pairs(value)) for name,value in components.items()},
        primary_added_over_triple=len(pairs(primary)-pairs(components['triple'])),
        diagnostic_added_over_triple=len(pairs(diagnostic)-pairs(components['triple'])),
        diagnostic_removed_from_triple=len(pairs(components['triple'])-pairs(diagnostic)))

def check_joint_freeze(freeze,plan,plan_ref,source_ref):
    assert freeze['status']=='ALL_FOUR_COMPLETE_PREDICTIONS_FROZEN_BEFORE_GOLD'
    assert freeze['plan']==plan_ref and freeze['source']==source_ref
    assert freeze['gold_read'] is False and freeze['new_quality_calls']==0
    assert set(freeze['predictions'])==set(FOLDS)
    for fold in FOLDS:
        assert set(freeze['predictions'][fold])==set(ARMS)
        for arm in ARMS:
            item=freeze['predictions'][fold][arm]
            assert item['path']==plan['outputs'][fold][arm]
            verify(item)
    return True
def guarded_gold_load(spec,freeze,plan,plan_ref,source_ref,reader):
    # No gold file stat/hash/open is reachable without all four frozen SHA checks.
    check_joint_freeze(freeze,plan,plan_ref,source_ref)
    verify(spec['gold_metadata_only'])
    return reader(spec['gold_metadata_only']['path'])
def fold_gate(candidate,baseline):
    assert all(math.isfinite(values[key]) for values in [candidate,baseline] for key in ['Score','macF1'])
    return candidate['Score']>baseline['Score'] and candidate['macF1']>=baseline['macF1']
def select(results):
    passes={arm:all(results[fold][arm]['fold_pass'] for fold in FOLDS) for arm in ARMS}
    totals={arm:sum(results[fold][arm]['delta_vs_triple']['Score'] for fold in FOLDS) for arm in ARMS}
    passing=[arm for arm in ARMS if passes[arm]]
    chosen=None
    if passing:
        chosen=ARMS[0] if ARMS[0] in passing else passing[0]
        for arm in passing:
            if totals[arm]>totals[chosen]:chosen=arm
    return dict(pass_both_folds=passes,summed_twofold_Score_delta=totals,selected_candidate=chosen,
                exact_tie_policy='primary-four-source',automatic_migration=False)

def main():
    assert __debug__, 'Use ordinary Python; assertions are required'
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    root=args.directory.resolve();began=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60 second CPU cap')));signal.alarm(60)
    plan=read(root/'plan.json');entry=read(root/'entry.json');plan_ref=ref(root/'plan.json');source_ref=ref(__file__)
    assert entry['plan']==plan_ref and entry['source']==source_ref and plan['source']==source_ref
    assert plan['schema']=='fixed-four-source-fulltext-twofold-v1' and plan['fold_order']==FOLDS and plan['arms']==ARMS
    assert plan['keep_rule']==KEEP_RULE and plan['cpu_cap_seconds']==60 and plan['new_evaluator_calls']==4 and plan['old_evaluator_calls']==0
    assert plan['posthoc_development_control'] and not plan['original_prequality_registration']
    assert plan['future_migration_must_preserve_35_public_new'] and not plan['automatic_dev_migration']
    assert not (root/'assembled').exists() and not (root/'evaluation').exists()
    # A global single-use marker precedes every actual prediction/input read.
    save(root/'started.json',dict(at_utc=now(),source=source_ref,plan=plan_ref,entry=ref(root/'entry.json'),gold_read=False,new_quality_calls=0))
    for record in plan['common_references'].values():verify(record)
    for fold in FOLDS:
        for record in plan['folds'][fold]['references'].values():verify(record)
    helper=load_module('literal_union_helpers',plan['common_references']['helpers']['path'])
    handoff=read(plan['common_references']['handoff']['path'])
    assert handoff['source']==plan['common_references']['triple_source']
    predictions={};contracts={};inputs_by_fold={};prior_metrics={}
    (root/'assembled').mkdir()
    for fold in FOLDS:
        spec=plan['folds'][fold];references=spec['references'];expected=spec['expected']
        endpoint=read(references['patient_endpoint_plan']['path'])
        assert endpoint['gold']==spec['gold_metadata_only']
        for name,original in [('old','baseline_pred'),('input','input'),('patient_prompts','prompts')]:assert references[name]==endpoint['references'][original]
        frozen=read(references['fulltext_freeze']['path'])
        assert frozen['old_baseline']==references['old'] and frozen['endpoints']['diagnostic-fulltext-alone']['prediction']==references['fulltext']
        assert frozen['documents']==16 and frozen['entities']==expected['entities'] and frozen['nonrelation_bytes_exact']
        assert read(references['fulltext_review']['path'])['status'].startswith('PASS_')
        assert references['triple']==handoff['folds'][fold]['references']['triple_prediction']
        assert references['triple_metrics']==handoff['folds'][fold]['references']['triple_metrics']
        oldmetrics=read(references['triple_metrics']['path'])
        assert oldmetrics['prediction_sha256']==references['triple']['sha256'] and oldmetrics['gold_sha256']==spec['gold_metadata_only']['sha256']
        assert oldmetrics['evaluator_version']=='patientphex-public-approx-v2' and oldmetrics['configuration']['body_scope']=='all' and oldmetrics['configuration']['unlinked_document_mode']=='text'
        assert oldmetrics['metrics']==spec['baseline_metrics']==handoff['folds'][fold]['triple_metrics_values']
        prior_metrics[fold]=oldmetrics['metrics']
        isolation=read(references['patient_data_summary']['path']);train=set(isolation['training']['document_ids']);held=set(isolation[fold+'_input_only']['document_ids'])
        assert len(train)==expected['train_documents'] and len(held)==16 and not train&held
        assert isolation[fold+'_input_only']['targets_read'] is False and isolation[fold+'_input_only']['answer_fields']==0
        prompts=rows(references['original_prompts']['path']);patient_prompts=rows(references['patient_prompts']['path']);inputs=rows(references['input']['path'])
        projection=lambda values:[{k:p[k] for k in ['row_id','pmc_id','code_map','patient_ids']} for p in values]
        assert projection(prompts)==projection(patient_prompts), 'Candidate/PID pool drift across source schemas'
        assert {str(d['pmc_id']) for d in inputs}==held
        pool,patient_ids=pool_from_prompts(prompts,inputs,expected)
        components={name:rows(references[name]['path']) for name in ['old','reverse','patient','triple','fulltext']}
        validate_components(components,inputs,pool,patient_ids,expected)
        raw=Path(references['triple']['path']).read_bytes();template_lines=raw.decode('utf-8').splitlines(keepends=True)
        assert ''.join(template_lines).encode('utf-8')==raw and len(template_lines)==expected['documents']
        output,contract=compose(components,prompts,template_lines,helper);predictions[fold]={}
        for arm in ARMS:
            path=Path(plan['outputs'][fold][arm]);assert path.parent==root/'assembled'
            publish(path,output[arm]);predictions[fold][arm]=ref(path)
        contracts[fold]=dict(expected=expected,all_source_PMCPID_complete=True,all_source_nonrelation_semantics_equal=True,
            all_positive_literal_relations_in_same_candidate_pool=True,training_validation_document_overlap=0,**contract)
        inputs_by_fold[fold]=inputs
    save(root/'assembled/contract.json',contracts)
    freeze=dict(status='ALL_FOUR_COMPLETE_PREDICTIONS_FROZEN_BEFORE_GOLD',at_utc=now(),source=source_ref,plan=plan_ref,
                entry=ref(root/'entry.json'),predictions=predictions,contract=ref(root/'assembled/contract.json'),gold_read=False,new_quality_calls=0)
    save(root/'assembled/common-freeze.json',freeze);check_joint_freeze(freeze,plan,plan_ref,source_ref)
    (root/'evaluation').mkdir()
    # H50: register the exact canonical sibling first; import no evaluator before freeze.
    common=plan['common_references'];validator=load_module('validate_predictions',common['validator']['path'])
    evaluator=load_module('evaluate',common['evaluator']['path'])
    assert ref(validator.__file__)==common['validator'] and ref(evaluator.__file__)==common['evaluator']
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    hpo=validator.read_hpo(common['hpo']['path']);assert hpo['version']=='hp/releases/2026-06-23'
    loaded_predictions={}
    for fold in FOLDS:
        loaded_predictions[fold]={}
        for arm in ARMS:
            prediction=validator.load_jsonl(predictions[fold][arm]['path']);loaded_predictions[fold][arm]=prediction
            report=validator.validate_predictions(inputs_by_fold[fold],prediction,hpo)
            save(root/'evaluation'/(fold+'.'+arm+'.validation.json'),report);assert report['valid']
    save(root/'evaluation/before-gold.json',dict(at_utc=now(),common_freeze=ref(root/'assembled/common-freeze.json'),predictions=predictions,
         all_four_format_valid=True,gold_not_read=True,new_quality_calls=0,old_quality_calls=0,
         exact_order=[[fold,arm] for fold in FOLDS for arm in ARMS]))
    results={};calls=0
    for fold in FOLDS:
        gold=guarded_gold_load(plan['folds'][fold],freeze,plan,plan_ref,source_ref,validator.load_jsonl);results[fold]={}
        for arm in ARMS:
            check_joint_freeze(freeze,plan,plan_ref,source_ref)
            # Exclusive call-intent precedes each of the four unique scorer calls.
            save(root/'evaluation'/(fold+'.'+arm+'.call-intent.json'),dict(at_utc=now(),ordinal=calls+1,prediction=predictions[fold][arm],common_freeze=ref(root/'assembled/common-freeze.json')))
            report=evaluator.evaluate_report(gold,loaded_predictions[fold][arm],'all','text');calls+=1
            report['gold_sha256']=plan['folds'][fold]['gold_metadata_only']['sha256'];report['prediction_sha256']=predictions[fold][arm]['sha256']
            save(root/'evaluation'/(fold+'.'+arm+'.metrics.json'),report)
            values=report['metrics'];old=prior_metrics[fold]
            assert set(values)==set(old) and all(math.isfinite(v) for v in values.values())
            delta={key:values[key]-old[key] for key in old};assert delta['menF1']==delta['docF1']==0
            results[fold][arm]=dict(metrics=values,baseline_triple=old,delta_vs_triple=delta,
                fold_pass=fold_gate(values,old),metrics_file=ref(root/'evaluation'/(fold+'.'+arm+'.metrics.json')))
    assert calls==4
    selection=select(results)
    summary=dict(status='COMPLETE_FIXED_FOUR_SOURCE_POSTHOC_TWO_FOLD_CONTROL',at_utc=now(),folds=results,selection=selection,
        keep_rule=KEEP_RULE,common_freeze=ref(root/'assembled/common-freeze.json'),source=source_ref,plan=plan_ref,
        new_evaluator_calls=calls,old_evaluator_calls=0,posthoc_development_control=True,original_prequality_registration=False,
        automatic_dev_migration=False,future_migration_must_preserve_35_public_new=True,new_training_GPU_or_official=False,
        patient_F0_disclosure=plan['patient_F0_disclosure'],elapsed_seconds=time.monotonic()-began)
    save(root/'evaluation/summary.json',summary);signal.alarm(0);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
