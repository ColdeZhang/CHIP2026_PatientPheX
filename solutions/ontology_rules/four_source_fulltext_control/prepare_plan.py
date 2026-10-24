"""Metadata-only registration; never open actual prediction/input/gold bodies."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
from datetime import datetime,timezone

def ref(path):
    p=Path(path);raw=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs-root',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    r=args.runs_root.resolve();b=args.directory.resolve();hpath=r/'research/triple-union-fulltext-handoff-0600/refs.json'
    assert ref(hpath)['sha256']=='c1ecab45445f3a71df62598e423b0ec07246bc9f2b777b5c2cd7fd73ddc4abb3'
    handoff=read(hpath);historical=read(handoff['plan']['path'])
    assert ref(handoff['plan']['path'])==handoff['plan'] and ref(handoff['source']['path'])==handoff['source']
    common={k:historical['common_references'][k] for k in ['evaluator','validator','hpo']}
    common.update(handoff=ref(hpath),triple_source=handoff['source'],triple_review=handoff['independent_review'],
                  triple_freeze=handoff['common_freeze'],historical_plan_metadata_only=handoff['plan'],
                  helpers=ref(b/'literal_union_helpers.py'),helper_provenance=ref(b/'helper-provenance.json'),synthetic_contract=ref(b/'synthetic-result.json'))
    source=ref(b/'control.py');fixture=read(b/'synthetic-result.json')
    assert fixture['source']==source and fixture['helpers']==common['helpers'] and all(fixture['tests'].values())
    assert not fixture['actual_predictions_or_inputs_opened'] and not fixture['actual_gold_opened'] and fixture['evaluator_calls']==0
    scopes={};outputs={};arms=['primary-four-source','diagnostic-without-old']
    for fold in ['F1','F0']:
        old=handoff['folds'][fold];refs=old['references'];expected=old['expected']
        full=r/'research/qwen35-sft32-full-report-feasibility'/('full80-v1' if fold=='F1' else 'f0-preparation/full58-v1')
        template_path=full/'evaluation-source/evaluation-template.json';template=read(template_path)
        freeze_path=full/'quality/paired-predictions-freeze.json';freeze=read(freeze_path)
        assert freeze['old_baseline']==refs['old'] and template['references']['old_baseline_pred']==refs['old']
        gold=historical['folds'][fold]['gold']
        assert template['gold_path']==gold['path'] and template['historical_'+fold+'_gold_sha256']==gold['sha256']
        fulltext=freeze['endpoints']['diagnostic-fulltext-alone']['prediction']
        review=r/'validation'/('qwen35-sft32-full-report-probe-review/review-metadata-scope-revision1.json' if fold=='F1' else 'qwen35-sft48-full-report-f0-review/review.json')
        assert read(review)['status'].startswith('PASS_')
        references={name:refs[name] for name in ['old','reverse','patient','patient_endpoint_plan','patient_data_summary','reverse_plan']}
        references.update(triple=refs['triple_prediction'],triple_metrics=refs['triple_metrics'],
            input=template['references']['input'],original_prompts=template['references']['prompts'],
            patient_prompts=historical['folds'][fold]['references']['prompts'],
            fulltext=fulltext,fulltext_freeze=ref(freeze_path),fulltext_review=ref(review),
            fulltext_template_metadata=ref(template_path))
        assert old['triple_metrics_values']['Score']==({'F1':.6638418206599839,'F0':.7095567430848545}[fold])
        assert expected['entities']==({'F1':1189,'F0':947}[fold])
        scopes[fold]=dict(expected=expected,references=references,baseline_metrics=old['triple_metrics_values'],
                         gold_metadata_only=gold,gold_metadata_source=handoff['plan'])
        outputs[fold]={arm:str(b/'assembled'/(fold+'.'+arm+'.pred.jsonl')) for arm in arms}
    tree=ast.parse((b/'control.py').read_text());keep=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KEEP_RULE' for t in n.targets))
    plan=dict(schema='fixed-four-source-fulltext-twofold-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Root via ontology owner: exactly two fixed literal arms on F1/F0; owner alone executes after source checkpoint. Preparation agent no actual predictions, gold, quality, remote or Git.',
        source=source,fold_order=['F1','F0'],arms=arms,
        arm_definitions={'primary-four-source':['old','reverse','patient','fulltext'],'diagnostic-without-old':['reverse','patient','fulltext']},
        comparison='Existing strong old+reverse+patient triple only; historical old_union_metrics is not the comparison baseline.',
        keep_rule=keep,cpu_cap_seconds=60,new_evaluator_calls=4,old_evaluator_calls=0,
        fold_order_and_call_order=[['F1',arms[0]],['F1',arms[1]],['F0',arms[0]],['F0',arms[1]]],
        folds=scopes,common_references=common,outputs=outputs,
        output_sha_state='Not yet generated; all four original bytes/SHA will be frozen jointly before any gold hash/open or scorer call.',
        posthoc_development_control=True,original_prequality_registration=False,
        development_disclosure='F1/F0 component results and previous tuning are already known. This is a newly registered fixed posthoc development control, not an untouched holdout or original prequality hypothesis.',
        patient_F0_disclosure=historical['patient_F0_disclosure'],
        no_model_raw_response_parser=True,no_threshold_or_subset_scan=True,no_model_or_training_or_remote=True,
        all_four_full_predictions_before_gold=True,all_four_format_before_gold=True,
        complete_nonassociation_UTF8_bytes_from_triple=True,baseline_entities_by_fold={'F1':1189,'F0':947},
        automatic_dev_migration=False,future_migration_must_preserve_35_public_new=True,
        source_preparation_did_not_read_actual_predictions_inputs_or_gold=True,
        execution_state='REGISTERED_NOT_EXECUTED',global_exclusive_started='started.json',
        preparation_source=ref(__file__))
    save(b/'plan.json',plan)
    entry=dict(status='FROZEN_SOURCE_AND_PLAN_OWNER_ONLY_SINGLE_CPU_EXECUTION',source=source,plan=ref(b/'plan.json'),
               helpers=common['helpers'],synthetic_contract=common['synthetic_contract'],
               command=['python',str(b/'control.py'),'--directory',str(b)],
               cpu_cap_seconds=60,execution_started=False,new_predictions_exist=False,new_quality_calls=0)
    save(b/'entry.json',entry)
    print(json.dumps(dict(plan=ref(b/'plan.json'),entry=ref(b/'entry.json'),source=source)))
if __name__=='__main__':main()
