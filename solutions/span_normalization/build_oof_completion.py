"""Freeze two missing outer-fold controls from checked existing nested origins."""
import argparse, copy, gzip, json
from pathlib import Path
from common import read_jsonl, write_json, write_jsonl, ref, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    r=Path(a.runs);d=Path(a.output);l=Path(__file__).resolve().parent
    template=json.loads((l/'oof-f0-plan.json').read_text());prepared=json.loads(Path(template['prepared_manifest']['path']).read_text())
    d.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha(p) for p in l.glob('*.py')}
    reuse={(2,0):('oof-feature-control-f0','inner2'),(2,1):('oof-feature-control','inner2'),
           (3,0):('oof-feature-control-f0','inner3'),(3,1):('oof-feature-control','inner3')}
    alljobs=[];plans=[];reuse_checks=[]
    for fold in (2,3):
        plan=copy.deepcopy(template);plan.update(experiment_id='E-20261024-SPAN-004',status='FROZEN_FOURFOLD_COMPLETION_KNOWN_FIRST_TWO_RESULTS',outer_fold=fold,expected_gpu_steps=11,sources=sources)
        oldroot=r/f'research/lightgbm-selector-readiness/formal-control/fold{fold}';m=json.loads((oldroot/'fitted/manifest.json').read_text())
        reference=json.loads(Path(m['reference']['path']).read_text());assert sha(m['reference']['path'])==m['reference']['sha256']
        prep=prepared['folds'][fold];outerids=sorted(prep['input_ids']);outertrain=sorted(prep['train_ids']);assert len(outerids)==16 and len(outertrain)==48 and not set(outerids)&set(outertrain)
        plan.update(outer_held_ids=outerids,same_old_features=m['actual_feature_count'],same_training_rows=m['training_examples'],same_input_candidates=m['input_candidates'],training_positives=m['training_positive'],old_fit=ref(m['reference']['path']),strong_fit_manifest=ref(oldroot/'fitted/manifest.json'),strong_baseline=ref(oldroot/'lightgbm.nested.jsonl'),strong_metrics=ref(oldroot/'lightgbm.metrics.json'),existing_baseline_model=m['outputs']['model-vectorizer.pkl'],existing_baseline_scores=m['outputs']['selector-scores.jsonl'])
        plan.pop('existing_weights');plan['jobs']=[]
        origins=json.loads(Path(reference['config']['training_manifest']).read_text())
        assert len(origins)==len(m['training_origins'])==3
        for o,recorded in zip(origins,m['training_origins']):
            for k,h in recorded.items():assert sha(o[k])==h,(fold,k)
        with gzip.open(prep['train']['path'],'rt') as f:trainpool=[json.loads(s) for s in f]
        with gzip.open(prep['input']['path'],'rt') as f:heldpool=[json.loads(s) for s in f]
        for role in ('train','input'):assert sha(prep[role]['path'])==prep[role]['sha256']
        specs=[(int(Path(o['input']).parent.name.split('-')[-1]),o['generator_train'],o['input'],trainpool) for o in origins]
        specs.append((fold,reference['config']['train'],reference['config']['input'],heldpool))
        for inner,trainpath,inputpath,pool in specs:
            final=inner==fold;name=f'f{fold}_'+('final48' if final else f'inner{inner}')
            ti=sorted(str(x['pmc_id']) for x in read_jsonl(trainpath));hi=sorted(str(x['pmc_id']) for x in read_jsonl(inputpath))
            assert len(ti)==(48 if final else 32) and len(hi)==16 and not set(ti)&set(hi) and not set(ti)&set(outerids)
            assert ti==sorted(set(outertrain)-(set() if final else set(hi)))
            targets={}
            for row in pool:
                if str(row['pmc_id']) not in hi:continue
                for c in row['candidates']:
                    k=(str(row['pmc_id']),int(c['offset']),int(c['length']));v={'pmc_id':k[0],'offset':k[1],'length':k[2],'text':c['text']}
                    if k in targets:assert targets[k]==v
                    targets[k]=v
            vals=[targets[k] for k in sorted(targets)];assert {v['pmc_id'] for v in vals}==set(hi)
            targetpath=d/'data'/name/'targets.jsonl';write_jsonl(targetpath,vals)
            job={'name':name,'train_new':final or (fold,inner)==(2,3),'train':ref(trainpath),'input':ref(inputpath),'training_documents':len(ti),'train_ids':ti,'input_ids':hi,'outer_held_ids':outerids,'targets':ref(targetpath),'target_spans':len(vals)}
            if (fold,inner) in reuse:
                route,group=reuse[(fold,inner)];old=r/'stage8-span-normalization'/route/'actual';cp=old/'run'/group/'model/complete.json';pr=old/'prepared'/group/'preparation.json'
                completion=json.loads(cp.read_text());preparation=json.loads(pr.read_text())
                assert completion['status']=='COMPLETE_TRAINING_AND_FULL_HELD_INFERENCE' and preparation['train_pmcs']==ti and preparation['sources']['train']['sha256']==sha(trainpath)
                assert preparation['configuration']==template['training_configuration']
                job['existing_weights']=completion['model'];job['reuse_evidence']={'completion':ref(cp),'preparation':ref(pr),'exact_training_bytes':True,'exact_training_ids':True}
                reuse_checks.append({'job':name,**job['reuse_evidence'],'weights':completion['model']})
            elif (fold,inner)==(3,2):job['reuse_new_job']='f2_inner3'
            plan['jobs'].append(job);alljobs.append(job)
        plan.update(evaluation='Both missing full-fold predictions jointly freeze before either new v2; exactly F2,F3,pooled v2; reuse F0/F1 metrics.',success='Fourfold pooled Score strictly improves and at least three folds do not decline; else close without sweep.',estimate_basis='Observed F1 277.219s / F0 272.023s; three new models plus eight passes estimated338s, maximum900+30.',estimated_gpu_seconds=338)
        path=l/f'oof-f{fold}-completion-plan.json';write_json(path,plan);plans.append(ref(path))
    pooled=r/'research/lightgbm-selector-readiness/formal-control/result.json'
    plan={'experiment_id':'E-20261024-SPAN-004','status':'FROZEN_KNOWN_F0_NEGATIVE_F1_POSITIVE','jobs':alljobs,'fold_plans':plans,'training_configuration':template['training_configuration'],'gpu_budget_seconds':900,'term_grace_seconds':30,'expected_gpu_steps':11,'estimated_gpu_seconds':338,'sources':sources,'old_pooled_result':ref(pooled),'gate':'pooled_strictly_up_and_at_least_three_folds_non_decreasing','known_results':{},'new_quality_calls':[2,3,'pooled64'],'old_quality_calls':0}
    for fold,route in [(0,'oof-feature-control-f0'),(1,'oof-feature-control')]:
        q=r/'stage8-span-normalization'/route/'quality';plan['known_results'][str(fold)]={n:ref(q/n) for n in ('prediction.jsonl','metrics.json','result.json','complete-freeze.json')}
    plan['evaluation_sources']={str(f):ref(Path('/workspace/patientphex-data/round2-inner4')/f'fold-{f}'/'validation16.gold.jsonl') for f in (2,3)}
    plan['evaluation_sources'].update(pooled64=ref('/workspace/patientphex-data/day1/train64.jsonl'),evaluator=ref('/workspace/CHIP2026_PatientPheX/scripts/evaluate.py'))
    write_json(d/'reuse-inventory.json',{'status':'EXACT_TRAINING_BYTES_AND_IDS_PASS','reused_models':reuse_checks,'new_trainings':3,'target_passes':8})
    write_json(l/'oof-completion-plan.json',plan)
    print(json.dumps({'jobs':len(alljobs),'new_trainings':sum(x['train_new'] for x in alljobs),'reuse':len(reuse_checks),'fold_features':[json.loads(Path(x['path']).read_text())['same_old_features'] for x in plans]}))
if __name__=='__main__':main()
