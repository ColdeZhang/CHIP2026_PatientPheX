"""Bind a stage only to complete actual artifacts; never run interpretation or v2."""
import argparse
import json
from pathlib import Path
from common import (BASE_SCORE, BASE_SHA, GOLD, HERE, OLD, POLICY_SHA, RUN, SEEDS,
                    checked, guard, read, ref, save, source_contract)


def template():
    old=read(OLD/'quality/metadata_recovery/plan.json')['references']
    bdir=Path('/workspace/patientphex-runs/round2-20261023/research/four-source-fusion-20261024/F0-v1')
    bf=read(bdir/'freeze.json')
    refs={name:old[name] for name in ['byte_helper','validator','evaluator','hpo']}
    refs['F0_input']=read(bf['plan']['path'])['refs']['input']
    refs.update(train32=ref('/workspace/patientphex-runs/round2-20261023/ontology_rules/nested-outer0-inner1/train32.jsonl'),
        policy=ref(OLD/'normalization/policy_v1.json'),
        prompts=ref('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/F0-preparation-v1/F0.input47.jsonl'),
        baseline=ref(bdir/'prediction.jsonl'),baseline_metrics=ref(bdir/'metrics.json'),
        baseline_freeze=ref(bdir/'freeze.json'),baseline_plan=bf['plan'])
    for item in refs.values():checked(item)
    assert refs['baseline']['sha256']==BASE_SHA and refs['policy']['sha256']==POLICY_SHA
    assert refs['train32']['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert read(refs['baseline_metrics']['path'])['metrics']['Score']==BASE_SCORE
    assert read(refs['baseline_plan']['path'])['refs']['gold']==GOLD
    return dict(schema='grounded-sampled-postprocess-template-v1',state='UNBOUND_NO_ACTUAL_RUN',
        seeds=list(SEEDS),presence_penalty=0,formula='B4 UNION (S1 INTERSECT S2)',
        refs=refs,sources=source_contract(),gold_metadata_only=GOLD,
        scope='Complete94 required; no single-seed quality; no cap-prefix salvage; no fit or bank encoding',
        outputs={name:str(RUN/path) for name,path in [('extract','extraction-v1'),('prepare','normalization-preparation-v1'),
            ('finalize','normalized-v1'),('quality','quality-F0-v1')]})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['template','extract','prepare','finalize','quality'])
    ap.add_argument('--collection');ap.add_argument('--runner');ap.add_argument('--inference-plan');ap.add_argument('--query-collection')
    ap.add_argument('--plan-output');a=ap.parse_args();state=guard();t=template()
    target=Path(a.plan_output) if a.plan_output else HERE/('plan-template.json' if a.stage=='template' else a.stage+'-plan.json')
    if a.stage=='template':value=t
    else:
        refs=t['refs'];inputs={}
        if a.stage=='extract':
            assert a.collection and a.runner and a.inference_plan
            c=read(a.collection);assert c['all_actual_SHA_verified'] and c['inference_reported_success']
            assert c['wrapper']['complete94_actual_members_verified']
            inputs={k:refs[k] for k in ['F0_input','prompts']}
            inputs.update(collection=ref(a.collection),runner=ref(a.runner),inference_plan=ref(a.inference_plan))
        elif a.stage=='prepare':
            p=Path(t['outputs']['extract'])/'manifest.json';v=read(p);assert v['complete_all94']
            inputs={k:refs[k] for k in ['F0_input','train32','hpo','policy']};inputs['extraction']=ref(p)
        elif a.stage=='finalize':
            p=Path(t['outputs']['prepare'])/'manifest.json';v=read(p);assert v['complete_all94']
            inputs={'preparation':ref(p)}
            if v['necessary_public_query_count']:
                assert a.query_collection;c=read(a.query_collection)
                assert c['all_actual_SHA_verified'] and c['query_reported_success']
                inputs.update(query_collection=ref(a.query_collection),query_scores=c['files']['output/query-scores.json'],
                              query_complete=c['files']['output/complete.json'])
            else:inputs.update(query_collection=None,query_scores=None,query_complete=None)
        else:
            p=Path(t['outputs']['finalize'])/'manifest.json';v=read(p);assert v['complete_all94']
            inputs={k:refs[k] for k in ['F0_input','train32','hpo','policy','byte_helper','validator','evaluator',
                'baseline','baseline_metrics','baseline_freeze','baseline_plan']};inputs['normalization']=ref(p)
        value=dict(schema='grounded-sampled-postprocess-v1',stage=a.stage,state='BOUND_COMPLETE_ACTUAL_ARTIFACTS',
            seeds=list(SEEDS),sources=t['sources'],gold_metadata_only=GOLD,inputs=inputs,
            output_directory=t['outputs'][a.stage])
        if a.stage=='quality':value.update(formula=t['formula'],baseline_score=BASE_SCORE,expected_baseline_relations=343,
            new_v2_max=1,old_v2=0,score_budget_seconds=30,term_grace_seconds=1,keep_rule='Score strictly above fixed B4; macro disclosed with no extra veto',
            no_added_means_baseline_byteexact_and_zero_v2=True)
    assert not state['F0_quality_enabled'];save(target,value);print(json.dumps(ref(target)))


if __name__=='__main__':main()
