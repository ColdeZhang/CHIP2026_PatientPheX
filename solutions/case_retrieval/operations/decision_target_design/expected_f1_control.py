#!/usr/bin/env python3
"""One <=30s patient-set expected-F1 arm, only after the fixed .5 main endpoint."""
from pathlib import Path
import argparse,json,hashlib,datetime,time,subprocess,collections,re,ast
import numpy as np
from expected_f1_math import expected_f1_by_size

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def checked(r):assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes']
def save(p,x):
    with p.open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def command(argv,p):
    x=subprocess.run(argv,capture_output=True,text=True)
    save(p,dict(argv=argv,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr));assert x.returncode==0,x.stderr

def main():
    p=argparse.ArgumentParser();p.add_argument('--main-directory',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    a=p.parse_args();start=time.monotonic();m=a.main_directory.resolve();b=a.directory.resolve()
    policy=read(a.policy);assert sha(__file__)==policy['control_sha256']
    assert sha(Path(__file__).parent/'expected_f1_math.py')==policy['math_sha256']
    # Main control must really have finished its immutable fixed-.5 quality endpoint first.
    assert read(m/'summary.json')['status']=='complete_same_template_base_vs_fresh_slot_adapter_F1_control'
    mp=read(m/'run-plan.json');r=mp['evaluation_references']
    assert not b.exists();b.mkdir()
    for key in ['input','prompts','entities','parser','validator','evaluator','hpo']:checked(r[key])
    mainfreeze=read(m/'paired-full-predictions-freeze.json');checked(mainfreeze['arms']['adapter'])
    complete=read(m/'complete.json');probpath=m/'adapter.probabilities.jsonl'
    assert sha(probpath)==complete['arms']['adapter']['probabilities_sha256']
    plan=dict(status='one_predefined_expected_F1_control_after_main',policy=ref(a.policy),source=ref(Path(__file__)),
        probability=ref(probpath),main_summary=ref(m/'summary.json'),main_freeze=ref(m/'paired-full-predictions-freeze.json'),
        fixed_point5_prediction=mainfreeze['arms']['adapter'],fixed_point5_metrics=ref(m/'adapter.metrics.json'),
        unit_audit=policy['unit_audit'],no_search=True,threshold=None,cap_seconds=30,
        tie='exact score ties choose smaller k; probability ties retain original prompt/CID order',
        retention='delta(micF1+macF1)>0 AND delta(macF1)>=0; otherwise close',
        assumptions='Independent Bernoulli candidate units, probabilities need calibration. Missing candidates excluded; not guaranteed full-public micro+macro optimum.')
    save(b/'run-plan.json',plan);checked(policy['unit_audit']);assert read(policy['unit_audit']['path'])['one_to_one']
    prompts=rows(r['prompts']['path']);prob={v['row_id']:v for v in rows(probpath)}
    assert len(prompts)==len(prob)==80 and {v['row_id'] for v in prompts}==set(prob)
    supplied={(str(d['pmc_id']),str(p['patient_id'])) for d in rows(r['input']['path']) for p in d['patient']}
    groups=collections.defaultdict(list);maps={p['row_id']:{cid:[] for cid in p['code_map']} for p in prompts}
    units=set();compound=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
    for prompt in prompts:
        values=prob[prompt['row_id']];keys=[[c,p] for c in prompt['code_map'] for p in prompt['patient_ids']]
        assert values['keys']==keys and len(values['probabilities'])==len(keys)
        for (cid,pid),pr in zip(keys,values['probabilities']):
            concept=prompt['code_map'][cid];assert isinstance(concept,str) and not compound.fullmatch(concept)
            key=(str(prompt['pmc_id']),pid,concept);assert key not in units;units.add(key)
            groups[key[:2]].append(dict(row_id=prompt['row_id'],cid=cid,pid=pid,concept=concept,probability=pr))
    assert len(units)==1557 and set(groups)==supplied and len(groups)==39
    diagnostics=[]
    for key,items in groups.items():
        order,scores=expected_f1_by_size([v['probability'] for v in items]);k=int(np.argmax(scores))
        for i in order[:k]:
            v=items[int(i)];maps[v['row_id']][v['cid']].append(v['pid'])
        diagnostics.append(dict(pmc_id=key[0],patient_id=key[1],candidates=len(items),chosen_k=k,
            expected_F1_by_size=scores.tolist(),selected_indices=[int(i) for i in order[:k]],
            expected_F1=float(scores[k]),fixed_point5_count=sum(v['probability']>=.5 for v in items)))
    # Restore each prompt's supplied PID order; no HPO/None/text normalization.
    with (b/'responses.jsonl').open('x') as f:
        for prompt in prompts:
            answer={cid:[pid for pid in prompt['patient_ids'] if pid in selected] for cid,selected in maps[prompt['row_id']].items()}
            f.write(json.dumps(dict(row_id=prompt['row_id'],response=json.dumps(answer),hit_token_limit=False,generated=False))+'\n')
    save(b/'input-only-decisions.json',dict(patients=diagnostics,units=1557,no_gold=True))
    pred=b/'expected-f1.pred.jsonl'
    command([mp['python'],r['parser']['path'],'--input',r['input']['path'],'--entities',r['entities']['path'],
        '--prompts',r['prompts']['path'],'--responses',str(b/'responses.jsonl'),'--output',str(pred),'--fallback','baseline'],b/'merge-command.json')
    assert read(pred.with_suffix('.manifest.json'))['stats']=={'valid_candidate_decisions':587}
    previous=rows(plan['fixed_point5_prediction']['path']);current=rows(pred)
    assert len(current)==len(previous)==16
    for old,new in zip(previous,current):
        assert {k:v for k,v in old.items() if k!='association'}=={k:v for k,v in new.items() if k!='association'}
        assert [v['patient_id'] for v in old['association']]==[v['patient_id'] for v in new['association']]
    save(b/'prediction-freeze.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prediction=ref(pred),source_probabilities=ref(probpath),plan=ref(b/'run-plan.json'),decisions=ref(b/'input-only-decisions.json'),
        baseline=plan['fixed_point5_prediction'],no_new_control_gold_read=True))
    # Only now open original literal gold, after the new full prediction is frozen.
    save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=ref(b/'prediction-freeze.json')))
    checked(mp['gold_after_freeze'])
    command([mp['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',str(pred),'--hpo',r['hpo']['path'],
        '--output',str(b/'validation.json')],b/'validation-command.json');assert read(b/'validation.json')['valid']
    command([mp['python'],r['evaluator']['path'],'--gold',mp['gold_after_freeze']['path'],'--pred',str(pred),
        '--body-scope','all','--no-id-doc','text','--output',str(b/'metrics.json')],b/'evaluation-command.json')
    # Exact existing evaluator association-unit function, no new alias or text policy.
    fn=next(x for x in ast.parse(Path(r['evaluator']['path']).read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='associations')
    ns=dict(COMPOUND_HPO=compound,Any=object);exec(compile(ast.Module(body=[fn],type_ignores=[]),r['evaluator']['path'],'exec'),ns)
    gold={str(d['pmc_id']):ns['associations'](d) for d in rows(mp['gold_after_freeze']['path'])};patients=[]
    for old,new in zip(previous,current):
        did=str(old['pmc_id']);aa=ns['associations'](old);bb=ns['associations'](new)
        for pid,truth in gold[did].items():
            def stats(v):
                tp=len(v&truth);fp=len(v-truth);fn=len(truth-v)
                return dict(tp=tp,fp=fp,fn=fn,F1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.)
            patients.append(dict(pmc_id=did,patient_id=pid,before=stats(aa[pid]),after=stats(bb[pid])))
    metrics=read(b/'metrics.json');checked(plan['fixed_point5_metrics']);old=read(plan['fixed_point5_metrics']['path'])
    delta={k:metrics['metrics'][k]-old['metrics'][k] for k in old['metrics']}
    assert delta['menF1']==delta['docF1']==0
    keep=delta['micF1']+delta['macF1']>0 and delta['macF1']>=0
    assert time.monotonic()-start<30
    save(b/'summary.json',dict(status='complete_one_expected_F1_CPU_control',metrics=metrics,baseline=old,delta=delta,
        retention_criterion_pass=keep,patients=patients,freeze=ref(b/'prediction-freeze.json'),
        elapsed_seconds=time.monotonic()-start,no_search=True,no_new_GPU=True,no_original_dev=True,
        boundary=plan['assumptions'],next_action='retain_for_root_review_only' if keep else 'close_no_parameter_sweep'))
    print(json.dumps(dict(retain=keep,delta=delta,seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
