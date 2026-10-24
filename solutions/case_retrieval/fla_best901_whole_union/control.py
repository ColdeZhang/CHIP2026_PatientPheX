"""One explicitly authorized post-result whole-pipeline union; no NN or subset search."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

SCHEMA='original901-or-complete-FLA-post-result-one-union-v1'

def ref(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(record):
    p=Path(record['path']);assert ref(p)==record;return p
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_bytes().splitlines() if s.strip()]
def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def publish(path,raw):
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_dir(Path(path).parent)
def save(path,value):publish(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def now():return datetime.now(timezone.utc).isoformat()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def pure_helpers(plan):
    scope=dict(json=json,re=re,copy=copy)
    for key,names,constants in [('union_source',{'values','relations','union','contracts'},{'COMPOUND'}),('literal_source',{'field_span'},set())]:
        source=checked(plan['references'][key]);nodes=[]
        for n in ast.parse(source.read_text()).body:
            if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
            elif isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in constants:nodes.append(n)
        assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    return scope

def check(plan):
    assert plan['schema']==SCHEMA and plan['execution_ready']
    assert plan['source']==ref(__file__)
    assert (plan['cpu_seconds'],plan['wall_seconds'],plan['grace_seconds'])==(60,65,5)
    assert plan['new_v2_calls']==1 and plan['old_v2_calls']==0
    assert plan['baseline_score']==.7543613981990362 and plan['known_replacement_negative'] is True
    for r in plan['references'].values():
        assert Path(r['path']).resolve()!=Path(plan['gold_path']).resolve();checked(r)
    assert plan['references']['baseline_prediction']['sha256']=='9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67'
    assert plan['references']['FLA_prediction']['sha256']=='fba773f750e851398e47bf4fdd27a1865b8c0e821ad19145079849506ccf95ef'

def worker(plan_path):
    start=time.monotonic();pr=ref(plan_path);p=read(plan_path);check(p);r=p['references'];h=pure_helpers(p);contracts=h['contracts']()
    oldlines=Path(r['baseline_prediction']['path']).read_text().splitlines(keepends=True)
    old=[json.loads(s) for s in oldlines];new=rows(r['FLA_prediction']['path']);inputs=rows(r['input']['path'])
    assert len(old)==len(new)==len(inputs)==16
    assert [str(d['pmc_id']) for d in old]==[str(d['pmc_id']) for d in new]==[str(d['pmc_id']) for d in inputs]
    assert sum(len(d['association']) for d in old)==sum(len(d['patient']) for d in inputs)==36
    assert sum(len(d['entities']) for d in old)==1061
    cached=read(r['baseline_metrics']['path']);fla=read(r['FLA_metrics']['path']);assert cached['metrics']['Score']==p['baseline_score']
    prior=read(r['FLA_summary']['path']);assert prior['keep'] is False and prior['metrics']==fla['metrics'] and prior['prediction']==r['FLA_prediction']
    review=read(r['FLA_review']['path']);assert review['status']=='PASS_COMPLETE_FIXED_ORIGINALDEV901_FLA_CONTROL_NEGATIVE_CLOSED'
    lines=[];out=[];added=[];old_count=new_count=union_count=0
    for line,x,y,inp in zip(oldlines,old,new,inputs):
        assert {str(a['patient_id']) for a in x['association']}=={str(a['patient_id']) for a in inp['patient']}
        z,log=h['union'](x,y);added.extend(log);a,b=h['field_span'](line,'association')
        rendered=line if not log else line[:a]+json.dumps(z['association'],ensure_ascii=False)+line[b:]
        c,d=h['field_span'](rendered,'association');assert (line[:a],line[b:])==(rendered[:c],rendered[d:])
        assert json.loads(rendered)==z
        ox,ny,uz=[h['relations'](q) for q in (x,y,z)]
        assert all(uz[pid]==ox[pid]|ny[pid] for pid in ox)
        old_count+=sum(map(len,ox.values()));new_count+=sum(map(len,ny.values()));union_count+=sum(map(len,uz.values()))
        out.append(z);lines.append(rendered)
    assert union_count==old_count+len(added)
    q=Path(p['quality_directory']);q.mkdir(exist_ok=False);fsync_dir(q.parent)
    publish(q/'prediction.jsonl',''.join(lines).encode());save(q/'added-relations.json',added)
    save(q/'assembly.json',dict(old_relation_count=old_count,FLA_relation_count=new_count,union_relation_count=union_count,
        added_relations=len(added),old_removed=0,documents=16,patients=36,entities=1061,
        all_nonassociation_literal_bytes_exact=True,old_patient_order_and_phenotype_prefix_preserved=True,
        generic_union_contracts=contracts,no_hardcoded_additions=True,no_response_parser=True,no_gold_read=True))
    predref=ref(q/'prediction.jsonl');freeze=dict(status='COMPLETE_WHOLE_UNION_FROZEN_BEFORE_FIRST_GOLD_BYTE',at_utc=now(),plan=pr,
        prediction=predref,assembly=ref(q/'assembly.json'),added=ref(q/'added-relations.json'),sources=p['references'],
        gold_read=False,new_v2_calls=0,old_v2_calls=0,known_replacement_result_development=True)
    save(q/'freeze.json',freeze);freeze_ref=ref(q/'freeze.json')
    save(q/'evaluation-started.json',dict(at_utc=now(),freeze=freeze_ref,new_v2_calls=1,old_v2_calls=0))
    va=module('validate_predictions',checked(r['validator']));ev=module('fixed_whole_pipeline_union_v2',checked(r['evaluator']))
    assert ev.VERSION=='patientphex-public-approx-v2' and ev.COMPOUND_HPO.pattern==h['COMPOUND'].pattern
    validation=va.validate_predictions(inputs,out,va.read_hpo(r['hpo']['path']));save(q/'validation.json',validation);assert validation['valid']
    assert ref(p['gold_path'])['sha256']==p['gold_sha256'];gold=rows(p['gold_path'])
    assert [str(d['pmc_id']) for d in gold]==[str(d['pmc_id']) for d in inputs]
    checked(predref);metric=ev.evaluate_report(gold,out,'all','text');save(q/'metrics.json',metric)
    assert metric['documents']==16 and metric['patients']==36
    assert all(metric['metrics'][k]==cached['metrics'][k] for k in ('menF1','docF1'))
    assert all(math.isfinite(v) for v in metric['metrics'].values())
    delta={k:v-cached['metrics'][k] for k,v in metric['metrics'].items()};keep=delta['Score']>0
    check(p);checked(pr);checked(predref);checked(freeze_ref)
    summary=dict(status='COMPLETE_ONE_POST_RESULT_WHOLE_PIPELINE_UNION',at_utc=now(),prediction=predref,metrics=metric['metrics'],
        baseline_metrics=cached['metrics'],complete_FLA_cached_metrics=fla['metrics'],delta=delta,keep=keep,
        criterion='Score strictly above original901; macro disclosed only',new_v2_calls=1,old_v2_calls=0,no_fit_or_NN=True,
        no_subset_threshold_or_weight_search=True,no_official_action=True,known_replacement_negative_before_plan=True,
        repeated_originaldev_development_not_blind=True,prior_replacement_negative_unchanged=True,
        freeze=freeze_ref,assembly=ref(q/'assembly.json'),metrics_file=ref(q/'metrics.json'),validation=ref(q/'validation.json'),
        elapsed_seconds=time.monotonic()-start)
    save(q/'summary.json',summary)
    print(json.dumps(dict(summary=ref(q/'summary.json'),Score=metric['metrics']['Score'],keep=keep)))

def limit():resource.setrlimit(resource.RLIMIT_CPU,(60,60))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.worker:worker(a.plan);return 0
    start=time.monotonic();deadline=start+65;p=read(a.plan);check(p);directory=Path(p['execution_directory']);directory.mkdir(parents=True,exist_ok=False);fsync_dir(directory.parent)
    assert not Path(p['quality_directory']).exists()
    state=dict(status='RUNNING_UNIQUE_WHOLE_PIPELINE_UNION',started_utc=now(),plan=ref(a.plan),source=ref(__file__),timeout_triggered=False,new_v2_calls_authorized=1,retries=0)
    save(directory/'intent.json',state);child=None
    try:
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(a.plan.resolve()),'--worker'],stdout=stdout,stderr=stderr,env=env,start_new_session=True,preexec_fn=limit)
            try:code=child.wait(timeout=max(0,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                state['timeout_triggered']=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=max(0,deadline+5-time.monotonic()))
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
        state['exit_code']=code;state['status']='COMPLETE_ONE_WHOLE_UNION_QUALITY' if code==0 and not state['timeout_triggered'] else 'FAILED_PRESERVED_NO_RETRY'
    except BaseException as e:
        state.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=max(0,min(5,deadline+5-time.monotonic())))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start)
        q=Path(p['quality_directory']);state['quality_files']={x.name:ref(x) for x in sorted(q.iterdir()) if x.is_file()} if q.exists() else {}
        state['execution_files']={x.name:ref(x) for x in sorted(directory.iterdir()) if x.is_file()}
        save(directory/'terminal.json',state);print(json.dumps({k:state.get(k) for k in ['status','exit_code','elapsed_seconds']}))
    return 0 if state['status']=='COMPLETE_ONE_WHOLE_UNION_QUALITY' else 1
if __name__=='__main__':raise SystemExit(main())
