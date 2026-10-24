"""Fixed alpha05 F1 endpoint support; preparation/assembly cannot open gold."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ARMS=['primary','newP-alone']
METRICS=('menF1','docF1','micF1','macF1','Score')
KEEP='primary Score strictly greater than original N/R/P triple; macro disclosed without veto'
BYTE_HELPER_SHA='c713d2f5d5b2768cfb80ce4c17f920c6228dc3abb833080c972255d6e34c6e06'


def now():return datetime.now(timezone.utc).isoformat()


def ref(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def checked(item):
    assert isinstance(item,dict) and ref(item['path'])==item,item
    return Path(item['path'])


def read(path):return json.loads(Path(path).read_text())


def publish(path,raw):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink()
    parent=path.resolve().parent
    while True:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if parent.parent==parent:break
        parent=parent.parent


def save(path,value):publish(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def rows(path):
    lines=Path(path).read_bytes().decode('utf-8').splitlines(keepends=True)
    docs=[json.loads(line) for line in lines];ids=[str(d['pmc_id']) for d in docs]
    assert len(ids)==len(set(ids))
    return lines,docs,ids


def literal_keys(docs):
    keys=set()
    for doc in docs:
        seen=set()
        for item in doc['association']:
            pid=item['patient_id'];assert pid not in seen;seen.add(pid)
            values=item['phenotype'];assert all(isinstance(v,str) for v in values)
            assert len(values)==len(set(values)),'Predictions must not duplicate literal relations'
            keys.update((str(doc['pmc_id']),pid,value) for value in values)
    return keys


def load_module(name,item):
    spec=importlib.util.spec_from_file_location(name,checked(item));module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def gold_guard(plan):
    target=os.path.realpath(plan['gold']['path']);state={'enabled':False}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0]))==target and not state['enabled']:
                raise PermissionError('Gold open/hash forbidden until both frozen endpoints and score worker gate')
    sys.addaudithook(audit);return state


def load_plan(path):
    assert __debug__
    plan=read(path);guard=gold_guard(plan)
    assert plan['schema']=='patient-balanced-F1-fixed-two-endpoint-v1'
    assert plan['arms']==ARMS and plan['keep_rule']==KEEP and plan['alpha']==.5
    assert plan['score_budget_seconds']==30 and plan['term_grace_seconds']==1
    assert plan['new_evaluator_calls']==2 and plan['old_evaluator_calls']==0
    assert plan['state']=='BOUND_ACTUAL_STRICT80' and isinstance(plan['actual'],dict),'Template blocks until actual final80 binds'
    for name,item in plan['sources'].items():assert Path(item['path']).name==name;checked(item)
    assert plan['sources']['support.py']==ref(__file__)
    refs=plan['references'];design=read(checked(refs['quality_design']))
    majority=read(checked(refs['majority_plan']))
    assert plan['gold']==majority['folds']['F1']['gold']  # metadata only; never checked here
    for key in ['normal','reverse','old_patient','method_baseline','method_baseline_metrics','F1_input',
                'stronger_F1_reference','stronger_reference_summary']:
        assert refs[key]==design[key];checked(refs[key])
    assert plan['expected']==majority['folds']['F1']['expected']
    assert plan['baseline_score']==design['baseline_score']==0.6638418206599839
    assert plan['stronger_reference_score']==design['stronger_F1_reference_score']==0.676546354662594
    for key in ['evaluator','validator','hpo']:assert refs[key]==majority['references'][key]
    assert refs['byte_helper']==majority['sources']['common.py']
    assert refs['byte_helper']['sha256']==BYTE_HELPER_SHA
    for key in ['byte_helper','prompts','input_token_audit','schema','parser','tokenizer','evaluator','validator']:
        checked(refs[key])
    baseline=read(refs['method_baseline_metrics']['path'])
    assert baseline['evaluator_version']=='patientphex-public-approx-v2'
    assert baseline['configuration']['body_scope']=='all' and baseline['configuration']['unlinked_document_mode']=='text'
    assert baseline['prediction_sha256']==refs['method_baseline']['sha256']
    assert baseline['gold_sha256']==plan['gold']['sha256'] and baseline['metrics']['Score']==plan['baseline_score']
    assert read(refs['stronger_F1_reference']['path'])['metrics']['Score']==plan['stronger_reference_score']
    return plan,guard


def byte_helpers(item):
    assert item['sha256']==BYTE_HELPER_SHA
    source=checked(item).read_text();tree=ast.parse(source)
    wanted=['field_span','array_spans','phenotype_spans','mask_values']
    nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in wanted]
    namespace={'json':json}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),item['path'],'exec'),namespace)
    return namespace['phenotype_spans'],namespace['mask_values']


def replace_phenotypes(line,values,spans,mask):
    doc=json.loads(line);positions=spans(line);assert len(positions)==len(doc['association'])
    result=line
    for item,(begin,end) in reversed(list(zip(doc['association'],positions))):
        selected=values[item['patient_id']]
        if selected!=item['phenotype']:
            result=result[:begin]+json.dumps(selected,ensure_ascii=False)+result[end:]
    assert mask(result)==mask(line),'All bytes outside phenotype values must remain unchanged'
    return result


def mapped_raw_ref(item,actual):
    remote=Path(actual['remote_prediction_root']);local=Path(actual['local_prediction_root']).resolve()
    relative=Path(item['path']).relative_to(remote)
    assert relative.parts and '..' not in relative.parts and not relative.is_absolute()
    path=local/relative;assert path.resolve().is_relative_to(local) and not path.is_symlink()
    found=ref(path);assert found['bytes']==item['bytes'] and found['sha256']==item['sha256']
    return path


def validate_freeze(plan_path,plan):
    path=Path(plan['output_directory'])/'assembled/common-freeze.json';frozen=read(path)
    assert frozen['plan']==ref(plan_path) and frozen['sources']==plan['sources']
    assert frozen['both_endpoints_complete'] and frozen['gold_read_or_hashed'] is False
    assert frozen['new_evaluator_calls_before_freeze']==0 and list(frozen['predictions'])==ARMS
    assert frozen['actual_inference']==plan['actual']
    for item in frozen['predictions'].values():checked(item)
    contract=read(checked(frozen['contract']));checked(frozen['parsed'])
    assert (contract['documents'],contract['patients'],contract['entities'],contract['strict_rows'])==(16,39,1189,80)
    assert contract['candidate_count_sum_code_map']==587 and contract['pair_count_sum_code_map_times_patients']==1557
    assert contract['all_phenotype_external_bytes_exact'] and contract['primary_literal_keys_exact_N_union_R_union_newP']
    assert contract['oldP_only_relations_not_carried'] and contract['gold_read_or_hashed'] is False
    assert contract['raw_IDs_decoded_replay_count']==80
    for name in ['responses','manifest','terminal','inference_source']:checked(plan['actual'][name])
    return path,frozen
