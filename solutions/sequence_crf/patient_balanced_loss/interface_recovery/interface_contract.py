#!/usr/bin/env python3
"""Pure-Python real-forward interface contract; no Torch/model/numeric loss.

The supervisor is single-use and bounded by 10 seconds plus TERM grace 1.
Only after source/plan freeze may it execute against an empty output directory.
"""
from __future__ import annotations
import __future__
import argparse
import ast
import copy
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_parents(path):
    current=Path(path).resolve().parent
    while True:
        fd=os.open(current,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if current.parent==current: break
        current=current.parent


def write(path,obj):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    sync_parents(path)


def append(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    sync_parents(path)


def reference(path):
    return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}


def function_node(path,name,owner=None):
    tree=ast.parse(Path(path).read_text())
    body=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
    return copy.deepcopy(next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name))


def compile_function(node,namespace,path,decorators=None):
    original_body=ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)
    if decorators is not None: node.decorator_list=decorators
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    # Future annotations prevent any Tensor annotation from importing Torch.
    exec(compile(module,str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    assert ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)==original_body
    return namespace[node.name],{'line':node.lineno,'end_line':node.end_lineno,
        'body_ast_sha256':hashlib.sha256(original_body.encode()).hexdigest()}


def load_scope(path):
    tree=ast.parse(Path(path).read_text())
    imports=[name.name for n in tree.body if isinstance(n,ast.Import) for name in n.names]
    assert sorted(imports)==['copy','hashlib','json']
    assert not any(isinstance(n,ast.ImportFrom) for n in tree.body)
    namespace={}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['RowLossScope']


class TensorSentinel:
    """Shape and identity only: no storage, target values, tensor ops or devices."""
    def __init__(self,shape=(1,5,7),dtype='torch.bfloat16',device='cuda:fake'):
        self.shape,self.dtype,self.device=shape,dtype,device
        self.slices=[]
    def __getitem__(self,key):
        self.slices.append(key)
        return self


class OutputSentinel(SimpleNamespace):
    def to_tuple(self):
        return tuple(v for v in vars(self).values() if v is not None)


def extract_chain(plan):
    refs=plan['references']
    decorator,decorator_meta=compile_function(
        function_node(refs['return_decorator']['path'],'can_return_tuple'),{'wraps':wraps},refs['return_decorator']['path'])
    node=function_node(refs['qwen_modeling']['path'],'forward','Qwen3_5ForCausalLM')
    assert [n.id for n in node.decorator_list]==['can_return_tuple','auto_docstring']
    qwen,qwen_meta=compile_function(node,{'can_return_tuple':decorator,'CausalLMOutputWithPast':OutputSentinel},
        refs['qwen_modeling']['path'],[ast.Name(id='can_return_tuple',ctx=ast.Load())])
    # Tested path has ordinary LoRA, no prompt/POLY/MPT/aLoRA or special args.
    # Identity aLoRA helper and empty hook context are explicitly fake boundaries.
    alora_calls=[]
    def no_alora(self,input_ids,inputs_embeds,**kwargs):
        alora_calls.append({'keys':sorted(kwargs),'stub':'identity_no_aLoRA_only'})
        return kwargs
    peft,peft_meta=compile_function(function_node(refs['peft_forward']['path'],'forward'),
        {'get_alora_offsets_for_forward':no_alora,'PeftType':SimpleNamespace(POLY='POLY')},refs['peft_forward']['path'])
    return peft,qwen,alora_calls,{'peft_forward':peft_meta,'qwen_forward':qwen_meta,'return_decorator':decorator_meta,
        'removed_decorator':'auto_docstring only; can_return_tuple is original executed source',
        'unchanged_real_function_bodies':True,'no_aLoRA_helper_or_adapter_hook_implementation_claim':True}


def run_case(case,old_scope,new_scope,chain,out):
    peft_forward,qwen_forward,alora_calls,_=chain
    before_alora=len(alora_calls)
    events=[];core_calls=[];observer_calls=[];text_calls=[];head_calls=[]
    logits=TensorSentinel(dtype=case.get('dtype','torch.bfloat16'),device=case.get('device','cuda:fake'),
        shape=case.get('shape',(1,5,7)))
    labels=TensorSentinel(shape=(1,5),dtype='torch.int64')
    input_ids=object();attention_mask=object();hidden=TensorSentinel()
    loss_result=object()
    def native_fixed(*args,**kwargs):
        raise AssertionError('Native numerical CE must never execute')
    metadata={'row_index':3,'row_id':'synthetic:interface:3','sequence_length':5,
        'supervised_label_positions':[2,3,4],'supervised_logit_positions':[1,2,3],
        'patients':[{'opaque_patient_key':'synthetic-only','n_p':3,'value_label_positions':[3,4]}]}
    metadata_before=copy.deepcopy(metadata)
    g=case['group_size'];vocab=case.get('vocab',7)
    def original_loss(*args,**kwargs):
        raise AssertionError('Original native numerical loss must never execute')
    def core(*args,**kwargs):
        events.append('core')
        core_calls.append({'logits_same':args[0] is logits,'labels_same':args[1] is labels,
            'row_is_scope_copy':args[2] is scope.row,'row_content_unchanged':args[2]==metadata_before,
            'row_is_not_caller_mutable_object':args[2] is not metadata,
            'actual_group_size':args[3],'patients':args[4],'groups':args[5],
            'native_fixed_same':args[6] is native_fixed,'kwargs':kwargs,'positional_arguments':len(args)})
        return loss_result
    def observer(actual):
        events.append('observer');observer_calls.append(copy.deepcopy(actual))
        # Persist from inside observer, before the first compatibility guard.
        append(out/'observer-calls.jsonl',{'case':case['name'],'actual':actual,
            'events_so_far':list(events),'core_calls_so_far':len(core_calls)})
    def text_model(**kwargs):
        text_calls.append({'extra_keys':sorted(set(kwargs)-{'input_ids','attention_mask','position_ids','past_key_values','inputs_embeds','use_cache'}),
            'input_ids_same':kwargs['input_ids'] is input_ids,'attention_mask_same':kwargs['attention_mask'] is attention_mask,
            'use_cache_is_false':kwargs['use_cache'] is False})
        return OutputSentinel(last_hidden_state=hidden,past_key_values=None,hidden_states=None,attentions=None)
    def head(value):
        head_calls.append({'hidden_identity_same':value is hidden})
        return logits
    class FakeQwen:
        forward=qwen_forward
        def __call__(self,**kwargs): return self.forward(**kwargs)
    base=FakeQwen();base.config=SimpleNamespace(return_dict=True,model_type='qwen3_5_text',vocab_size=vocab)
    base.model=text_model;base.lm_head=head;base.loss_function=original_loss
    class FakePeft:
        forward=peft_forward
        def _enable_peft_forward_hooks(self,**kwargs): return nullcontext()
    wrapped=FakePeft();wrapped.base_model=base;wrapped.special_peft_forward_args=frozenset()
    wrapped.active_peft_config=SimpleNamespace(is_prompt_learning=False,peft_type='LORA')
    scope_class=old_scope if case.get('old') else new_scope
    args=(base,metadata,g,87,16,core,native_fixed,original_loss,case['name'])
    scope=scope_class(*args) if case.get('old') else scope_class(*args,observer)
    exception=None;returned=None
    try:
        with scope:
            if case.get('direct'):
                returned=base.loss_function(logits=logits,labels=labels,vocab_size=vocab,**case.get('kwargs',{}))
            else:
                returned=wrapped.forward(input_ids=input_ids,attention_mask=attention_mask,labels=labels,
                    use_cache=False,**case.get('kwargs',{}))
    except Exception as error:
        exception={'type':type(error).__name__,'message':str(error),
            'frames':[{'file':frame.filename,'line':frame.lineno,'function':frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)]}
    restored=base.loss_function is original_loss and not hasattr(base,'_patient_balanced_active_scope') and scope.closed
    accepted=exception is None
    exact_core=(len(core_calls)==1 and all(core_calls[0][k] for k in
        ['logits_same','labels_same','row_is_scope_copy','row_content_unchanged','row_is_not_caller_mutable_object','native_fixed_same'])
        and (core_calls[0]['actual_group_size'],core_calls[0]['patients'],core_calls[0]['groups'])==(g,87,16)
        and core_calls[0]['kwargs']=={'alpha':.5} and core_calls[0]['positional_arguments']==7)
    if case.get('direct'): returned_loss_same=returned is loss_result
    elif isinstance(returned,tuple): returned_loss_same=returned[0] is loss_result
    else: returned_loss_same=getattr(returned,'loss',None) is loss_result
    diagnostics=None if not observer_calls else observer_calls[0]['extra_kwargs']
    checks={'expected_acceptance':accepted==case['accept'],'scope_restored':restored,'caller_metadata_unchanged':metadata==metadata_before,
        'core_called_only_when_accepted':len(core_calls)==int(case['accept']),
        'accepted_arguments_and_result_exact':not case['accept'] or (exact_core and returned_loss_same),
        'rejection_is_parameter_assertion':case['accept'] or (exception is not None and exception['type']=='AssertionError'),
        'observer_before_guard_and_core':case.get('old',False) or
            (len(observer_calls)==1 and events[0]=='observer' and events==(['observer','core'] if case['accept'] else ['observer'])),
        'observer_reports_pre_loss':case.get('old',False) or
            (bool(observer_calls) and observer_calls[0]['no_loss_computed_yet'] is True)}
    if not case.get('direct'):
        checks['real_chain_once']=len(text_calls)==len(head_calls)==len(alora_calls)-before_alora==1
        checks['return_dict_consumed_before_qwen_loss']=not observer_calls or 'return_dict' not in diagnostics
        checks['original_input_identity_to_fake_backbone']=bool(text_calls) and text_calls[0]['input_ids_same'] and text_calls[0]['attention_mask_same'] and text_calls[0]['use_cache_is_false']
        checks['original_complete_head_selection']=hidden.slices==[(slice(None),slice(0,None),slice(None))]
        if not case.get('old'):
            expected_diagnostics={k:v for k,v in case.get('kwargs',{}).items() if k not in {'return_dict','num_items_in_batch','ignore_index','shift_labels'}}
            expected_diagnostics.setdefault('output_attentions',None);expected_diagnostics.setdefault('output_hidden_states',None)
            checks['two_default_diagnostics_and_no_other_implicit_kwargs']=diagnostics=={
                k:{'is_none':v is None,'type':type(v).__name__} for k,v in expected_diagnostics.items()}
    observed={'case':case['name'],'group_size':g,'expected_accept':case['accept'],'old_hook':case.get('old',False),
        'accepted':accepted,'exception':exception,'events':events,'observer_calls':observer_calls,'core_calls':core_calls,
        'fake_backbone_calls':text_calls,'fake_head_calls':head_calls,'scope_restored':restored,
        'checks':checks,'passed':all(checks.values()),'synthetic_identity_and_interface_only':True}
    append(out/'observations.jsonl',observed)
    assert observed['passed'],case['name']
    return observed


def worker(plan_path,out):
    started=time.monotonic();plan=json.loads(plan_path.read_text())
    try:
        assert plan['schema']=='patient-balanced-interface-contract-plan-v1'
        assert (plan['budget_seconds'],plan['grace_seconds'])==(10,1)
        for ref in [plan['source'],*plan['references'].values()]:
            assert Path(ref['path']).stat().st_size==ref['bytes'] and sha(ref['path'])==ref['sha256'],ref['path']
        assert sha(__file__)==plan['source']['sha256']
        assert not any(n.split('.')[0] in {'torch','transformers','peft'} for n in sys.modules)
        source_manifest=json.loads(Path(plan['references']['source_manifest']['path']).read_text())
        for key in ['peft_forward','return_decorator']:
            assert source_manifest[key]['extracted']['sha256']==plan['references'][key]['sha256']
        old_scope=load_scope(plan['references']['old_hook']['path'])
        new_scope=load_scope(plan['references']['new_hook']['path'])
        chain=extract_chain(plan)
        write(out/'source-extraction.json',chain[3])
        cases=[]
        for g in [4,8]:
            cases.extend([
                {'name':f'old_real_chain_default_reject_g{g}','old':True,'group_size':g,'accept':False},
                {'name':f'old_direct_empty_kwargs_accept_g{g}','old':True,'direct':True,'group_size':g,'accept':True},
                {'name':f'new_real_chain_default_accept_g{g}','group_size':g,'accept':True},
                {'name':f'new_explicit_None_return_None_g{g}','group_size':g,'accept':True,
                 'kwargs':{'output_attentions':None,'output_hidden_states':None,'return_dict':None}},
                {'name':f'new_decorator_return_tuple_g{g}','group_size':g,'accept':True,'kwargs':{'return_dict':False}},
                {'name':f'new_direct_no_diagnostic_keys_g{g}','direct':True,'group_size':g,'accept':True}])
        for key,value in [('unexpected_kwarg',None),('unexpected_kwarg',False),
            ('output_attentions',True),('output_attentions',False),('output_attentions',0),
            ('output_hidden_states',True),('output_hidden_states',False),('output_hidden_states',0),
            ('num_items_in_batch',1),('ignore_index',-99),('shift_labels','synthetic-no-target')]:
            cases.append({'name':f'reject_{key}_{str(value)}','group_size':8,'accept':False,'kwargs':{key:value}})
        for name,change in [('bad_vocab',{'vocab':8}),('bad_dtype',{'dtype':'torch.float32'}),
                            ('bad_device',{'device':'cpu'}),('bad_sequence',{'shape':(1,6,7)})]:
            cases.append({'name':name,'group_size':4,'accept':False,**change})
        assert len(cases)==27 and len({c['name'] for c in cases})==27
        observations=[run_case(c,old_scope,new_scope,chain,out) for c in cases]
        assert not any(n.split('.')[0] in {'torch','transformers','peft'} for n in sys.modules)
        write(out/'result.json',{'status':'PASS_PURE_PYTHON_REAL_FORWARD_INTERFACE_CONTRACT',
            'cases':len(observations),'all_passed':all(o['passed'] for o in observations),
            'references':plan['references'],'source':reference(__file__),'plan':reference(plan_path),
            'observations':reference(out/'observations.jsonl'),'observer_calls':reference(out/'observer-calls.jsonl'),
            'source_extraction':reference(out/'source-extraction.json'),'elapsed_seconds':time.monotonic()-started,
            'real_forward_bodies_executed_with_fakes':True,'real_can_return_tuple_decorator_executed':True,
            'core_recorder_only_no_loss_math':True,'no_Torch_Transformers_PEFT_import':True,
            'no_real_target_values_or_tokenization_or_remote_or_GPU':True,
            'scope':'ordinary LoRA, no prompt/POLY/MPT/aLoRA; identity aLoRA helper and null adapter-hook context',
            'not_proven':['real tensor arithmetic','gradient/checkpoint recomputation','model output','LoRA/Adam update','quality']})
    except BaseException as error:
        write(out/'failure.json',{'exception':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();plan=args.plan.resolve();out=args.output.resolve()
    if args.worker:
        worker(plan,out);return
    assert not out.exists(),'Single-use output; preserve any prior attempt, no automatic retry'
    config=json.loads(plan.read_text());assert (config['budget_seconds'],config['grace_seconds'])==(10,1)
    assert out==Path(config['output_directory']).resolve(),'Output must match the frozen independent namespace'
    assert config['source']['sha256']==sha(__file__)
    out.mkdir(parents=True);sync_parents(out)
    command=[sys.executable,'-I',str(Path(__file__).resolve()),'--worker','--plan',str(plan),'--output',str(out)]
    write(out/'intent.json',{'command':command,'plan':reference(plan),'source':reference(__file__),
        'budget_seconds':10,'grace_seconds':1,'no_automatic_retry':True})
    started=time.monotonic();timed_out=False
    with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=environment)
        write(out/'child-started.json',{'pid':child.pid,'started_unix_seconds':time.time()})
        try: code=child.wait(timeout=max(.001,10-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try: code=child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
    files={p.name:reference(p) for p in sorted(out.iterdir()) if p.is_file()}
    write(out/'terminal.json',{'returncode':code,'timed_out':timed_out,'seconds':time.monotonic()-started,
        'budget_seconds':10,'grace_seconds':1,'files':files,'original_child_pid':child.pid})
    if code!=0 or timed_out:raise SystemExit(1)


if __name__=='__main__':main()
