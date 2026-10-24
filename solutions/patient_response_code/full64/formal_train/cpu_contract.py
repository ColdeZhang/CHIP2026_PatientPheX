"""Stdlib full64 binding/schedule/resume/activation contracts; no model/tokenizer."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, math, random, sys, tempfile, time
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent


def ref(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module
def function(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
def treefn(tree, name, ns):
    exec(compile(ast.Module(body=[function(tree,name)],type_ignores=[]),name,'exec'),ns); return ns[name]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True,type=Path); a=ap.parse_args()
    start=time.monotonic(); p=json.loads(a.plan.read_text()); checks=[]
    def okay(name, result=True):
        if not result: raise AssertionError(name)
        checks.append(dict(name=name,pass_=True))
    def reject(name, call):
        try: call()
        except (ValueError,KeyError,AssertionError,FileNotFoundError,TimeoutError):okay(name);return
        raise AssertionError('Invalid fixture accepted: '+name)
    for item in list(p['source_files'].values())+list(p['data_arguments'].values())+[p['adapted_trainer'],p['expected_probe_plan']]:
        okay('exact byte identity '+Path(item['path']).name,ref(item['path'])==item)
    text=(B/'train.py').read_text(); t=ast.parse(text); old=ast.parse(Path(p['adapted_trainer']['path']).read_text()); dump=lambda n:ast.dump(n,include_attributes=False)
    for name in ['semantic_configuration','batch_plan','verify_fingerprint','validate_stop_epoch','epoch_outcome','collate','model_files']:
        okay('unchanged '+name+' AST',dump(function(t,name))==dump(function(old,name)))
    micro=lambda root:next(n for n in ast.walk(root) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='indices')
    okay('complete indexed forward/rowmean/group-normalized backward AST unchanged',dump(micro(function(t,'run')))==dump(micro(function(old,'run'))))
    restore=lambda root:next(n for n in ast.walk(root) if isinstance(n,ast.If) and ast.unparse(n.test)=='checkpoint' and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='load_state_dict' for c in ast.walk(n)))
    expected=copy.deepcopy(restore(function(old,'run')))
    for n in ast.walk(expected):
        if isinstance(n,ast.Constant) and type(n.value) is int and n.value==16:n.value=34
    okay('complete actual state restore AST changes only epoch-step count16to34',dump(restore(function(t,'run')))==dump(expected))
    ns={'random':random}
    for n in t.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_SHA'):ns[n.targets[0].id]=ast.literal_eval(n.value)
    batch=treefn(t,'batch_plan',ns); treefn(t,'native_training_rows',ns); validate=treefn(t,'validate_response_metadata',ns)
    for epoch in [0,1]:
        g=batch(265,1,8,20261023,epoch); flat=[i for group in g for item in group for i in item]
        expected_order=list(range(265)); random.Random(20261023+epoch).shuffle(expected_order)
        okay('actual265 once and original seeded order epoch'+str(epoch),flat==expected_order and sorted(flat)==list(range(265)))
        okay('33 complete groups8 plus tail1 epoch'+str(epoch),[len(x) for x in g]==[8]*33+[1])
    steps=math.ceil(265/8);total=steps*2;warmup=math.ceil(.05*total)
    okay('derived original recipe34x2 and ceil(.05*68)=4',(steps,total,warmup)==(34,68,4))
    okay('scheduler and manifest use derived counts','get_linear_schedule_with_warmup(optimizer,4,68)' in text and "'steps_per_epoch':34" in text and "'warmup_steps':4" in text)
    structural=json.loads(Path(p['data_arguments']['response_preparation']['path']).read_text())
    native=json.loads(Path(p['data_arguments']['native_receipt']['path']).read_text())
    audit=json.loads(Path(p['data_arguments']['native_row_audit']['path']).read_text())
    original=json.loads(Path(p['data_arguments']['data_manifest']['path']).read_text())
    train_ids,held_ids=original['training_document_ids'],original['excluded_document_ids']
    okay('actual full64 structure/native/excludeddev identity',validate(structural,native,audit,train_ids,held_ids))
    okay('all four candidate generators exclude own targets and originaldev',len(original['provenance'])==4 and all(not set(x['train_ids'])&set(x['input_ids']) and not set(held_ids)&(set(x['train_ids'])|set(x['input_ids'])) for x in original['provenance']))
    okay('originaldev16 exact exclusion source',original['excluded_dev_input_sha256']==p['data_arguments']['excluded_input']['sha256'])
    for key,val in [('rows',124),('total_tokens',427560),('max_tokens',5910),('supervised_tokens',8138),('encoded_sha256','0'*64)]:
        bad=copy.deepcopy(native);bad['groups']['train']['arms']['response_code'][key]=val
        reject('reject stale C32 native '+key,lambda bad=bad:validate(structural,bad,audit,train_ids,held_ids))
    for key in ['all_original_users_UTF8_preserved','all_train_targets_inverse_exact','train64_originaldev16_disjoint','all_candidate_ancestors_exclude_originaldev']:
        bad=copy.deepcopy(structural);bad[key]=False
        reject('reject structural failure '+key,lambda bad=bad:validate(bad,native,audit,train_ids,held_ids))
    bad=copy.deepcopy(audit);next(x for x in bad if x['group']=='train' and x['arm']=='response_code')['prefix_mask_exact']=False
    reject('reject changed label mask proof',lambda:validate(structural,native,bad,train_ids,held_ids))
    contract=load(B/'runtime_contract.py','full64_runtime_contract');names=['A','B']
    state=dict(completed_epochs=1,global_step=34,optimizer={'state':{i:dict(step=34,exp_avg=[.1],exp_avg_sq=[.2]) for i in range(2)},'param_groups':[{'params':[0,1]}]},scheduler={'last_epoch':34,'_step_count':35},python_rng=[1],numpy_rng=[2],torch_rng=[3],cuda_rng=[[4]],optimizer_parameter_names=names)
    okay('new complete first epoch state structure accepted',contract.validate_state_structure(state,1,names))
    for key in contract.STATE_KEYS:
        bad=copy.deepcopy(state);del bad[key]
        reject('missing fullstate field '+key,lambda bad=bad:contract.validate_state_structure(bad,1,names))
    bad=copy.deepcopy(state);bad['global_step']=16
    reject('C32 step16 checkpoint rejected',lambda:contract.validate_state_structure(bad,1,names))
    for last,count in [(0,1),(16,17)]:
        bad=copy.deepcopy(state);bad['scheduler'].update(last_epoch=last,_step_count=count)
        reject('reset or old scheduler rejected '+str(last),lambda bad=bad:contract.validate_state_structure(bad,1,names))
    bad=copy.deepcopy(state);del bad['optimizer']['state'][0]['exp_avg_sq']
    reject('missing Adam second moment rejected',lambda:contract.validate_state_structure(bad,1,names))
    reject('shared absolute deadline cannot reset',lambda:contract.check_deadline(time.time()-1))
    for fragment in ["args.absolute_deadline_unix-time.time()>5400","helpers.json_sha(row['input_ids'])!=actual['input_ids_sha256']","helpers.json_sha(row['labels'])!=actual['labels_sha256']","torch.__version__!='2.6.0+cu124'","'CPU_census_torch':'2.7.1+cu126'","scheduler.last_epoch!=global_step"]:
        okay('fixed runtime/source '+fragment,fragment in text)
    C=B.parent/'controller';activation=load(C/'activation.py','full64_activation')
    with tempfile.TemporaryDirectory(prefix='full64-gate-contract-') as temporary:
        root=Path(temporary);w=root/'wrapper.json';d=root/'cost.json'
        w.write_text(json.dumps(dict(returncode=1,timed_out=False,all12_backwards_and_one_update_verified=False,target_has_no_process=True)));d.write_text('{}')
        reject('actual failed probe cannot activate',lambda:activation.require_actual_probe(p,w,d))
        reject('missing actual probe cannot activate',lambda:activation.require_actual_probe(p,root/'missing',d))
    prepare=(C/'prepare.py').read_text()
    okay('actual probe gate before deployable directory/authorization',prepare.index('gate=require_actual_probe')<prepare.index('B.mkdir(exist_ok=False)')<prepare.index("save(B/'GPU-AUTHORIZATION.json'"))
    for name in ['operations.py','epoch_driver.py','io_driver.py','watch_authorized_campaign.py','watch_state_recovery.py','bootstrap.py']:
        s=(C/name).read_text();ast.parse(s)
        okay('fixed5400/no old3600 '+name,'5400' in s and '3600' not in s)
    watch=(C/'watch_authorized_campaign.py').read_text()
    okay('read-only LUNA UNKNOWN recovery retained','safe_read=' in watch and "'LUNA_OFFLINE'" in watch and "'UNKNOWN_OUTCOME'" in watch and 'unknown mutation never repeated' in watch)
    okay('HTTP has own original export deadline',"while time.time()<a.deadline:server.handle_request()" in (C/'export_checkpoint_server.py').read_text())
    okay('no tokenizer/Torch/model imports',all(x not in sys.modules for x in ['torch','transformers','tokenizers']))
    result=dict(status='PASS_STDLIB_FULL64_DATA_SCHEDULE_RESUME_AND_DISABLED_ACTIVATION_CONTRACT',checks_count=len(checks),checks=checks,seconds=time.monotonic()-start,source=ref(B/'train.py'),runtime_contract=ref(B/'runtime_contract.py'),cpu_source=ref(__file__),plan_sha256=ref(a.plan)['sha256'],actual_native_repeated=False,raw_training_target_parsed=False,model_GPU_remote_quality=False,formal_GPU_authorization_created=False,scope='New binding/schedule/epoch guards and copied mathematical source; future actual full-state restoration remains runtime assertion, not new GPU numerical equivalence proof.')
    with (B/'CPU-contract-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','checks_count','seconds']}))


if __name__=='__main__':main()
