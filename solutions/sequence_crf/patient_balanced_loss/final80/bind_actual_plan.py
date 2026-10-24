#!/usr/bin/env python3
"""Bind final80 to collected real final epoch2 metadata; local files only.

Required: --formal-directory (collection.json, package.json, terminal tree) and
--output-directory (new review/binding directory). --formal-plan defaults to the
sibling frozen formal-plan.json. Without --apply, write a nondeployable preview
with projected canonical source refs; do not change final_infer or input-sources.
With --apply, retain both original files, replace exactly three UNBOUND values,
update input-sources, and check the real controller's pure payload/file functions.
No deployment, remote inventory, model, tokenizer, gold or scoring entry is called.
Any partial apply is retained for explicit inspection, never automatically retried.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import traceback

PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
FORMAL=PROJECT+'/sequence-round2/patient-balanced-alpha05/formal-v1'
ROOT=PROJECT+'/sequence-round2/patient-balanced-alpha05/final80-v1'
TRAINER='3eab37084ad45a7e0acde7ce1d1c32fd3023825490f0d87aa76aa4187c1c965a'
TRAIN='14af86cffe1a8b87af022d13e42f9b910281f0da478e3cc731bc2fbc9ba68c3b'
HOOK='7b9900fc4896f53b02f4edcbbd94799258c3bb8f91fec7c3b3768054d4690da8'
UNBOUND_SOURCE='4575c2113b38f8291b1e7770eb43999ef085f4465d5c6cf744cfda77b3b7a5ce'
UNBOUND_MAP='e7e102fad416f65f27a2cfee9785e08aaef1f6e07bac88bc456c3cb6d724d26b'
SLOTS={
    'training_complete':('TRAINING_COMPLETE_SHA','UNBOUND_ACTUAL_ALPHA05_TRAINING_COMPLETE_SHA256','training_complete.json'),
    'training_checkpoint':('TRAINING_CHECKPOINT_SHA','UNBOUND_ACTUAL_ALPHA05_EPOCH002_CHECKPOINT_SHA256','epoch-002/checkpoint.json'),
    'training_durability':('TRAINING_DURABILITY_SHA','UNBOUND_ACTUAL_ALPHA05_EPOCH002_DURABILITY_SHA256','durability-epoch-002.json'),
}


def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def digest(raw):return hashlib.sha256(raw).hexdigest()
def projected(path,raw):return dict(path=str(Path(path).resolve()),bytes=len(raw),sha256=digest(raw))
def ref(path):return projected(path,Path(path).read_bytes())
def identity(item):return {key:item[key] for key in ('bytes','sha256')}


def checked(item):
    assert ref(item['path'])==item,item
    return Path(item['path'])


def safe(name):
    p=PurePosixPath(name)
    assert name and not p.is_absolute() and p.as_posix()==name and '..' not in p.parts and '\\' not in name
    return p


def sync(directory):
    directory=Path(directory).resolve()
    while True:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if directory.parent==directory:break
        directory=directory.parent


def publish(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync(path.parent)


def save(path,value):publish(path,encoded(value))


def replace_checked(path,raw,before):
    path=Path(path);assert ref(path)==before and not path.is_symlink()
    temporary=path.with_name(path.name+'.actual-binding.pending')
    publish(temporary,raw);assert ref(path)==before
    os.replace(temporary,path);sync(path.parent)
    assert ref(path)==projected(path,raw)


def inspect_formal(directory,plan_path):
    plan=read(plan_path);package_path=directory/'package.json';package=read(package_path)
    assert package['plan']==ref(plan_path) and package['remote']==FORMAL and package['mount']==MOUNT
    assert package['members']['formal-plan.json']==ref(plan_path)
    assert plan['phase']=='train' and plan['formal_training_authorized'] is True
    assert (plan['epochs'],plan['rows'],plan['optimizer_calls'],plan['warmup_steps'],plan['alpha'])==(2,124,32,2,.5)
    assert plan['trainer_sha256']==TRAINER and plan['output_dir']==FORMAL+'/formal2ep'
    assert plan['storage_parent']==PROJECT
    collection_path=directory/'collection.json';collection=read(collection_path)
    assert collection['schema']=='patient-balanced-formal-metadata-collection-v1'
    for field in ['formal_reported_success','all_actual_SHA_verified','files_and_ancestor_directories_fsynced',
                  'zip_safety_checked','all_original_nonweight_output_files_included']:
        assert collection[field] is True,field
    assert collection['package']==ref(package_path) and collection['original_remote']==FORMAL and collection['mount']==MOUNT
    terminal=(directory/'terminal').resolve();assert Path(collection['local']).resolve()==terminal
    index=read(checked(collection['member_index']))
    assert checked(collection['member_index']).resolve()==terminal/'member-index.json'
    assert index['remote_root']==FORMAL and index['package']==ref(package_path)
    assert index['original_files_fsynced'] and index['child_exit']['no_running_process']
    assert collection['child_exit_before_freeze']==index['child_exit']
    assert set(index['files'])==set(collection['files'])
    for name,item in collection['files'].items():
        relative=safe(name);path=checked(item)
        assert path.resolve()==terminal/relative and not path.is_symlink()
        assert identity(item)==identity(index['files'][name])
        assert index['files'][name]['original_path']==FORMAL+'/'+name
    def item(name):return collection['files'][name]
    wrapper=read(item('wrapper-complete.json')['path'])
    assert wrapper==collection['wrapper']==index['wrapper']
    assert wrapper['returncode']==0 and wrapper['timed_out'] is False and 'endpoint_error' not in wrapper
    for field in ['all_epochs_persistent_and_readback_verified','no_pending_export','no_probe_state_or_old_adapter_used']:
        assert wrapper[field] is True,field
    driver=read(item('driver-started.json')['path']);child=read(item('child-started.json')['path'])
    assert driver['plan_sha256']==ref(plan_path)['sha256']
    assert child['driver_pid']==driver['driver_pid'] and child['child_pid']==wrapper['child_pid']==index['child_exit']['pid']
    external=collection['actual_external_checkpoints'];assert external==index['external_checkpoints']
    expected_large=set();acks={};checkpoints={}
    for epoch in (1,2):
        prefix='formal2ep/epoch-%03d/'%epoch
        cp_ref=item(prefix+'checkpoint.json');cp=read(cp_ref['path'])
        ack_ref=item('formal2ep/durability-epoch-%03d.json'%epoch);ack=read(ack_ref['path'])
        assert cp['completed_epochs']==ack['completed_epochs']==epoch
        assert cp['global_step']==ack['global_step']==16*epoch and cp['smoke_only'] is False
        fp=cp['fingerprint']
        assert fp['source_sha256']==ack['source_sha256']==TRAINER and fp['train_sha256']==TRAIN
        assert fp['schema']=='patientphex-qwen35-meta32-patient-balanced-alpha05-lora-v1'
        assert fp['objective']['alpha']==.5 and fp['row_loss_scope_sha256']==HOOK
        assert ack['status']=='PASS_file_and_directory_fsync_rename_and_reopened_SHA'
        assert ack['checkpoint']==FORMAL+'/'+prefix.rstrip('/') and ack['persistent_root']==PROJECT
        assert ack['checkpoint_metadata_sha256']==cp_ref['sha256']
        for field in ['file_directory_fsync','reopened_SHA_all_exact','all_members_reopened_and_SHA_verified']:
            assert ack[field] is True,field
        ancestors=[];parent=Path(FORMAL+'/formal2ep')
        while True:
            ancestors.append(str(parent))
            if str(parent)==PROJECT:break
            assert PROJECT in str(parent);parent=parent.parent
        assert ack['ancestor_directories_fsynced']==ancestors
        assert wrapper['durability'][str(epoch)]==dict(path=FORMAL+'/formal2ep/durability-epoch-%03d.json'%epoch,
            sha256=ack_ref['sha256'],receipt=ack)
        assert {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(cp['files'])
        assert set(ack['files'])==set(cp['files'])|{'checkpoint.json'}
        assert identity(cp_ref)==ack['files']['checkpoint.json']
        for name,sha in cp['files'].items():
            safe(name);member=ack['files'][name];assert member['sha256']==sha
            relative=prefix+name
            if name in ['training_state.pt','adapter/adapter_model.safetensors']:
                expected_large.add(relative);large=external[relative]
                assert identity(large)==member and large['original_path']==FORMAL+'/'+relative
                assert large['not_downloaded'] and large['actual_fsynced_and_reopened_SHA']
            else:assert identity(item(relative))==member
        acks[str(epoch)]=ack_ref;checkpoints[str(epoch)]=cp_ref
    assert set(external)==expected_large
    complete_ref=item('formal2ep/training_complete.json');complete=read(complete_ref['path'])
    assert complete['epochs']==2 and complete['global_step']==32 and complete['smoke_only'] is False
    assert complete['source_sha256']==TRAINER and wrapper['training_complete_sha256']==complete_ref['sha256']
    receipts={key:dict(local=item('formal2ep/'+relative),remote=FORMAL+'/formal2ep/'+relative)
              for key,(_,_,relative) in SLOTS.items()}
    return plan,dict(collection=ref(collection_path),terminal=item('wrapper-complete.json'),
        member_index=collection['member_index'],formal_plan=ref(plan_path),formal_package=ref(package_path),
        receipts=receipts,epoch_ACKs=acks,epoch_checkpoint_metadata=checkpoints,external_checkpoint_members=external,
        adapter=FORMAL+'/formal2ep/epoch-002/adapter',both_actual_epoch_ACKs_verified=True,
        original_physical_training_paths_retained=True,no_large_state_or_weight_download=True)


def pure_controller(path):
    """Extract only pure local payload schema functions; never import the controller."""
    tree=ast.parse(path.read_text());names={'digest','ref','identity','safe_name','checked_plan','payload_sources','expected_remote_files'}
    constants={'MOUNT','PROJECT','REL','ROOT','INPUT_NAMES','SOURCE_NAMES'}
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names)
           or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets))]
    assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names
    ns=dict(__file__=str(path),hashlib=hashlib,json=json,Path=Path,PurePosixPath=PurePosixPath)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    assert ns['ROOT']==ROOT and ns['MOUNT']==MOUNT
    return ns


def main():
    assert __debug__
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--formal-directory',required=True,type=Path)
    ap.add_argument('--output-directory',required=True,type=Path)
    ap.add_argument('--formal-plan',type=Path,default=here.parent/'formal-plan.json')
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args();directory=args.formal_directory.resolve();out=args.output_directory.resolve()
    assert not out.exists(),'Use a fresh binding directory; preserve any prior attempt'
    formal,evidence=inspect_formal(directory,args.formal_plan.resolve())
    source_path=here/'final_infer.py';map_path=here/'input-sources.json'
    before_source=ref(source_path);before_map=ref(map_path)
    assert before_source['sha256']==UNBOUND_SOURCE and before_map['sha256']==UNBOUND_MAP
    original_source=source_path.read_bytes();original_map=map_path.read_bytes()
    mapping=json.loads(original_map);assert mapping['sources']['final_infer.py']==before_source
    for value in mapping['sources'].values():checked(value)
    for value in mapping['inputs'].values():checked(value['local'])
    source=original_source.decode();slots={}
    for key,(name,placeholder,relative) in SLOTS.items():
        actual=evidence['receipts'][key];sha=actual['local']['sha256']
        old=name+' = '+json.dumps(placeholder);new=name+' = '+json.dumps(sha)
        assert source.count(old)==1;source=source.replace(old,new)
        slots[key]=dict(path=actual['remote'],bytes=actual['local']['bytes'],sha256=sha,
            expected_basename=relative,local_receipt=actual['local'])
    bound_source=source.encode();old_tree=ast.parse(original_source);new_tree=ast.parse(bound_source)
    for tree in [old_tree,new_tree]:
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {v[0] for v in SLOTS.values()} for t in node.targets):
                node.value=ast.Constant(value='ONLY_ACTUAL_SHA_SLOT')
    assert ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False)
    mapping['sources']['final_infer.py']=projected(source_path,bound_source)
    mapping.update(status='ACTUAL_FINAL_EPOCH2_RECEIPTS_BOUND',actual_final_receipt_slots=slots,
        execution_blocked_until_all_three_actual_SHA_constants_are_bound=False,actual_training=evidence,
        current_actions='Local actual metadata binding only; no remote/model/tokenizer/quality execution',
        binding_source=ref(__file__))
    bound_map=encoded(mapping);input_ref=projected(map_path,bound_map)
    environment=copy.deepcopy(formal['environment'])
    assert environment['CUDA_VISIBLE_DEVICES']==formal['gpu_uuid']
    assert environment['PATIENTPHEX_DURABILITY_ROOT']==PROJECT and not {'MCP_TOKEN','MCP_URL'}&set(environment)
    argv=[formal['python'],'-u',ROOT+'/final_infer.py','--model',formal['model'],
        '--source-manifest',ROOT+'/model-source-manifest.json','--prompts',ROOT+'/prompts.jsonl',
        '--output',ROOT+'/prediction/responses.jsonl','--adapter',evidence['adapter'],
        '--token-audit',ROOT+'/input-token-audit.json','--training-checkpoint',slots['training_checkpoint']['path'],
        '--training-complete',slots['training_complete']['path'],'--training-durability',slots['training_durability']['path'],
        '--max-input-tokens','14000','--max-new-tokens','1600']
    plan=dict(schema='patient-balanced-alpha05-final80-v1',phase='infer',inference_authorized=args.apply,
        status='ACTUAL_BOUND_READY_FOR_SOURCE_PLAN_PUSH' if args.apply else 'PREVIEW_ONLY_NOT_DEPLOYABLE_NO_SOURCE_APPLIED',
        budget_seconds=600,grace_seconds=30,expected_rows=80,fresh_only=True,strict_all80=True,alpha=.5,
        python=formal['python'],gpu_uuid=formal['gpu_uuid'],model=formal['model'],storage_parent=PROJECT,
        output_dir=ROOT+'/prediction',environment=environment,argv=argv,input_sources=input_ref,
        driver_source=ref(here/'infer_driver.py'),operations_source=ref(here/'infer_remote.py'),
        inference_source_sha256=mapping['sources']['final_infer.py']['sha256'],
        external_training_files={value['path']:identity(value) for value in slots.values()},
        training_actual=evidence,reuse_packages=[evidence['formal_package']]+formal.get('reuse_packages',[]),
        runtime_difference=formal['runtime_difference'],binding_source=ref(__file__),
        no_automatic_deploy_start_inference_or_score=True,source_and_plan_push_required_before_remote=True)
    items={name:value['local'] for name,value in mapping['inputs'].items()};items.update(mapping['sources'])
    items.update({'infer_driver.py':plan['driver_source'],'input-sources.json':input_ref});assert len(items)==9
    plan['files']={ROOT+'/'+name:identity(value) for name,value in items.items()}
    plan['files'].update(plan['external_training_files']);assert len(plan['files'])==12
    out.mkdir(parents=True,exist_ok=False);sync(out.parent)
    save(out/'intent.json',dict(apply=args.apply,at_utc=datetime.now(timezone.utc).isoformat(),
        before_source=before_source,before_input_sources=before_map,source=ref(__file__),training_actual=evidence))
    try:
        publish(out/'original/final_infer.py',original_source);publish(out/'original/input-sources.json',original_map)
        publish(out/'proposed/final_infer.py',bound_source);publish(out/'proposed/input-sources.json',bound_map)
        if args.apply:
            replace_checked(source_path,bound_source,before_source);replace_checked(map_path,bound_map,before_map)
            controller=pure_controller(here/'infer_remote.py')
            assert controller['payload_sources'](plan)==items
            assert controller['expected_remote_files'](plan)==plan['files']
        save(out/'final80-plan.json',plan)
        if args.apply:assert controller['checked_plan'](out/'final80-plan.json')==plan
        save(out/'binding-result.json',dict(applied=args.apply,plan=ref(out/'final80-plan.json'),
            source=ref(__file__),training_actual=evidence,three_SHA_literals_only_AST_exact=True,
            actual_source=ref(source_path),actual_input_sources=ref(map_path),
            prospective_source=mapping['sources']['final_infer.py'],prospective_input_sources=input_ref,
            controller_payload_and_files_checked=args.apply,no_remote_or_model_or_quality=True))
        print(json.dumps(dict(plan=ref(out/'final80-plan.json'),result=ref(out/'binding-result.json'),applied=args.apply)))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,
            source_now=ref(source_path),input_sources_now=ref(map_path),no_remote_or_model_or_quality=True))
        raise


if __name__=='__main__':main()
