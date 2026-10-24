#!/usr/bin/env python3
"""Incremental standard-library source and transition checks; no model execution."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

ROOT=Path('/workspace/patientphex-runs/round2-20261023')
B=ROOT/'research/candidate-order-warmstart-entry'
OUT=ROOT/'validation/candidate-order-warmstart-entry-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
ref=lambda p:dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))

def main():
    inventory=load(B/'trainer-source-manifest.json')
    for record in inventory['files']:
        p=B/record['path'];assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
    assert sha(B/'train_order_warmstart.py')=='93301f6070f624672d8331b8fda7304089a2af44b95c83424513ef4895873ab4'
    old=ast.parse((B/'upstream-trainer.py').read_text());new=ast.parse((B/'train_order_warmstart.py').read_text())
    compile(new,str(B/'train_order_warmstart.py'),'exec')
    of={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    nf={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    dump=lambda n:ast.dump(n,include_attributes=False)
    identical=['training_rows','verify_sources_and_data','model_files','validate_checkpoint','batch_plan','verify_fingerprint','validate_stop_epoch','epoch_outcome','collate']
    for name in identical:assert dump(of[name])==dump(nf[name]),name
    def loadblock(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='checkpoint' and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='torch' and c.func.attr=='load' for stmt in n.body for c in ast.walk(stmt)))
    ob,nb=loadblock(of['run']),loadblock(nf['run'])
    assert dump(ast.Module(body=ob.body,type_ignores=[]))==dump(ast.Module(body=nb.body,type_ignores=[]))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' for stmt in nb.orelse for n in ast.walk(stmt))
    def lossbody(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='indices')
    assert dump(lossbody(of['run']))==dump(lossbody(nf['run']))
    namespace=dict(Path=Path,json=json,random=random,sha256=sha,CONFIG={'sentinel':'same'})
    selected=['verify_export_ack','validate_stage_transition','semantic_configuration','order_variant','validate_checkpoint','verify_fingerprint']
    exec(compile(ast.Module(body=[nf[n] for n in selected],type_ignores=[]),'<frozen-pure-functions>','exec'),namespace)
    transition=namespace['validate_stage_transition']
    for args in [('cpu',None,None,None),('train',None,1,None),('train','own/epoch-001',2,'ack')]:transition(*args)
    rejects=[]
    for args in [('train',None,2,None),('train',None,None,None),('train',None,1,'ack'),('train','own/epoch-001',2,None),('train','own/epoch-001',1,'ack'),('cpu',None,None,'ack')]:
        try:transition(*args)
        except ValueError:rejects.append(list(args))
        else:raise AssertionError('transition accepted '+str(args))
    initial=SimpleNamespace(mode='train',order_arm='original',resume=None,stop_after_epoch=1,export_ack=None)
    resumed=SimpleNamespace(mode='train',order_arm='original',resume='own/epoch-001',stop_after_epoch=2,export_ack='ack')
    assert namespace['semantic_configuration'](initial)==namespace['semantic_configuration'](resumed)
    resumed.order_arm='reverse-first'
    assert namespace['semantic_configuration'](initial)!=namespace['semantic_configuration'](resumed)
    assert [namespace['order_variant']('original',e) for e in range(2)]==['O','O']
    assert [namespace['order_variant']('reverse-first',e) for e in range(2)]==['R','O']
    cases=[]
    with tempfile.TemporaryDirectory(dir=OUT,prefix='ack-fixture-') as tmp:
        run=Path(tmp);cp=run/'epoch-001';(cp/'adapter').mkdir(parents=True)
        for name,value in {'adapter/adapter_model.safetensors':b'fixture-weights','adapter/adapter_config.json':b'{}','training_state.pt':b'fixture-Adam-scheduler-RNG'}.items():(cp/name).write_bytes(value)
        (run/'run_manifest.json').write_text('{"synthetic":true}')
        (run/'progress.json').write_text('{"epoch":1}')
        meta=dict(completed_epochs=1,global_step=16,smoke_only=False,files={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()})
        (cp/'checkpoint.json').write_text(json.dumps(meta))
        inventory={n:dict(bytes=(cp/n).stat().st_size,sha256=s) for n,s in meta['files'].items()}
        inventory['checkpoint.json']=dict(bytes=(cp/'checkpoint.json').stat().st_size,sha256=sha(cp/'checkpoint.json'))
        inventory['_run/run_manifest.json']=dict(bytes=(run/'run_manifest.json').stat().st_size,sha256=sha(run/'run_manifest.json'))
        ack=dict(status='persistent_checkpoint_verified',checkpoint_sha256=sha(cp/'checkpoint.json'),completed_epochs=1,global_step=16,source_checkpoint_directory=str(cp),persistent_mount_id='5a9955da-5240-41cc-9e52-05d3d8da8a3c',persistent_directory='case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/synthetic/epoch-001',run_manifest_sha256=sha(run/'run_manifest.json'),files=inventory)
        ackpath=run/'ack.json';ackpath.write_text(json.dumps(ack));namespace['verify_export_ack'](cp,meta,ackpath);cases.append('complete_ack_pass')
        changes={
            'missing_optimizer':lambda x:x['files'].pop('training_state.pt'),
            'changed_optimizer_sha':lambda x:x['files']['training_state.pt'].update(sha256='0'*64),
            'changed_optimizer_size':lambda x:x['files']['training_state.pt'].update(bytes=0),
            'wrong_source_path':lambda x:x.update(source_checkpoint_directory=str(run/'other')),
            'wrong_epoch':lambda x:x.update(completed_epochs=2),
            'wrong_global_step':lambda x:x.update(global_step=32),
            'wrong_mount':lambda x:x.update(persistent_mount_id='other'),
            'wrong_manifest':lambda x:x.update(run_manifest_sha256='0'*64),
            'missing_run_manifest':lambda x:x['files'].pop('_run/run_manifest.json'),
            'path_escape':lambda x:x['files'].update({'../escape':dict(bytes=1,sha256='0'*64)}),
        }
        for name,change in changes.items():
            bad=copy.deepcopy(ack);change(bad);ackpath.write_text(json.dumps(bad))
            try:namespace['verify_export_ack'](cp,meta,ackpath)
            except ValueError:cases.append(name+'_rejected')
            else:raise AssertionError(name+' accepted')
    fixture=load(B/'export-cpu-contract.json')
    for name,digest in fixture['source_sha256'].items():assert sha(B/name)==digest
    assert all(fixture['checks'].values()) and len(fixture['checks'])==8
    result=dict(status='PASS_INCREMENTAL_TRAINER_AND_ACK_SOURCE_NOT_COMPLETE_RUNTIME_ENTRY',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reviewer_source=ref(__file__),references={p.name:ref(p) for p in [B/'trainer-source-manifest.json',B/'train_order_warmstart.py',B/'cpu-source-contract.json',B/'export_checkpoint_server.py',B/'fetch_checkpoint.py',B/'test_export_contract.py',B/'export-cpu-contract.json']},
        unchanged_old_function_AST=identical,unchanged_old_microbatch_loss_and_own_resume_AST=True,
        old_optimizer_never_loaded_at_initial_warm_start=True,
        fresh_initial_Adam_scheduler_and_seed_asserted_in_source=True,
        same_old_adapter_SHA_and_FP32_LoRA_parameter_counts_checked_at_runtime=True,
        source_stage_transition_negative_cases=rejects,actual_reviewer_ACK_cases=cases,
        owner_transport_fixture_source_SHA_verified=True,
        owner_transport_fixture=dict(checks=fixture['checks'],seconds=fixture['seconds']),
        transport_review='Allowlisted immutable complete-epoch paths, pinned manifest, exact size/SHA and Range recovery, file fsync before exclusive final ACK, immediate failure on unexpected short source. Owner CPU fixture inspected, not rerun.',
        limitations=['Controller/endpoint and global deadline are a separate pending whole-entry check.','Actual remote CPU token/meta coverage and actual training/export are pending.','Same start seed does not imply identical later dropout masks when token shapes differ.','Fresh optimizer stage after old epoch2 is not exact old-stage continuation or a fresh4epoch run.','Current invocation elapsed must be aggregated by controller; old epoch1 export ACK is required before own-stage epoch2.'],
        no_model_or_tokenizer_import=True,no_network=True,no_gold_read=True)
    (OUT/'trainer-and-ack-source.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],report=ref(OUT/'trainer-and-ack-source.json'))))

if __name__=='__main__':main()
