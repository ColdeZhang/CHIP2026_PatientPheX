"""Freeze the one authorized warm261 campaign after actual CPU and pressure PASS.

Reuse existing verified remote data and modules. Package only the common trainer,
small plans, and the proven single-epoch instance of the verified persistence controller. No model execution.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/quote-set-completion-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL='quote-set-completion-v2/formal-v1'
REMOTE=P+'/'+REL
SHM='/dev/shm/patientphex-quote-set-completion-v2-formal'
sys.path.insert(0,str(S))
import configuration

def ref(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def identity(v):return {k:v[k] for k in ('bytes','sha256')}
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=D/'formal-v1');a=ap.parse_args()
    out=a.directory.resolve();assert not out.exists(),'Keep an existing preparation immutable'
    cpu_dir=D/'native-cpu-v2';probe_dir=D/'probe-v2'
    cpu_collection=read(cpu_dir/'collection.json');probe_collection=read(probe_dir/'collection.json')
    assert cpu_collection['CPU_reported_success'] and probe_collection['probe_reported_success']
    for collection in [cpu_collection,probe_collection]:
        assert collection['all_actual_SHA_verified']
        for item in collection['files'].values():assert ref(item['path'])==item
    cpu_package=read(cpu_dir/'package.json');probe_package=read(probe_dir/'package.json')
    cpu_plan=read(S.parent/'runtime_cpu/cpu-plan.json')
    probe_plan=read(S.parent/'engineering/probe-plan.json')
    cpu=read(cpu_dir/'terminal/actual/result.json');step=read(probe_dir/'terminal/step.json')
    complete=read(probe_dir/'terminal/complete.json')
    assert complete['actual_backwards']==13 and complete['actual_steps']==1 and complete['fresh_state_discarded']
    assert step['all_two_Pareto_completed'] and step['covered_frontier']==probe_plan['frontier']
    assert probe_collection['wrapper']['target_has_no_process']
    # Measured token-rate estimate is a budget proxy, not a runtime guarantee.
    rows=[read(probe_dir/('terminal/stage-%02d-after.json'%i)) for i in range(1,14)]
    actual_bw_seconds=sum(x['row_forward_backward_seconds'] for x in rows)
    token_proxy=actual_bw_seconds/(probe_plan['diagnostic_tokens'])*1380333
    first_update=read(probe_dir/'terminal/first-update.json')
    conservative=token_proxy*1.25+300
    assert conservative<3600,'Actual pressure cost does not fit authorized conservative full run; report before training'
    cfg=configuration.validate(configuration.EXPECTED,'quote-set-completion-v2')
    local={};remote={}
    def add(name,local_path,remote_path):
        item=ref(local_path);local[name]=item;remote[name]={**item,'path':remote_path}
    cpu_root=cpu_package['remote'];probe_root=probe_package['remote']
    for name,filename in [('train','train.jsonl'),('train_audit','train-audit.json'),('cpu_plan','cpu-plan.json'),('native_cpu_source','native_cpu.py')]:
        item=cpu_package['members'][filename];assert ref(item['path'])==item
        add(name,item['path'],cpu_root+'/'+filename)
    for name,filename in [('cpu_result','result.json'),('cpu_runtime','runtime.json'),('encoded','encoded.json.gz'),('state_cpu_result','state-interface.json')]:
        add(name,cpu_dir/'terminal/actual'/filename,cpu_root+'/actual/'+filename)
    for name,filename in [('probe_complete','complete.json'),('probe_step','step.json'),('probe_manifest','run-manifest.json')]:
        add(name,probe_dir/'terminal'/filename,probe_root+'/'+filename)
    for name in ['indexed_head','warm_model','qwen35_runner','qwen35_relation_cost','qwen35_smoke']:
        item=probe_package['members'][name+'.py'];assert ref(item['path'])==item
        add(name,item['path'],probe_root+'/'+name+'.py')
    payload={n:ref(S/n) for n in ['train.py','configuration.py','runtime_contract.py']}
    payload.update({n:ref(S/'controller'/n) for n in ['epoch_driver.py','io_driver.py','export_checkpoint_server.py','fetch_checkpoint.py']})
    payload['model-source-manifest.json']=probe_package['members']['model-source-manifest.json']
    for name,filename in [('trainer','train.py'),('configuration','configuration.py'),('runtime_contract','runtime_contract.py'),('model_manifest','model-source-manifest.json')]:
        add(name,payload[filename]['path'],REMOTE+'/'+filename)
    assert identity(local['runtime_contract'])==identity(cpu_package['members']['runtime_contract.py'])
    env={**cpu_plan['environment'],'CUDA_VISIBLE_DEVICES':probe_plan['gpu_uuid'],'PATIENTPHEX_DURABILITY_ROOT':SHM}
    formal=dict(schema='quote-set-completion-warm-oneepoch-v2',formal_training_authorized=True,route='quote-set-completion-v2',configuration=cfg,
        budget_seconds=3600,grace_seconds=30,output_dir=SHM+'/output',storage_parent=SHM,
        persistent_checkpoint_root=P+'/research/quote-set-completion-v2/checkpoints',persistent_mount_id='5a9955da-5240-41cc-9e52-05d3d8da8a3c',
        minimum_SHM_free_bytes=4*1024**3,minimum_persistent_free_bytes=2*1024**3,
        model=cpu_plan['model'],gpu_uuid=probe_plan['gpu_uuid'],assets=remote,warm_adapter=probe_plan['warm_adapter'],warm_model_sha256=local['warm_model']['sha256'],
        probe_contract=dict(backwards=13,status=step['status'],frontier=step['covered_frontier']),
        actual_CPU_collection=ref(cpu_dir/'collection.json'),actual_probe_collection=ref(probe_dir/'collection.json'),
        budget_basis=dict(actual13_backward_seconds=actual_bw_seconds,actual13_tokens=probe_plan['diagnostic_tokens'],
            complete_oneepoch_tokens=1380333,token_rate_proxy_seconds=token_proxy,
            conservative_proxy_seconds=conservative,conservative_formula='1.25 * measured_backward_token_proxy + 300s loading/33updates/save/export/control allowance; within shared3600 with120ACK reserve',
            actual_runtime_not_guaranteed=True,original_absolute_cap_seconds=3600,TERM_grace_seconds=30),
        state_policy='Fresh public base with original G32d52 warm weights only; freshAdam/scheduler/RNG, no probe state. One261 epoch/33calls then fullSHM state fsync/reopen and .100 durableACK before inference. No resume/epoch selection.',
        quality_policy='After actual final complete-state persistentACK: fixed39 fullF1 completion, known originalG32accepted quotes, greedy1600 under separate1800+30. Same train32aliases/public41099/b59 necessarySap180+30. Only B4union additionalrelations completefreeze then1v2/old0; positive strictly toF0, no originaldev.')
    # Run the real trainer's metadata verifier against the identical local bytes.
    spec=importlib.util.spec_from_file_location('grounded_shared_formal_preflight',S/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    encoded,audit=module.verify_inputs({**formal,'assets':local})
    assert len(encoded)==261 and len(audit)==261
    out.mkdir(parents=True)
    for name,item in payload.items():
        assert ref(item['path'])==item;shutil.copyfile(item['path'],out/name)
    save(out/'formal-plan.json',formal);payload['formal-plan.json']=ref(out/'formal-plan.json')
    run_files=['run_manifest.json','train.py','configuration.py','runtime_contract.py','warm_model.py','indexed_head.py','qwen35_smoke.py','qwen35_relation_cost.py','progress.json']
    controller=dict(schema='quote-set-completion-oneepoch-persistent-controller-v2',formal_training_authorized=True,
        mount_id='69b991ee-183f-4add-af1a-579af92479c9',persistent_mount_id=formal['persistent_mount_id'],
        remote_relative=REL,remote_absolute=REMOTE,persistent_remote_relative=REL,persistent_remote_absolute=REMOTE,
        persistent_checkpoint_root=formal['persistent_checkpoint_root'],storage_parent=SHM,output_dir=formal['output_dir'],
        python=cpu_plan['python'],persistent_python='/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',source_private_IP='10.5.4.7',
        model=formal['model'],gpu_uuid=formal['gpu_uuid'],configuration=cfg,cap_seconds=3600,term_grace_seconds=30,persistent_ack_reserve_seconds=120,
        export_seconds_each=600,trainer_sha256=local['trainer']['sha256'],expanded_train_sha256=local['train']['sha256'],
        fingerprint_schema='quote-set-completion-warm-oneepoch-indexed-CE-v2',export_trainer_basename='train.py',export_run_files=run_files,
        source_training_plan=ref(out/'formal-plan.json'),CPU_contract=local['cpu_result'],
        train_argv=[cpu_plan['python'],'-u',REMOTE+'/train.py','--plan',REMOTE+'/formal-plan.json'],
        payload_files={n:identity(v) for n,v in payload.items()},decompress={},env=env,
        no_probe_state_reuse=True,no_inference_or_quality_here=True,resume_policy=formal['state_policy'],
        local_operations=ref(S/'controller/operations.py'),local_watcher=ref(S/'controller/watch_authorized_campaign.py'),
        binding_source=ref(__file__),actual_CPU_collection=formal['actual_CPU_collection'],actual_probe_collection=formal['actual_probe_collection'])
    save(out/'controller-plan.json',controller)
    members={n:identity(v) for n,v in payload.items()};members['controller-plan.json']=identity(ref(out/'controller-plan.json'))
    with tarfile.open(out/'deployment.tar.gz','w:gz') as archive:
        for name in sorted(members):archive.add(out/name,arcname=name,recursive=False)
    save(out/'deployment-manifest.json',dict(archive=ref(out/'deployment.tar.gz'),files=members,controller_plan=ref(out/'controller-plan.json'),raw_inputs_verified_after_decompression={}))
    save(out/'preparation-complete.json',dict(status='PASS_REAL_TRAINER_METADATA_VERIFY_NATIVE261_WITHOUT_GPU_OR_FIT',formal_plan=ref(out/'formal-plan.json'),
        controller_plan=ref(out/'controller-plan.json'),deployment=ref(out/'deployment-manifest.json'),local_actual_assets=local,
        data_retransmitted=False,no_GPU=True,no_quality=True,probe_state_discarded=True,budget_basis=formal['budget_basis']))
    print(json.dumps(dict(plan=ref(out/'controller-plan.json'),preparation=ref(out/'preparation-complete.json'))))

if __name__=='__main__':main()
