"""Bind the already completed CPU contract into one fixed fresh GPU probe plan."""
import ast
import gzip
import hashlib
import json
from pathlib import Path

S=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-full64-v1')
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
ROOT=PROJECT+'/grounded-generation-full64-v1/probe-v1'
LEGACY=Path('/workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-training-readiness/stress-probe-v1')


def ref(p):
    p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    c=B/'native-cpu-v1';collection=json.loads((c/'collection.json').read_text());assert collection['CPU_reported_success'] and collection['all_actual_SHA_verified']
    for item in collection['files'].values():assert ref(item['path'])==item
    result=c/'terminal/actual/result.json';cpu=json.loads(result.read_text())
    assert cpu['status']=='PASS_NATIVE_FULL_INPUT_AND_INDEXED_POSITION_CPU' and cpu['train_rows']==173 and cpu['held_rows']==36
    actual=json.loads(gzip.decompress((c/'terminal/actual/encoded.json.gz').read_bytes()))
    local=json.loads(gzip.decompress((B.parent/'grounded-generation-full64-readiness-v1/token-census-v1/encoded.json.gz').read_bytes()))
    assert actual==local,'Exact original full native arrays must match, no regenerated corpus'
    data=B/'probe-preparation-v1';prep=json.loads((data/'manifest.json').read_text());rows=json.loads(gzip.decompress((data/'probe-encoded.json.gz').read_bytes()))['rows']
    assert len(rows)==13 and [r['original_index'] for r in rows]==prep['indices']
    for row in rows:
        assert {k:v for k,v in row.items() if k!='original_index'}==actual['train'][row['original_index']]
    payload={name:ref(LEGACY/name) for name in ['slot_model.py','qwen35_runner.py','qwen35_relation_cost.py','qwen35_smoke.py','selected_loss.py']}
    payload.update({name:ref(S/name) for name in ['probe.py','probe_driver.py']})
    payload['indexed_head.py']=ref(S.parent/'runtime/indexed_head.py')
    payload.update({'model-source-manifest.json':ref('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/model-source-manifest.json'),'probe-encoded.json.gz':ref(data/'probe-encoded.json.gz'),'CPU-result.json':ref(result),'CPU-collection.json':ref(c/'collection.json')})
    identity=lambda item:{k:item[k] for k in ['bytes','sha256']}
    cpu_plan=json.loads((S/'cpu-plan.json').read_text());env=cpu_plan['environment'];env['CUDA_VISIBLE_DEVICES']='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
    tree=ast.parse((S.parent/'runtime/indexed_head.py').read_text());tol=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOLERANCES' for t in n.targets))
    plan=dict(schema='grounded-full64-indexed-GPU-probe-v1',phase='engineering',GPU_engineering_authorized=True,diagnostic_rows=13,
       budget_seconds=420,grace_seconds=30,optimizer_steps=1,learning_rate=1e-4,weight_decay=.01,clip_norm=1.,seed=20261023,fresh_state_discarded=True,
       python=cpu_plan['python'],storage_parent=PROJECT,model=cpu_plan['model'],gpu_uuid=env['CUDA_VISIBLE_DEVICES'],environment=env,
       output_dir=ROOT+'/probe-output',payload_sources=payload,remote_files={n:identity(v) for n,v in payload.items()},files={ROOT+'/'+n:identity(v) for n,v in payload.items()},
       operations_source=ref(S/'probe_remote.py'),collection_source=ref(S/'probe_collect.py'),binding_source=ref(__file__),
       modeling_source_sha256='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3',native_loss_sha256='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe',
       frontier=prep['frontier'],prior_indexed_review=cpu_plan['old_indexed_review'],diagnostic_indices=prep['indices'],diagnostic_tokens=prep['tokens'],supervised_tokens=prep['supervised_tokens'],probe_data=ref(data/'manifest.json'),
       head_contract_tolerances=tol,actual_native_CPU=ref(result),actual_native_collection=ref(c/'collection.json'),
       original_recipe=dict(seed=20261023,rank=16,alpha=32,dropout=.05,learning_rate=1e-4,weight_decay=.01,clip_norm=1.,gradient_accumulation=8,epochs=2,examples=173,optimizer_steps=44,warmup_steps=4,bf16=True,sdpa=True,checkpointing_use_reentrant=False),
       input_sources=dict(preparation=cpu_plan['local_actual_targets'],native=ref(result),probe=ref(data/'manifest.json')),
       reuse_packages=[ref(c/'package.json')],
       scope='Fresh unchanged standard indexed CE; reuse prior mathematical contracts,13 backwards/one nonzero Adam, first8+three accumulated+two Pareto. All state discarded. No checkpoint/inference/quality.',
       limitations='Actual A800 Torch2.7.1 same native indexed source. Prior loss/gradient proof reused; newlong shapes actual pressure required. No GPU full-backbone bitwise equivalence claim.',
       budget_basis='420+30 parent-authorized;13 rows86278 fulltokens2726 supervised;10637/557 and6346/646 Pareto with existingAdam+grad. No truncation/allocator or precision workaround. Probe writes only smallmetadata under project512MiB gate; formalstate usesSHM4GiB andpersistent2GiB gates.',
       formal_training_authorized=False)
    p=S/'probe-plan.json';assert not p.exists();p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(p)))


if __name__=='__main__':main()
