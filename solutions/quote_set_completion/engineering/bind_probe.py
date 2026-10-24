"""Bind one completed CPU contract to warm-d52 full-state engineering pressure."""
from pathlib import Path
import ast,gzip,hashlib,json,random
S=Path(__file__).resolve().parent
Q=S.parent
N=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1')
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
ROOT=P+'/quote-set-completion-v2/probe-v1'
def ref(p):
 p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(p,value):
 assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
 c=N/'native-cpu-v2';collection=json.loads((c/'collection.json').read_text())
 assert collection['CPU_reported_success'] and collection['all_actual_SHA_verified']
 for item in collection['files'].values():assert ref(item['path'])==item
 cpu_path=c/'terminal/actual/result.json';cpu=json.loads(cpu_path.read_text())
 assert cpu['train_rows']==261 and cpu['held_rows']==0 and cpu['all_local_native_actual_IDs_labels_rendering_exact']
 actual=json.loads(gzip.decompress((c/'terminal/actual/encoded.json.gz').read_bytes()))
 local=json.loads(gzip.decompress((N/'train-only-cpu-v2-recovery/encoded.json.gz').read_bytes()))
 assert actual==local
 rows=actual['train'];audit=json.loads((N/'runtime-preparation-v2/train-audit.json').read_text())
 frontier=[max(audit,key=lambda r:r['full_tokens']),max(audit,key=lambda r:r['supervised_tokens'])]
 frontier=[dict(index=r['index'],tokens=r['full_tokens'],supervised=r['supervised_tokens']) for r in frontier]
 assert frontier==[dict(index=132,tokens=9307,supervised=187),dict(index=249,tokens=5773,supervised=307)]
 indices=list(range(261));random.Random(20261023).shuffle(indices)
 indices=[i for i in indices if i not in [f['index'] for f in frontier]][:11]+[f['index'] for f in frontier]
 selected=[dict(rows[i],original_index=i) for i in indices];assert len(selected)==13
 data=N/'probe-preparation-v2';data.mkdir(exist_ok=False)
 (data/'probe-encoded.json.gz').write_bytes(gzip.compress(json.dumps(dict(rows=selected),ensure_ascii=False).encode(),mtime=0))
 save(data/'manifest.json',dict(indices=indices,frontier=frontier,tokens=sum(len(r['input_ids']) for r in selected),supervised_tokens=sum(r['supervised_tokens'] for r in selected),encoded_source=ref(c/'terminal/actual/encoded.json.gz'),encoded=ref(data/'probe-encoded.json.gz'),selection='fixed shuffled first11 excluding the two extremes; group8 update then3 accumulation thenlongest/maxsupervised'))
 prep=json.loads((data/'manifest.json').read_text())
 plan=json.loads((G/'full64_runtime/probe-plan.json').read_text())
 payload={n:i for n,i in plan['payload_sources'].items() if n in ['qwen35_runner.py','qwen35_relation_cost.py','qwen35_smoke.py','selected_loss.py','indexed_head.py','model-source-manifest.json']}
 payload.update({n:ref(S/n) for n in ['probe.py','probe_driver.py']})
 payload.update({'warm_model.py':ref(Q/'training/warm_model.py'),'runtime_contract.py':ref(Q/'runtime_cpu/runtime_contract.py'),'probe-encoded.json.gz':ref(data/'probe-encoded.json.gz'),'CPU-result.json':ref(cpu_path),'CPU-collection.json':ref(c/'collection.json')})
 identity=lambda v:{k:v[k] for k in ['bytes','sha256']}
 warm=dict(directory=P+'/grounded-generation-v1/F0-inference-cache/epoch-002',files={
 'adapter_model.safetensors':dict(bytes=173181072,sha256='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba'),
 'adapter_config.json':dict(bytes=1341,sha256='a9e753c405c1a0b0f3811d54224437455629c9cd96d647cfe5f6cf8222dfd307')})
 plan.update(schema='quote-set-completion-warm-d52-indexed-probe-v2',budget_seconds=240,learning_rate=1e-5,warm_adapter=warm,
  output_dir=ROOT+'/probe-output',probe_state_directory='/dev/shm/patientphex-quote-set-completion-probe-v1/state-proof',
  payload_sources=payload,remote_files={n:identity(v) for n,v in payload.items()},files={ROOT+'/'+n:identity(v) for n,v in payload.items()},
  operations_source=ref(S/'probe_remote.py'),collection_source=ref(S/'probe_collect.py'),binding_source=ref(__file__),
  frontier=frontier,diagnostic_indices=indices,diagnostic_tokens=prep['tokens'],supervised_tokens=prep['supervised_tokens'],probe_data=ref(data/'manifest.json'),
  actual_native_CPU=ref(cpu_path),actual_native_collection=ref(c/'collection.json'),reuse_packages=[],
  original_recipe=dict(seed=20261023,rank=16,alpha=32,dropout=.05,learning_rate=1e-5,weight_decay=.01,clip_norm=1.,gradient_accumulation=8,epochs=1,examples=261,optimizer_steps=33,warmup_steps=2,tail_group_size=5,bf16=True,sdpa=True,checkpointing_use_reentrant=False),
  input_sources=dict(preparation=ref(N/'train-only-cpu-v2-recovery/manifest.json'),native=ref(cpu_path),probe=ref(data/'manifest.json')),
  scope='Fixed warmG32d52 weights; freshAdam/RNG;13backwards/one nonzeroAdam;8+3+longest/maxsupervised. Full adapter/Adam/scheduler/allRNG state is serialized and reopened exactly on privateSHM, then deleted; never used for formal.',
  budget_basis='240+30 rootauthorized, no held/probe quality; new261 inputcomplete actualnative. Formal remaining budget assessed after measuredcost. Project512MiB metadata gate, scratchSHM4GiB state-save gate; no persistent checkpoint claimed.',
  limitations='Original indexedloss/core reused. Warm nonzeroLoRA is explicit; diagnostic constantLR differs from formal warmup2. No model update from this diagnostic may initialize formal.')
 save(S/'probe-plan.json',plan);print(json.dumps(ref(S/'probe-plan.json')))
if __name__=='__main__':main()
