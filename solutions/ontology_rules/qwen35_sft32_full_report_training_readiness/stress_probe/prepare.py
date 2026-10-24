from pathlib import Path
import ast,gzip,hashlib,importlib.util,io,json,random,shutil,sys,tarfile,time
R=Path('/workspace/patientphex-runs/round2-20261023');B=Path(__file__).parent;T=B.parent
W=Path('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/qwen35_sft32_full_report_training_readiness/stress_probe')
S=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_first_full64/memory_recovery')
F=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_first/frozen')
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(name,v):
 with (B/name).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def jsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
assert not (B/'plan.json').exists()
# Re-encode only the12 fixed diagnostic samples with the frozen native template;
# all IDs and label hashes must equal the already measured A800 native row audit.
p=json.loads((R/'research/patient-first-full64-cpu/plan.json').read_text());refs=p['references'];sys.path.insert(0,p['lightweight_libraries']);sys.path.insert(0,str(R/'research/patient-first-full64-cpu'))
spec=importlib.util.spec_from_file_location('native_frozen_helper',refs['parent_prepare']['path']);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
tok=u.FrozenTokenizer(refs['tokenizer']['path'],refs['template']['path']);enc=u.function(T/'qwen35_smoke.py','encode',{'hashlib':hashlib})
train=[json.loads(s) for s in (T/'expanded124.jsonl').read_text().splitlines()];audit=json.loads((T/'native-row-audit.json').read_text());native=json.loads((T/'native-result.json').read_text())
assert len(train)==124 and ref(T/'expanded124.jsonl')['sha256']=='9e80e54125d3d46479acb9a3720e8b5f4c13ffacb5f65e82d3d9d10ac839d25e'
assert native['expanded_encoded_sha256']=='0755d5fc1e21675e85b96dc09311a4587e9c7b93a0c0d728360f76c6b2d7a980'
ids=list(range(124));random.Random(20261023).shuffle(ids);position=ids.index(98);assert position==56
picked=ids[:8]+ids[position-3:position]+[98];assert len(set(picked))==12
rows=[]
for i in picked:
 row=enc(train[i],tok);a=audit[i]['expanded'];assert row['row_id']==audit[i]['row_id'];assert jsha(row['input_ids'])==a['input_ids_sha256'];assert jsha(row['labels'])==a['labels_sha256'];assert len(row['input_ids'])==a['full_tokens']<=13824
 rows.append(dict(original_index=i,**row))
assert len(rows[-1]['input_ids'])==13549
payload=dict(policy='Original epoch0 first8; then original shuffle positions53..55 accumulated before longest position56. Deliberate cross-group pressure order, not formal order/warmup replay.',rows=rows)
blob=gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0);(B/'probe-encoded.json.gz').write_bytes(blob)
metadata=dict(policy=payload['policy'],rows=[dict(row_id=r['row_id'],original_index=r['original_index'],tokens=len(r['input_ids']),prefix_tokens=r['prefix_tokens'],supervised_tokens=r['supervised_tokens']) for r in rows],total_tokens=sum(len(r['input_ids']) for r in rows),encoded=ref(B/'probe-encoded.json.gz'),all12_match_previous_actual_native_audit=True,original124_native=ref(T/'native-result.json'),audit=ref(T/'native-row-audit.json'),source=ref(__file__),no_new_native_census=True,no_heldout_gold=True)
save('probe-data.json',metadata)
# Reuse H51 pressure implementation, change only actual dataset/runtime/bindings.
source=(S/'probe.py').read_text();source=source.replace('"run-plan.json"','"plan.json"').replace('plan["budget_seconds"] == 240','plan["budget_seconds"] == 420')
source=source.replace('cpu["status"] == "PASS_actual_Torch_CPU_fixed_full_logits_selected_CE_contract"','cpu["status"] == "PASS_ACTUAL_A800_TORCH271_CPU_SELECTED_CE_SYNTHETIC_CONTRACT_NO_MODEL"')
source=source.replace('assert rows[-1]["row_id"]=="3563609:relations:0" and len(rows[-1]["input_ids"])==6166','assert rows[-1]["original_index"]==98 and len(rows[-1]["input_ids"])==13549')
source=source.replace('assert rows[-1]["prefix_tokens"]==6044 and rows[-1]["supervised_tokens"]==122','assert rows[-1]["prefix_tokens"]==plan["longest"]["prefix_tokens"] and rows[-1]["supervised_tokens"]==plan["longest"]["supervised_tokens"]\n        assert [r["original_index"] for r in rows] == plan["diagnostic_indices"]\n        assert sum(len(r["input_ids"]) for r in rows)==plan["diagnostic_tokens"] and all(len(r["input_ids"])<=13824 for r in rows)\n        native=json.loads(Path("native-result.json").read_text());assert native["expanded_encoded_sha256"]==plan["expanded_encoded_sha256"] and native["expanded"]["supervised_tokens"]==7170')
source=source.replace('torch.__version__=="2.6.0+cu124"','torch.__version__=="2.7.1+cu126"')
source=source.replace('model, named, loaded = loader.load_fresh','torch.backends.cuda.matmul.allow_tf32=False\n        torch.backends.cudnn.allow_tf32=True\n        model, named, loaded = loader.load_fresh')
source=source.replace('native_bindings=bindings, FLA=False,','native_bindings=bindings, FLA=False, diagnostic_data=plan["probe_data"], original_training_configuration=plan["original_recipe"],\n            TF32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32),')
source=source.replace('for ordinal,row in enumerate(rows,1):','for ordinal,row in enumerate(rows,1):\n            row_tick=time.monotonic()')
source=source.replace('elapsed_update_window_seconds=time.monotonic()-tick))','elapsed_update_window_seconds=time.monotonic()-tick,row_forward_backward_seconds=time.monotonic()-row_tick))')
(B/'probe.py').write_text(source)
copies={'selected_loss.py':S/'selected_loss.py','slot_model.py':F/'slot_model.py','qwen35_runner.py':F/'qwen35_runner.py','qwen35_relation_cost.py':F/'qwen35_relation_cost.py','qwen35_smoke.py':T/'qwen35_smoke.py','model-source-manifest.json':R/'research/qwen35-sft32-full-report-feasibility/probe-v1/model-source-manifest.json','CPU-result.json':T/'selected-ce-cpu/contract-result.json','native-result.json':T/'native-result.json'}
for name,path in copies.items():shutil.copyfile(path,B/name)
launch=(R/'research/qwen35-sft32-full-report-feasibility/f0-preparation/full58-v1/launch.py').read_text().replace('full80 input control','fresh-base selected-CE training pressure probe').replace('1200','420').replace('infer_full58.py','probe.py');(B/'launch.py').write_text(launch)
root='/home/zhangyuheng/projects/CHIP2026_PatientPheX';uuid='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
files={name:{k:v for k,v in ref(B/name).items() if k!='path'} for name in ['probe.py','launch.py','probe-encoded.json.gz','probe-data.json',*copies]}
plan=dict(status='FROZEN_UNIQUE420_TRAINING_ENGINEERING_PROBE_NOT_FORMAL_OR_QUALITY',budget_seconds=420,grace_seconds=30,cap_seconds=420,term_grace_seconds=30,optimizer_steps=1,learning_rate=1e-4,weight_decay=.01,clip_norm=1.,seed=20261023,fresh_state_discarded=True,python=root+'/ontology-round2/qwen-venv/bin/python',model='/dev/shm/patientphex-qwen35-1045-c20223623576',gpu_uuid=uuid,mount_id='69b991ee-183f-4add-af1a-579af92479c9',remote='ontology-round2/qwen35-sft32-full-report-training-stress-v1',modeling_source_sha256='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3',native_loss_sha256='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe',CPU_contract_source_sha256='c11868cfa284e4b7fc756e3f2fc816429e08c50c48c13689f47dc4dc2f40fc86',files=files,remote_files=files,diagnostic_indices=picked,diagnostic_tokens=metadata['total_tokens'],longest=metadata['rows'][-1],input_rejection_cap=13824,expanded_encoded_sha256=native['expanded_encoded_sha256'],probe_data=ref(B/'probe-data.json'),native_CPU=ref(T/'native-result.json'),selected_CE_CPU=ref(T/'selected-ce-cpu/contract-result.json'),H51_source=ref(S/'probe.py'),original_trainer=ref(R/'case_retrieval/qwen35-preparation/sft-meta1-fusion/train_qwen35_meta32.py'),original_recipe=dict(seed=20261023,rank=16,alpha=32,dropout=.05,learning_rate=.0001,weight_decay=.01,clip_norm=1.,batch_size=1,gradient_accumulation=8,epochs=2,examples=124,optimizer_steps=32,warmup_steps=2,bf16=True,sdpa=True,checkpointing_use_reentrant=False,targets=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj']),training_sources=dict(expanded_train=ref(T/'expanded124.jsonl'),original_train=ref(T/'original124.jsonl'),preparation=ref(T/'preparation.json'),supervised_tokens=7170,train_documents=32,folds=[2,3]),sequence=payload['policy'],limitations='One nonzero LR1e-4 diagnostic optimizer step, formal warmup/scheduler not replayed; pressure order differs at group boundary. All state discarded. Full varying-shape cost/export and quality unknown. No retry/full training authorization.',no_checkpoint=True,no_quality=True,full_training_authorized=False,env=dict(CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=root+'/case-retrieval/qwen35-preparation/a800-migration/runtime',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
save('plan.json',plan)
print(json.dumps(dict(status='SOURCE_PREPARED',plan=ref(B/'plan.json'),source=ref(B/'probe.py'),rows=metadata['rows'],total_tokens=metadata['total_tokens'])))
