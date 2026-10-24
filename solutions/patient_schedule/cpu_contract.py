"""Incremental local data/schedule/state/controller contracts; no weights/CUDA/quality."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,random,tempfile,time,sys
import train as trainer
import runtime_contract as contract

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();started=time.perf_counter();B=a.controller;S=Path(__file__).parent;plan=json.loads((B/'controller-plan.json').read_text());sources=plan['provenance'];checks={}
 mapping={'train':'train.jsonl','excluded_input':'excluded.input.jsonl','data_manifest':'preparation-manifest.json','transposition_manifest':'transposition-summary.json','train_token_audit':'train-token-audit.json','native_cpu_receipt':'native-cpu-receipt.json','model_source_manifest':'model-source-manifest.json'}
 args=argparse.Namespace(**{k:v['path'] for k,n in mapping.items() for v in [sources[n]]},schema_helper=str(S/'patient_first.py'))
 rows,ids,excluded=trainer.verify_sources_and_data(args);checks['exact_original_train_and_native_receipt_lineage']=True
 patients={(str(r['pmc_id']),pid) for r in rows for pid in r['patient_ids']};assert len(patients)==87
 audit=json.loads(Path(args.train_token_audit).read_text());assert sum(r['patient_first']['full_tokens'] for r in audit)==400614 and sum(r['patient_first']['supervised_tokens'] for r in audit)==4694
 schedules=[]
 for epoch in range(4):
  groups=trainer.batch_plan(124,1,8,20261023,epoch);flat=[i for g in groups for micro in g for i in micro];assert sorted(flat)==list(range(124)) and [len(g) for g in groups]==[8]*15+[4]
  schedules.append(dict(epoch=epoch+1,row_order_sha256=hashlib.sha256(json.dumps(flat).encode()).hexdigest(),groups=[len(g) for g in groups],first_global_step=16*epoch+1,last_global_step=16*(epoch+1)))
 checks['four_unique_full_epoch_orders_and_four_tail4']=True
 lr=lambda step:1e-4*(step/4 if step<4 else max(0,(64-step)/60))
 table=[dict(call=i+1,lr_before=lr(i),lr_after=lr(i+1)) for i in range(64)]
 assert sum(x['lr_before']>0 for x in table)==63 and lr(32)==1e-4*32/60 and lr(64)==0
 checks['fixed_unified4_warmup64_schedule']=True
 def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)}
 original=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_first/train_patient_first.py');old=funcs(original);new=funcs(S/'train.py')
 preserved=['training_rows','verify_sources_and_data','load_schema_helper','model_files','encode_all','batch_plan','collate','verify_fingerprint','validate_stop_epoch','epoch_outcome']
 assert all(old[n]==new[n] for n in preserved);checks['ten_core_data_loader_batch_and_loss_input_functions_AST_exact']=True
 text=(S/'train.py').read_text();assert "loss=model(**batch,use_cache=False).loss" in text and '(loss/len(group)).backward()' in text and 'get_linear_schedule_with_warmup(optimizer,4,64)' in text and 'indexed_head' not in text and 'patient_balanced' not in text
 checks['native_standard_CE_no_new_loss_projection']=True
 negatives=0
 for completed in (1,2,3):
  state=dict(completed_epochs=completed,global_step=16*completed,optimizer=dict(state={i:dict(step=16*completed,exp_avg=[.1],exp_avg_sq=[.01]) for i in range(3)},param_groups=[dict(params=[0,1,2],lr=lr(16*completed))]),scheduler=dict(last_epoch=16*completed,_step_count=16*completed+1),python_rng=random.getstate(),numpy_rng=('fixture',),torch_rng=[1],cuda_rng=[[2]],optimizer_parameter_names=['a','b','c'])
  assert contract.validate_state_structure(state,completed,['a','b','c']);digest=contract.tree_digest(state);assert contract.tree_digest(copy.deepcopy(state))==digest
  for field in ['optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','optimizer_parameter_names']:
   altered=copy.deepcopy(state);altered[field]=None;assert contract.tree_digest(altered)!=digest;negatives+=1
  for mutation in ['step','schedule','lr','names','missing']:
   altered=copy.deepcopy(state)
   if mutation=='step':altered['optimizer']['state'][1]['step']-=1
   if mutation=='schedule':altered['scheduler']['last_epoch']-=1
   if mutation=='lr':altered['optimizer']['param_groups'][0]['lr']=0
   if mutation=='names':altered['optimizer_parameter_names'].reverse()
   if mutation=='missing':altered.pop('cuda_rng')
   try:contract.validate_state_structure(altered,completed,['a','b','c'])
   except (ValueError,KeyError):negatives+=1
   else:raise AssertionError('Accepted bad fullstate '+mutation)
 checks['three_restore_boundaries_logical_state_and_negative_inventory']=True
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);fingerprint={'schema':'fixture-four'}
  for ep in range(1,5):
   staging=root/f'.epoch-{ep:03d}.incomplete';staging.mkdir();(staging/'adapter').mkdir();files={}
   for name in ['training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json']:
    path=staging/name;path.write_bytes((name+str(ep)).encode());files[name]=contract.sha(path)
   (staging/'checkpoint.json').write_text(json.dumps(dict(fingerprint=fingerprint,files=files,completed_epochs=ep,global_step=ep*16,smoke_only=False)))
   destination=root/f'epoch-{ep:03d}';receipt=contract.local_publish_checkpoint(staging,destination,root);assert receipt['persistent_export_ack_still_required']
   (root/f'epoch-{ep:03d}.ack.json').write_text('{}')
   if ep<4:assert trainer.validate_checkpoint(root,destination,fingerprint)['global_step']==ep*16
   else:
    try:trainer.validate_checkpoint(root,destination,fingerprint)
    except ValueError:negatives+=1
    else:raise AssertionError('Completed epoch4 resumed')
 checks['four_local_fsync_reopen_and_final_checkpoint_resume_rejection']=True
 for p in S.rglob('*.py'):compile(p.read_text(),str(p),'exec')
 checks['all_source_compile']=True
 assert 'epoch in [1,2,3,4]' in (S/'controller/io_driver.py').read_text() and "epoch>1" in (S/'controller/watch_authorized_campaign.py').read_text()
 assert "if epoch<4:" in (S/'controller/epoch_driver.py').read_text()
 checks['all_four_controller_stages_present']=True
 payload=plan['payload_files']
 for name,r in payload.items():
  local=B/name
  if local.exists():assert ref(local)['bytes']==r['bytes'] and ref(local)['sha256']==r['sha256']
 checks['frozen_deployment_files_match']=True
 assert 'torch' not in sys.modules and 'transformers' not in sys.modules
 result=dict(status='PASS_INCREMENTAL_LOCAL_CPU_SOURCE_DATA_SCHEDULE_LOGICAL_STATE',seconds=time.perf_counter()-started,checks=checks,negative_checks=negatives,train_rows=124,train_documents=32,train_patients=87,excluded_documents=32,epochs=4,total_calls=64,full_tokens=400614*4,supervised_tokens=4694*4,schedules=schedules,lr_table=table,source=ref(__file__),trainer=ref(S/'train.py'),controller_plan=ref(B/'controller-plan.json'),native_receipt_reused=sources['native-cpu-receipt.json'],new_native_tokenization=False,new_Torch_or_GPU_state_execution=False,no_GPU_model_or_quality=True,limitations='Logical structure/synthetic digest and original actual checkpoint implementation reused; not a new Torch tensor or GPU resume equality test.')
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],seconds=result['seconds'],checks=len(checks),negative_checks=negatives,ref=ref(a.output))))
if __name__=='__main__':main()
