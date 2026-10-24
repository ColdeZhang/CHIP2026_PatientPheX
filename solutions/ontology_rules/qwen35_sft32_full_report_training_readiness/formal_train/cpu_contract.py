"""Stdlib-only new epoch boundary contracts. Never instantiate Torch/model/tokenizer."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,os,random,signal,sys,tempfile,time
sys.dont_write_bytecode=True
B=Path(__file__).parent
sys.path.insert(0,str(B))
import runtime_contract as contract

def ref(path):
 p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def extracted(path,name,namespace):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace[name]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.plan.read_text());start=time.monotonic();signal.alarm(60);checks=[]
 def okay(name,condition=True):
  assert condition,name;checks.append({'name':name,'pass':True})
 def rejected(name,call):
  try:call()
  except (AssertionError,ValueError,TimeoutError,FileNotFoundError,KeyError):okay(name);return
  raise AssertionError('Invalid fixture accepted: '+name)
 for record in plan['source_files'].values():assert ref(record['path'])==record
 old=plan['old_trainer'];assert ref(old['path'])==old
 ns={'random':random};old_batch=extracted(old['path'],'batch_plan',dict(ns));new_batch=extracted(B/'train.py','batch_plan',dict(ns))
 orders=[]
 for epoch in range(2):
  a=old_batch(124,1,8,20261023,epoch);b=new_batch(124,1,8,20261023,epoch)
  okay('epoch%d exact old order and full16 groups'%epoch,a==b and len(a)==16 and [len(x) for x in a]==[8]*15+[4])
  flat=[i for group in a for batch in group for i in batch];okay('epoch%d complete124 once'%epoch,sorted(flat)==list(range(124)));orders.append(flat)
 okay('two epochs keep distinct shuffle',orders[0]!=orders[1])
 # No tensors are created. This fixture checks state presence/counters/order;
 # runtime validates actual Tensor values/device-agnostic digests after load.
 names=['A','B'];state={'completed_epochs':1,'global_step':16,'optimizer':{'state':{i:{'step':16,'exp_avg':[.1+i],'exp_avg_sq':[.2+i]} for i in range(2)},'param_groups':[{'params':[0,1],'lr':1e-4*(32-16)/(32-2)}]},'scheduler':{'last_epoch':16,'_step_count':17},'python_rng':[1,2],'numpy_rng':['rng',[3,4]],'torch_rng':[5,6],'cuda_rng':[[7,8]],'optimizer_parameter_names':names}
 okay('complete first epoch structural state accepted',contract.validate_state_structure(state,1,names))
 for key in contract.STATE_KEYS:
  bad=copy.deepcopy(state);del bad[key];rejected('missing '+key+' rejected',lambda bad=bad:contract.validate_state_structure(bad,1,names))
 for field,val in [('completed_epochs',0),('global_step',0),('optimizer_parameter_names',list(reversed(names)))]:
  bad=copy.deepcopy(state);bad[field]=val;rejected('wrong '+field+' rejected',lambda bad=bad:contract.validate_state_structure(bad,1,names))
 for field,val in [('last_epoch',0),('_step_count',1)]:
  bad=copy.deepcopy(state);bad['scheduler'][field]=val;rejected('reset scheduler '+field+' rejected',lambda bad=bad:contract.validate_state_structure(bad,1,names))
 bad=copy.deepcopy(state);del bad['optimizer']['state'][1];rejected('partial Adam inventory rejected',lambda:contract.validate_state_structure(bad,1,names))
 bad=copy.deepcopy(state);del bad['optimizer']['state'][0]['exp_avg_sq'];rejected('missing Adam second moment rejected',lambda:contract.validate_state_structure(bad,1,names))
 for key in state:
  changed=copy.deepcopy(state);changed[key]=['altered'];okay('logical digest detects '+key,contract.tree_digest(changed)!=contract.tree_digest(state))
 okay('logical digest dictionary order invariant',contract.tree_digest(dict(reversed(list(state.items()))))==contract.tree_digest(state))
 # Exact per-row identity binding is checked from already measured receipts;
 # no repeated tokenization, native numerics, model, or gold access.
 native=json.loads(Path(plan['native_receipt']['path']).read_text());audit=json.loads(Path(plan['native_audit']['path']).read_text());prep=json.loads(Path(plan['fulltext_preparation']['path']).read_text())
 for record in [plan['native_receipt'],plan['native_audit'],plan['fulltext_preparation']]:okay('receipt SHA '+Path(record['path']).name,ref(record['path'])==record)
 okay('124 fulltext rows /7170 exact supervision binding',len(audit)==124 and sum(r['expanded']['supervised_tokens'] for r in audit)==7170 and all(r['assistant_supervised_IDs_exact'] and r['native_Auto_template_and_decode_exact'] for r in audit))
 okay('actual fulltext token identity',sum(r['expanded']['full_tokens'] for r in audit)==1009181 and max(r['expanded']['full_tokens'] for r in audit)==13549 and native['expanded_encoded_sha256']==plan['encoded_sha256'])
 okay('32train and32heldout no overlap',len(prep['training_document_ids'])==len(prep['excluded_document_ids_metadata_only'])==32 and not set(prep['training_document_ids'])&set(prep['excluded_document_ids_metadata_only']))
 okay('contract native shadow not rerun','torch' not in sys.modules and 'transformers' not in sys.modules)
 # H44 regression and durable ACK binding use only synthetic tiny files.
 with tempfile.TemporaryDirectory(prefix='fulltext-source-contract-') as temporary:
  output=Path(temporary);cp=output/'epoch-001';(cp/'adapter').mkdir(parents=True)
  for name,data in [('training_state.pt',b'synthetic-state'),('adapter/adapter_model.safetensors',b'synthetic-adapter'),('adapter/adapter_config.json',b'{}')]:
   p=cp/name;p.write_bytes(data)
  meta={'files':{str(p.relative_to(cp)):contract.sha(p) for p in cp.rglob('*') if p.is_file()}}
  (cp/'checkpoint.json').write_text(json.dumps(meta));(output/'run_manifest.json').write_text('{}');(output/'epoch-001-export-ack.json').write_text('{}');(output/'epoch-999-note.txt').write_text('')
  okay('H44 ignores same-prefix receipt files',contract.complete_epoch_directories(output)==[cp])
  fake={'status':'persistent_checkpoint_verified','completed_epochs':1,'global_step':16,'trainer_sha256':'a'*64,'source_checkpoint_directory':str(cp),'checkpoint_sha256':contract.sha(cp/'checkpoint.json'),'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_directory':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen35-fulltext32-formal-v1/checkpoints/epoch-001','transport_manifest_sha256':'b'*64,'fsync_all_files':True,'reopened_final_filename_SHA_all_files':True,'destination_directories_fsynced':True,'files':{str(p.relative_to(cp)):{'bytes':p.stat().st_size,'sha256':contract.sha(p)} for p in cp.rglob('*') if p.is_file()}}
  fake['files']['_run/run_manifest.json']={'bytes':2,'sha256':contract.sha(output/'run_manifest.json')};ack=output/'export-ack-epoch-001.json'
  def call(value):ack.write_text(json.dumps(value));return contract.validate_export_ack(ack,output,cp,meta,'a'*64)
  okay('exact five-field durable ack contract accepted',call(fake)['global_step']==16)
  for key,val in [('global_step',0),('checkpoint_sha256','0'*64),('trainer_sha256','c'*64),('source_checkpoint_directory','/other'),('fsync_all_files',False),('reopened_final_filename_SHA_all_files',False),('destination_directories_fsynced',False),('persistent_directory','/tmp/foo')]:
   bad=copy.deepcopy(fake);bad[key]=val;rejected('ACK '+key+' mismatch rejected',lambda bad=bad:call(bad))
  bad=copy.deepcopy(fake);del bad['files']['training_state.pt'];rejected('ACK adapter-only warmstart rejected',lambda:call(bad))
  bad=copy.deepcopy(fake);bad['files']['training_state.pt']['sha256']='0'*64;rejected('ACK state SHA mismatch rejected',lambda:call(bad))
  bad=copy.deepcopy(fake);bad['files']['_run/../outside']={'bytes':0,'sha256':'0'*64};rejected('ACK traversal rejected',lambda:call(bad))
  rejected('old expired deadline rejected',lambda:contract.check_deadline(time.time()-1))
  staging=output/'.epoch-002.incomplete';staging.mkdir();(staging/'state').write_bytes(b'synthetic-local-state');dest=output/'epoch-002'
  receipt=contract.local_publish_checkpoint(staging,dest,output);okay('SHM publishing contract fsync/reopen synthetic',receipt['files']['state']['sha256']==contract.sha(dest/'state') and receipt['persistent_export_ack_still_required'])
 # Source-level invariants: keep original training numerical statements verbatim.
 old_tree=ast.parse(Path(old['path']).read_text());new_tree=ast.parse((B/'train.py').read_text());astdump=lambda node:ast.dump(node,include_attributes=False)
 for name in ['batch_plan','collate','validate_stop_epoch','epoch_outcome']:
  get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  okay('original '+name+' AST exact',astdump(get(old_tree))==astdump(get(new_tree)))
 text=(B/'train.py').read_text();need=["scheduler=get_linear_schedule_with_warmup(optimizer,2,32)","optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])","random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])","torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])","(loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)","with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss","optimizer.step();scheduler.step();global_step+=1"]
 for line in need:okay('original numerical/restore line '+hashlib.sha256(line.encode()).hexdigest()[:8],line in text)
 okay('ACK validation precedes model loading',text.index('ack_receipt=contract.validate_export_ack')<text.index("status('load_complete_text_base')"))
 okay('full state verification precedes resumed model train',text.index("restored_digest=contract.tree_digest(actual_state,torch)")<text.index("model.train();first_update_checked=False"))
 okay('fixed deadline checked initial/pergroup/checkpoint',text.count('contract.check_deadline(args.absolute_deadline_unix)')>=5)
 okay('scalar Tensor digest handles Adam step0d',"y.reshape(-1).view(torch.uint8)" in (B/'runtime_contract.py').read_text())
 result={'status':'PASS_STDLIB_SOURCE_AND_EPOCH_BOUNDARY_CONTRACT_NOT_TORCH_MODEL_RESUME_PROOF','seconds':time.monotonic()-start,'checks':checks,'checks_count':len(checks),'source':ref(B/'train.py'),'contract_source':ref(B/'runtime_contract.py'),'cpu_source':ref(__file__),'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),'runtime_Torch_or_model_tested':False,'native_census_repeated':False,'selected_CE_numeric_repeated':False,'GPU_or_remote_or_quality':False,'limitations':'Actual tensor/optimizer/RNG restoration is asserted and receipted inside the authorized future run; CPU fixture proves new guard rejection and retains original source semantics, not GPU bitwise equivalence.'}
 with (B/'CPU-contract-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 signal.alarm(0)
 print(json.dumps({'status':result['status'],'checks':len(checks),'seconds':result['seconds'],'source_sha256':result['source']['sha256']}))
if __name__=='__main__':main()
