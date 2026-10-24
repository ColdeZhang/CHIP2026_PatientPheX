"""One fresh original-H5 train32 epoch; no held labels or quality."""
import argparse,datetime,gzip,hashlib,json,math,os,subprocess,time
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def safe(x):
 if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':repr(x)}
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x
def write(p,x):
 with Path(p).open('x') as f:json.dump(safe(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def strings(xs):return [x.decode() if isinstance(x,bytes) else str(x) for x in xs]

def main(plan_path):
 started=time.monotonic();D=Path.cwd();plan=json.loads(Path(plan_path).read_text());P=Path(plan['native_root']);out=D/'actual';out.mkdir(exist_ok=False)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid'];assert sha(__file__)==plan['sources']['trainer.py']['sha256'];assert sha(D/'loss.py')==plan['sources']['loss.py']['sha256']
 for item in plan['assets'].values():assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
 weight=Path(plan['assets']['public_H5']['path']);before_file=ref(weight);dataset=D/'training-queries.jsonl.gz';assert sha(dataset)==plan['dataset']['sha256']
 rows=[json.loads(x) for x in gzip.open(dataset,'rt')];assert len(rows)==1252
 sampling=json.loads((D/'sampling.json').read_text());assert sha(D/'sampling.json')==plan['sampling']['sha256'];order=sampling['single_epoch_order'];assert sorted(order)==list(range(1252))
 for batch in sampling['formal_batches']:
  assert [rows[i]['token_sha256'] for i in batch['indices']]==batch['token_sha256']
 import numpy as np,h5py,tensorflow as tf,transformers
 from transformers import AutoConfig,TFAutoModel
 from loss import canonical_mass_batch_loss,canonical_mass_row_reference
 assert tf.__version__=='2.12.0' and transformers.__version__=='4.30.1'
 assert str(P/'tf-gpu-overlay') in tf.__file__;assert tf.keras.mixed_precision.global_policy().compute_dtype=='float32'
 gpus=tf.config.list_physical_devices('GPU');assert len(gpus)==1;tf.config.experimental.set_memory_growth(gpus[0],True)
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 def memory():
  result={'TF_allocator_bytes':{k:int(v) for k,v in tf.config.experimental.get_memory_info('GPU:0').items()}}
  try:
   line=subprocess.check_output(['nvidia-smi','-i',plan['gpu_uuid'],'--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip();parts=[x.strip() for x in line.split(',')];assert parts[0]==plan['gpu_uuid'];result['physical_device_observed_MiB']={'used':int(parts[1]),'free':int(parts[2])}
  except BaseException as e:result['physical_observation_error']=repr(e)
  return result
 runtime=dict(at=now(),tensorflow=tf.__version__,tensorflow_path=tf.__file__,transformers=transformers.__version__,numpy=np.__version__,GPU_visible=[x.name for x in gpus],CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],memory_growth=tf.config.experimental.get_memory_growth(gpus[0]),TF32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),compute_policy=tf.keras.mixed_precision.global_policy().name)
 write(out/'runtime.json',runtime)
 labels=Path(plan['assets']['label_vocab']['path']).read_text().splitlines();assert len(labels)==17266 and labels[17265]=='HP:None'
 load_start=time.monotonic();cfg=AutoConfig.from_pretrained(P/'pubmedbert-base',local_files_only=True);encoder=TFAutoModel.from_config(cfg)
 a=tf.keras.Input((32,),dtype=tf.int32,name='input_ids');b=tf.keras.Input((32,),dtype=tf.int32,name='token_type_ids');c=tf.keras.Input((32,),dtype=tf.int32,name='attention_mask')
 hidden=encoder(a,token_type_ids=b,attention_mask=c)[0][:,0,:];head=tf.keras.layers.Dense(17266,activation='softmax');prob=head(hidden);model=tf.keras.Model([a,b,c],prob);hidden_model=tf.keras.Model(model.inputs,head.input)
 assert model.count_params()==122759794 and len(model.weights)==len(model.trainable_variables)==201
 expected=json.loads((D/'variable-coverage.json').read_text());assert sha(D/'variable-coverage.json')==plan['prior_variable_coverage']['sha256'];assert [v.name for v in model.weights]==[x['model_variable'] for x in expected]
 for v in model.weights:
  assert v.dtype==tf.float32;v.assign(tf.fill(v.shape,tf.constant(float('nan'),v.dtype)))
 model.load_weights(str(weight))
 coverage=[]
 with h5py.File(weight,'r') as f:
  root=f['model_weights'] if 'model_weights'in f else f;saved=[]
  for name in strings(root.attrs['layer_names']):
   g=root[name];names=strings(g.attrs.get('weight_names',[]))
   if names:saved.append((name,g,names))
  actual=[x for x in model.layers if x.weights];assert len(actual)==len(saved)
  for layer,(name,g,names) in zip(actual,saved):
   assert len(layer.weights)==len(names)
   for v,n in zip(layer.weights,names):
    value=v.numpy();raw=np.asarray(g[n]);equal=value.shape==raw.shape and np.isfinite(value).all() and np.array_equal(value,raw)
    coverage.append(dict(model_variable=v.name,stored_variable=name+'/'+n,shape=list(value.shape),sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),H5_equal=bool(equal)))
 write(out/'strict-load-coverage.json',coverage)
 prior_variable_hashes_exact=all(x['sha256']==y['sha256'] for x,y in zip(coverage,expected)) and len(coverage)==len(expected)
 load_report=dict(seconds=time.monotonic()-load_start,parameters=model.count_params(),variables=len(coverage),all_exact=len(coverage)==201 and all(x['H5_equal'] for x in coverage) and prior_variable_hashes_exact,head_columns=17266,None_column=17265,memory=memory())
 write(out/'strict-load.json',load_report);assert load_report['all_exact']
 variables=model.trainable_variables;names=[x.name for x in variables];expected_disconnected=set(plan['expected_disconnected_variables']);assert len(expected_disconnected)==2 and expected_disconnected<=set(names)
 def hashes():return {v.name:hashlib.sha256(v.numpy().tobytes(order='C')).hexdigest() for v in variables}
 original_hashes=hashes()
 def batch_data(batch):
  selected=[rows[i] for i in batch['indices']];inputs=[tf.constant([x[k] for x in selected],tf.int32) for k in ['input_ids','token_type_ids','attention_mask']];mask=np.zeros((len(selected),17266),dtype=bool)
  for j,row in enumerate(selected):
   cols=row['target_columns'];assert cols and len(cols)==len(set(cols)) and all(type(x)is int and 0<=x<17266 for x in cols);mask[j,cols]=True
  return selected,inputs,tf.constant(mask)
 @tf.function
 def native_infer(a,b,c):return model([a,b,c],training=False)
 @tf.function
 def exposed_infer(a,b,c):
  h=hidden_model([a,b,c],training=False);z=tf.matmul(h,head.kernel)+head.bias;return z,tf.nn.softmax(z)
 selected,x,mask=batch_data(sampling['formal_batches'][0]);contract_start=time.monotonic();native=native_infer(*x).numpy();z,exposed=exposed_infer(*x);exposed=exposed.numpy();diff=np.abs(native-exposed)
 raw=out/'inference-contract-arrays.npz';np.savez_compressed(raw,native_softmax=native,shared_Dense_softmax=exposed)
 with raw.open('rb') as f:os.fsync(f.fileno())
 tol=plan['tolerances'];checks=dict(shape=bool(native.shape==exposed.shape==(64,17266)),finite=bool(np.isfinite(native).all() and np.isfinite(exposed).all()),normalized_rows=bool(np.allclose(native.sum(1),1.,rtol=0,atol=tol['probability_sum_atol']) and np.allclose(exposed.sum(1),1.,rtol=0,atol=tol['probability_sum_atol']) and (native>=0).all() and (exposed>=0).all()),allclose=bool(np.allclose(native,exposed,rtol=tol['inference_rtol'],atol=tol['inference_atol'])),row_L1=bool(np.max(diff.sum(1))<=tol['inference_row_L1_max']),same_argmax=bool(np.array_equal(np.argmax(native,axis=1),np.argmax(exposed,axis=1))),head_activation_unchanged=head.activation is tf.keras.activations.softmax,same_head_variables=model.weights[-2] is head.kernel and model.weights[-1] is head.bias,weights_unchanged_before_training=hashes()==original_hashes)
 contract=dict(batch=sampling['formal_batches'][0]['name'],seconds=time.monotonic()-contract_start,max_abs_diff=float(diff.max()),max_row_L1=float(diff.sum(1).max()),per_row_max_abs=diff.max(1).tolist(),checks=checks,arrays=ref(raw),memory=memory());write(out/'inference-contract.json',contract);assert all(checks.values()),'inference contract'
 del native,exposed,z,diff
 tf.keras.utils.set_random_seed(plan['seed']);optimizer=tf.keras.optimizers.Adam(learning_rate=plan['learning_rate'],beta_1=plan['Adam']['beta_1'],beta_2=plan['Adam']['beta_2'],epsilon=plan['Adam']['epsilon'],amsgrad=False);assert int(optimizer.iterations.numpy())==0
 @tf.function
 def gradients(a,b,c,target_mask):
  with tf.GradientTape() as tape:
   h=hidden_model([a,b,c],training=True);logits=tf.matmul(h,head.kernel)+head.bias;losses=canonical_mass_batch_loss(logits,target_mask);loss=tf.reduce_mean(losses)
  grads=tape.gradient(loss,variables);return logits,losses,loss,grads
 step_reports=[];previous_hashes=original_hashes;slot_ids=None
 for step,batch in enumerate(sampling['formal_batches']):
  assert time.time()<float(os.environ['CAMPAIGN_DEADLINE_EPOCH'])-15;assert len(sampling['formal_batches'])==20
  selected,x,mask=batch_data(batch);step_start=time.monotonic();state_before=int(optimizer.iterations.numpy());assert state_before==step
  opt_before=optimizer.variables();pre=dict(batch=batch['name'],size=len(selected),optimizer_iteration=state_before,optimizer_variables=len(opt_before),memory=memory());write(out/f'step{step}-before.json',pre)
  fb_start=time.monotonic();logits,losses,loss,grads=gradients(*x,mask);loss_value=float(loss.numpy());global_gradient_norm=float(tf.linalg.global_norm([g for g in grads if g is not None]).numpy());fb_seconds=time.monotonic()-fb_start
  disconnected=[v.name for g,v in zip(grads,variables) if g is None];grad_obs=[]
  for g,v in zip(grads,variables):
   if g is None:grad_obs.append(dict(name=v.name,connected=False));continue
   values=g.values if isinstance(g,tf.IndexedSlices) else g
   grad_obs.append(dict(name=v.name,connected=True,kind=type(g).__name__,finite=bool(tf.reduce_all(tf.math.is_finite(values)).numpy()),norm=float(tf.linalg.global_norm([g]).numpy())))
  reference=tf.stack([canonical_mass_row_reference(logits[j],row['target_columns']) for j,row in enumerate(selected)]);delta=tf.abs(reference-losses).numpy();loss_array=losses.numpy();raw_columns=[len(row['target_columns']) for row in selected]
  before_update=dict(step=step,batch=batch['name'],forward_backward_seconds=fb_seconds,mean_loss=loss_value,global_gradient_norm=global_gradient_norm,row_loss_finite=bool(np.isfinite(loss_array).all()),disconnected=disconnected,gradient_observations=grad_obs,batched_row_loss_max_abs_diff=float(delta.max()),batched_row_loss_allclose=bool(np.allclose(reference.numpy(),loss_array,rtol=tol['row_loss_rtol'],atol=tol['row_loss_atol'])),None_rows=sum(row['target_columns']==[17265] for row in selected),equivalent_column_rows=sum(n>1 for n in raw_columns),None_mean_loss=float(np.mean([value for value,row in zip(loss_array,selected) if row['target_columns']==[17265]])),equivalent_mean_loss=float(np.mean([value for value,n in zip(loss_array,raw_columns) if n>1])) if any(n>1 for n in raw_columns) else None,memory=memory())
  write(out/f'step{step}-before-update.json',before_update)
  assert np.isfinite(loss_value) and np.isfinite(global_gradient_norm) and before_update['row_loss_finite'] and before_update['batched_row_loss_allclose'];assert set(disconnected)==expected_disconnected
  assert all(x['finite'] for x in grad_obs if x['connected']);assert any(x.get('norm',0)>0 for x in grad_obs)
  update_start=time.monotonic();optimizer.apply_gradients([(g,v) for g,v in zip(grads,variables) if g is not None]);iteration=int(optimizer.iterations.numpy());post_model_finite=bool(tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in variables])).numpy());update_seconds=time.monotonic()-update_start
  current_hashes=hashes();changed={name for name in names if current_hashes[name]!=previous_hashes[name]};statevars=optimizer.variables();new_slot_ids=[id(v) for v in statevars];slot_stable=(slot_ids is None or slot_ids==new_slot_ids);slot_ids=new_slot_ids
  slot_finite=all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in statevars if v.dtype.is_floating)
  families={family:sum(name in changed and (name in [head.kernel.name,head.bias.name] if pattern=='__exact_original_head__' else pattern in name) for name in names) for family,pattern in plan['update_families'].items()}
  checks=dict(iteration_exact=iteration==step+1,optimizer_variable_count=len(statevars)==399,optimizer_state_not_recreated=slot_stable,all_slots_finite=slot_finite,unchanged_unused_pooler=not bool(changed&expected_disconnected),no_unexpected_updated_variables=changed<=(set(names)-expected_disconnected),all_required_families_updated=all(n>0 for n in families.values()),all_model_variables_finite=post_model_finite,nonzero_original_lr=bool(np.isclose(float(optimizer.learning_rate.numpy()),5e-6,rtol=1e-6,atol=0)))
  report=dict(step=step,batch=batch['name'],batch_size=len(selected),forward_backward_seconds=fb_seconds,Adam_apply_seconds=update_seconds,compute_update_seconds=fb_seconds+update_seconds,elapsed_including_audit_seconds=time.monotonic()-step_start,changed_variable_count=len(changed),changed_variables=sorted(changed),families=families,checks=checks,optimizer_iteration=iteration,optimizer_variables=len(statevars),memory=memory());write(out/f'step{step}-after-update.json',report);assert all(checks.values()),'update checks';step_reports.append(report);previous_hashes=current_hashes
  del grads,logits,losses,loss,reference,mask,x
 assert int(optimizer.iterations.numpy())==20 and sum(x['batch_size'] for x in step_reports)==1252
 # Persist exactly the terminal state; this campaign never resumes a partial epoch.
 import pickle,random
 state_dir=D/'trained-state';state_dir.mkdir(exist_ok=False)
 trained_h5=state_dir/'adapted.weights.h5';model.save_weights(str(trained_h5),save_format='h5')
 checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
 checkpoint_path=checkpoint.save(str(state_dir/'checkpoint'))
 rng=dict(python=random.getstate(),numpy=np.random.get_state(),tensorflow_global_generator_state=tf.random.get_global_generator().state.numpy(),tensorflow_global_generator_algorithm=int(tf.random.get_global_generator().algorithm),tensorflow_global_seed=plan['seed'],completed_epoch=1,global_step=20,order=sampling['single_epoch_order'],limitations='Legacy TensorFlow graph Dropout stateful kernel call counters are not exported. No claim of arbitrary mid-trace bitwise resume. Whole epoch finished; no continuation in this campaign.')
 with (state_dir/'rng-and-order.pkl').open('xb') as f:pickle.dump(rng,f,protocol=4);f.flush();os.fsync(f.fileno())
 terminal_model=[v.numpy() for v in model.weights];terminal_optimizer=[v.numpy() for v in optimizer.variables()]
 # New independent objects, same trainable topology. Build Adam slots without an update.
 enc2=TFAutoModel.from_config(cfg)
 a2=tf.keras.Input((32,),dtype=tf.int32);b2=tf.keras.Input((32,),dtype=tf.int32);c2=tf.keras.Input((32,),dtype=tf.int32)
 h2=enc2(a2,token_type_ids=b2,attention_mask=c2)[0][:,0,:];head2=tf.keras.layers.Dense(17266,activation='softmax');model2=tf.keras.Model([a2,b2,c2],head2(h2))
 assert len(model2.weights)==201 and model2.count_params()==122759794
 for v in model2.weights:v.assign(tf.fill(v.shape,tf.constant(float('nan'),v.dtype)))
 model2.load_weights(str(trained_h5))
 h5_equal=[bool(np.array_equal(v.numpy(),old)) for v,old in zip(model2.weights,terminal_model)]
 write(out/'adapted-H5-reload-before-gate.json',dict(variables=len(h5_equal),each_variable_exact=h5_equal,all_exact=all(h5_equal),new_object=True));assert all(h5_equal)
 opt2=tf.keras.optimizers.Adam(learning_rate=plan['learning_rate'],beta_1=plan['Adam']['beta_1'],beta_2=plan['Adam']['beta_2'],epsilon=plan['Adam']['epsilon'],amsgrad=False)
 # Mapping by original variable positions avoids new Keras layer suffix identities.
 active_indices=[i for i,v in enumerate(model.trainable_variables) if v.name not in expected_disconnected]
 opt2.build([model2.trainable_variables[i] for i in active_indices]);assert int(opt2.iterations.numpy())==0
 ckpt2=tf.train.Checkpoint(model=model2,optimizer=opt2);_ = ckpt2.save_counter;status=ckpt2.restore(checkpoint_path);status.assert_consumed()
 model_equal=[bool(np.array_equal(v.numpy(),old)) for v,old in zip(model2.weights,terminal_model)]
 optimizer_equal=[bool(np.array_equal(v.numpy(),old)) for v,old in zip(opt2.variables(),terminal_optimizer)]
 complete_state=dict(model_variables=len(model_equal),optimizer_variables=len(optimizer_equal),model_all_exact=len(model_equal)==201 and all(model_equal),optimizer_all_exact=len(optimizer_equal)==399 and all(optimizer_equal),new_object_global_step=int(opt2.iterations.numpy()),checkpoint_assert_consumed=True,training_updates_during_restore=0,dropout_resume_limitation=rng['limitations'])
 write(out/'state-reopen-before-gate.json',complete_state);assert complete_state['model_all_exact'] and complete_state['optimizer_all_exact'] and complete_state['new_object_global_step']==20
 selected,x,mask=batch_data(sampling['formal_batches'][0]);before_prob=model(x,training=False).numpy();after_prob=model2(x,training=False).numpy();diff=np.abs(before_prob-after_prob)
 restored_inference=dict(max_abs_diff=float(diff.max()),allclose=bool(np.allclose(before_prob,after_prob,rtol=1e-5,atol=1e-7)),same_argmax=bool(np.array_equal(before_prob.argmax(1),after_prob.argmax(1))))
 write(out/'restored-inference-before-gate.json',restored_inference);assert restored_inference['allclose'] and restored_inference['same_argmax']
 durable={}
 for p in sorted(state_dir.iterdir()):
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
   durable[p.name]=ref(p)
 for directory in [state_dir,D,D.parent]:
  fd=os.open(str(directory),os.O_RDONLY);os.fsync(fd);os.close(fd)
 for n,r in durable.items():assert sha(state_dir/n)==r['sha256'] and (state_dir/n).stat().st_size==r['bytes']
 write(out/'durability.json',dict(files=durable,all_fsync_reopen_SHA=True,full_state_restored=complete_state))
 after_file=ref(weight);after_assets={k:ref(v['path']) for k,v in plan['assets'].items()};assets_unchanged=all(after_assets[k]['sha256']==v['sha256'] and after_assets[k]['bytes']==v['bytes'] for k,v in plan['assets'].items());write(out/'original-file-after.json',dict(before=before_file,after=after_file,byte_unchanged=before_file==after_file,all_original_assets=after_assets,all_assets_unchanged=assets_unchanged));assert before_file==after_file and assets_unchanged
 complete=dict(status='COMPLETE_SINGLE_FRESH_NATIVE_H5_TRAIN32_EPOCH_NO_QUALITY',at_utc=now(),seconds=time.monotonic()-started,optimizer_iterations=20,examples=1252,epochs=1,steps=step_reports,trained_H5=ref(trained_h5),durability=ref(out/'durability.json'),complete_state_restored=complete_state,original_H5_unchanged=True,held_input_or_gold_read=False,final_memory=memory())
 write(out/'complete.json',complete);print(json.dumps(dict(status=complete['status'],seconds=complete['seconds'],trained_H5=complete['trained_H5'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)
