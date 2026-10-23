"""CPU-only native TensorFlow H5 restoration; no corpus, labels beyond fixed class IDs, or training."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json,time,hashlib,sys
from pathlib import Path
import numpy as np,h5py,tensorflow as tf
from transformers import AutoConfig,TFAutoModel
from nn_represent import BERT_RepresentationLayer
R=Path.cwd();start=time.monotonic();tf.config.set_visible_devices([],'GPU');tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
assert not tf.config.list_physical_devices('GPU')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def strings(xs):return [x.decode() if isinstance(x,bytes) else str(x) for x in xs]
plan=json.loads((R/'plan.json').read_text())
for n,h in plan['source_sha256'].items():assert sha(R/n)==h,n
weight=R/'biobert_PT_v1.2.h5';assert weight.stat().st_size==486624416 and sha(weight)=='1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573'
labels=(R/'lable.vocab').read_text().strip().split('\n');assert len(labels)==len(set(labels))==17266 and labels.count('HP:None')==1
rep=BERT_RepresentationLayer(str(R/'biobert-base'),str(R/'lable.vocab'),lowercase=False);assert [rep.index_2_label[str(i)] for i in range(len(labels))]==labels
assert rep.tokenizer.do_lower_case is False and len(rep.tokenizer)==28996
mapping_path=R.parent/'public-phenotagger-reuse/original-column-to-current-HPO.json';assert sha(mapping_path)=='8a07e31d30ca83c9b4ac35c971ad489b7e77394bea3011e5ac5ac0f078f72adf'
mapping=json.loads(mapping_path.read_text());assert len(mapping['mapping'])==17266
assert all(m['column']==i and m['original_label']==labels[i] for i,m in enumerate(mapping['mapping']))
cfg=AutoConfig.from_pretrained(R/'biobert-base',local_files_only=True);encoder=TFAutoModel.from_config(cfg)
x1=tf.keras.layers.Input(shape=(32,),dtype=tf.int32,name='input_ids');x2=tf.keras.layers.Input(shape=(32,),dtype=tf.int32,name='token_type_ids');x3=tf.keras.layers.Input(shape=(32,),dtype=tf.int32,name='attention_mask')
x=encoder(x1,token_type_ids=x2,attention_mask=x3)[0][:,0,:];y=tf.keras.layers.Dense(len(labels),activation='softmax')(x);model=tf.keras.Model(inputs=[x1,x2,x3],outputs=y)
# Every native float variable is made invalid before strict restoration. This does not alter the file.
for w in model.weights:
 assert w.dtype.is_floating;w.assign(tf.fill(w.shape,tf.constant(float('nan'),dtype=w.dtype)))
model.load_weights(str(weight)) # native H5 topology load: no by_name, no skip_mismatch, no conversion
checks=[];saved_count=0;saved_elements=0
with h5py.File(weight,'r') as f:
 root=f['model_weights'] if 'model_weights' in f else f
 layer_names=strings(root.attrs['layer_names']);saved_layers=[]
 for name in layer_names:
  g=root[name];names=strings(g.attrs.get('weight_names',[]))
  if names:saved_layers.append((name,g,names))
 actual_layers=[l for l in model.layers if l.weights];assert len(actual_layers)==len(saved_layers)
 for actual,(name,g,names) in zip(actual_layers,saved_layers):
  assert len(actual.weights)==len(names),(actual.name,name,len(actual.weights),len(names))
  for variable,stored in zip(actual.weights,names):
   v=variable.numpy();a=np.asarray(g[stored]);assert v.shape==a.shape and np.isfinite(v).all() and np.array_equal(v,a),(variable.name,stored,v.shape,a.shape)
   checks.append({'model_variable':variable.name,'stored_variable':name+'/'+stored,'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()});saved_count+=1;saved_elements+=a.size
assert saved_count==len(model.weights)==201;assert saved_elements==model.count_params()==121587826
(R/'variable-coverage.json').write_text(json.dumps(checks,indent=2)+'\n')
phrases=['microcephaly','hearing loss','short stature','clinical report'];instances=[[(word,) for word in p.split()] for p in phrases];inputs,_=rep.load_data(instances,[],word_max_len=32,training=False)
upper_instances=[[(w.upper(),) for w in p.split()] for p in phrases];upper_inputs,_=rep.load_data(upper_instances,[],word_max_len=32,training=False)
assert all(np.array_equal(a,b) for a,b in zip(inputs,upper_inputs)) # original load_data lowercases before cased tokenizer
(R/'synthetic-representation.json').write_text(json.dumps({'phrases':phrases,'tokenizer_do_lower_case':rep.tokenizer.do_lower_case,'upstream_pre_lowering_retained':True,'uppercase_lowercase_representation_exact':True,'input_ids':inputs[0].tolist(),'token_type_ids':inputs[1].tolist(),'attention_mask':inputs[2].tolist()},indent=2)+'\n')
t=time.monotonic();probs=model(inputs,training=False).numpy();infer_seconds=time.monotonic()-t;assert probs.shape==(4,17266) and np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1.,atol=1e-5)
probe=[{'synthetic_phrase':p,'current_HPO':mapping['mapping'][int(np.argmax(row))]['current_label'],'raw_column':int(np.argmax(row)),'raw_class_id':labels[int(np.argmax(row))],'raw_probability':float(np.max(row))} for p,row in zip(phrases,probs)]
report={'scope':'Native public H5 CPU loader and four synthetic-phrase interface probes only; no benchmark evaluation or training','tf':tf.__version__,'model_class':type(encoder).__name__,'encoder_initialization':'from_config only; all variables overwritten by strict native H5 load and exact dataset equality','keras_h5_load':'topological, by_name=False, skip_mismatch=False','parameters':int(model.count_params()),'native_variables':len(model.weights),'H5_variable_elements':int(saved_elements),'all_variables_poisoned_before_load':True,'all_variables_finite_after_load':True,'all_variables_exact_H5_dataset_equal':True,'label_columns':len(labels),'HP_None_index':labels.index('HP:None'),'original_label_bytes_sha256':sha(R/'lable.vocab'),'ordered_label_ids_sha256':hashlib.sha256('\n'.join(labels).encode()).hexdigest(),'variable_coverage_sha256':sha(R/'variable-coverage.json'),'weight_sha256':sha(weight),'source_sha256':sha(__file__),'upstream_rep_source_sha256':sha(R/'nn_represent.py'),'input_hashes':{n:sha(R/'biobert-base'/n) for n in ['config.json','vocab.txt']},'gpus_visible':tf.config.list_physical_devices('GPU'),'CPU_only':True,'tokenizer_do_lower_case':rep.tokenizer.do_lower_case,'original_upstream_pre_lowering_retained':True,'current_mapping_sha256':sha(mapping_path),'full_head_column_order_mapping_preserved':True,'synthetic_representation_sha256':sha(R/'synthetic-representation.json'),'plan_sha256':sha(R/'plan.json'),'synthetic_output':probe,'synthetic_outputs_are_not_quality_estimates':True,'four_phrase_forward_seconds':infer_seconds,'elapsed_seconds':time.monotonic()-start}
(R/'native-h5-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
