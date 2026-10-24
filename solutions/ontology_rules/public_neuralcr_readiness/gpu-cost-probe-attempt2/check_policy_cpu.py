"""CUDA-hidden actual TF2 policy API and original compat.v1 layer dtype test; no forward/model weights."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,traceback
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert not(R/'CPU-policy-result.json').exists();started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'status':'running','source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'GPU_hidden':True,'NCR_model_weights_loaded':False,'model_forward_calls':0,'competition_samples_read':False}
try:
 sys.path.insert(0,plan['native_module_directory']);sys.path.insert(0,plan['fasttext_runtime'])
 import numpy as np
 import tensorflow as tf
 assert tf.config.list_physical_devices('GPU')==[]
 assert '/tf-gpu-overlay/' in tf.__file__,tf.__file__
 tf.keras.mixed_precision.set_global_policy('float32');assert tf.keras.mixed_precision.global_policy().name=='float32'
 tf.config.experimental.enable_tensor_float_32_execution(False);assert not tf.config.experimental.tensor_float_32_execution_enabled()
 assert callable(tf.config.experimental.reset_memory_stats) and callable(tf.config.experimental.get_memory_info)
 import ncrmodel
 required=plan['required_files']['ncrmodel.py'];assert sha(ncrmodel.__file__)==required['sha256']
 conv=ncrmodel.tf.keras.layers.Conv1D(2,1);conv.build((None,2,100))
 dense=ncrmodel.tf.keras.layers.Dense(2);dense.build((None,2))
 hierarchy=ncrmodel.HierarchicalAggregate(2,np.array([[0,0],[1,1]],dtype=np.int64),np.ones(2,dtype=np.float32));hierarchy.build((None,2))
 checks=[]
 for name,layer in [('original_compat_Conv1D',conv),('original_compat_Dense',dense),('original_HierarchicalAggregate',hierarchy)]:
  assert layer.compute_dtype==layer.variable_dtype=='float32'
  assert all(x.dtype.name=='float32' for x in layer.weights)
  checks.append({'name':name,'class':type(layer).__module__+'.'+type(layer).__name__,'compute_dtype':layer.compute_dtype,'variable_dtype':layer.variable_dtype,'weights':[{'shape':list(x.shape),'dtype':x.dtype.name} for x in layer.weights]})
 result.update(status='pass_actual_CPU_policy_and_compat_layer_dtypes_no_forward',tensorflow=tf.__version__,tensorflow_file=tf.__file__,numpy=np.__version__,global_policy=tf.keras.mixed_precision.global_policy().name,TF32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),original_native_source_sha256=sha(ncrmodel.__file__),actual_layers=checks)
except BaseException as error:result.update(status='failed_preserved_stop',error=repr(error),traceback=traceback.format_exc());raise
finally:result['elapsed_seconds']=time.monotonic()-started;result['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'CPU-policy-result.json').write_text(json.dumps(result,indent=2)+'\n')
