"""CPU import plus private resource stat only; no weights, tensors or model."""
import os,time,json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
t0=time.monotonic()
import tensorflow as tf
import numpy,h5py
from importlib.metadata import version
P=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse')
def tree(p):
 if not p.exists():return {'path':str(p),'exists':False}
 files=[x for x in p.rglob('*') if x.is_file() and not x.is_symlink()]
 return {'path':str(p),'exists':True,'files':len(files),'bytes':sum(x.stat().st_size for x in files)}
libs=[dict(path=str(p),bytes=p.stat().st_size,realpath=str(p.resolve())) for p in sorted((P/'tf-gpu-overlay/nvidia/cudnn/lib').glob('*.so*')) if p.is_file()]
result=dict(at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],tensorflow_version=tf.__version__,tensorflow_path=tf.__file__,tensorflow_build_info=dict(tf.sysconfig.get_build_info()),numpy_version=numpy.__version__,h5py_version=h5py.__version__,transformers_metadata_version=version('transformers'),keras_metadata_version=version('keras'),cuDNN_distribution_version=version('nvidia-cudnn-cu11'),cuDNN_libraries=libs,nltk_data=tree(P/'nltk_data'),base_tokenizer_config=tree(P/'pubmedbert-base'),sources=tree(P/'native-pipeline/src'),label_file={'path':str(P/'lable.vocab'),'bytes':(P/'lable.vocab').stat().st_size},seconds=time.monotonic()-t0,scope='TensorFlow/numpy/h5py CPU import only under CUDA_VISIBLE_DEVICES empty. No model, weights, tensor/forward, GPU enumeration or operation.')
print(json.dumps(result))
