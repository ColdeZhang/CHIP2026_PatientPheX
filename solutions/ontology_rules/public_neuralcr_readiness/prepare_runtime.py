"""Install one exact binary wheel into owned overlay; CPU interface imports only."""
from pathlib import Path
import urllib.request,hashlib,json,subprocess,sys,os,time
R=Path(__file__).resolve().parent;t=time.monotonic()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'runtime-preparation.json').exists()
name='fasttext_wheel-0.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
url='https://files.pythonhosted.org/packages/3a/78/5af1ed52339a6c09277921f56e60cc1c167154147a3038ab77aaaf45ef19/'+name
expected='0aa4755a3ab0717e32627ede55e9c12cd7bbba464c73af7f08a3142bd6c62df7'
wheel=R/name;assert not wheel.exists()
with urllib.request.urlopen(url,timeout=45) as r: data=r.read(4421997)
assert len(data)==4421996 and hashlib.sha256(data).hexdigest()==expected;wheel.write_bytes(data)
runtime=R/'runtime';assert not runtime.exists()
cmd=[sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(runtime),str(wheel)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
report={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wheel':{'filename':name,'bytes':len(data),'sha256':expected,'url':url},'install':{'argv':cmd,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr},'shared_environment_unchanged':True,'model_NN_executed':False}
try:
 assert p.returncode==0
 sys.path.insert(0,str(runtime));import fasttext;import tensorflow.compat.v1 as tf
 symbols=['enable_eager_execution','sparse_reorder','sparse_tensor_dense_matmul']
 report['interfaces']={'fasttext_module':fasttext.__file__,'tensorflow_version':tf.__version__,'compat_v1_symbols':{x:hasattr(tf,x) for x in symbols},'visible_GPU':tf.config.list_physical_devices('GPU')}
 assert all(report['interfaces']['compat_v1_symbols'].values()) and report['interfaces']['visible_GPU']==[]
 report['status']='pass_import_only'
except Exception as e:
 report['status']='failed_preserved';report['error']=repr(e);raise
finally:
 report['elapsed_seconds']=time.monotonic()-t;(R/'runtime-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
