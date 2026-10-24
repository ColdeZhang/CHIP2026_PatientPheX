from pathlib import Path
import sys,json,shlex
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
io.M='69b991ee-183f-4add-af1a-579af92479c9';E=Path(__file__).resolve().parent
code='''import sys,os,shutil,json,hashlib,inspect,datetime
from pathlib import Path
import torch,transformers,peft,tokenizers
from transformers import Gemma3ForConditionalGeneration
root=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
s=Path(inspect.getsourcefile(Gemma3ForConditionalGeneration))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),uid=os.getuid(),python=sys.executable,python_version=sys.version,
packages=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__),
modeling_source=dict(path=str(s),bytes=s.stat().st_size,sha256=hashlib.sha256(s.read_bytes()).hexdigest()),
project=str(root),project_disk_free=shutil.disk_usage(root).free,shm_free=shutil.disk_usage('/dev/shm').free,
proposed_private_root_exists=(Path('/dev/shm/patientphex-medgemma15-1045-20261024')).exists(),cuda_initialized=torch.cuda.is_initialized(),no_model_loaded=True)))
'''
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a800-migration/runtime','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
command=shlex.join(['env',*[k+'='+v for k,v in env.items()],'/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python','-c',code])
r=io.call('run_command',{'command':command,'cwd':'.','timeout_seconds':45})
(E/'a800-reference-readonly.json').write_text(json.dumps({'mount_id':io.M,'env':env,'result':r},indent=2)+'\n')
print(r)
