"""One separate CPU setup: complete deployed supply, isolated TF4.50, one base asset."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


root=Path(sys.argv[1]);out=root/'cpu-setup';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
put(out/'started.json',dict(started_unix=time.time(),pid=os.getpid(),argv=sys.argv))
try:
    application=root/'probe-v1/application';supply=root/'probe-v1/supply';m=json.loads((supply/'manifest.json').read_text())
    data=b''.join(p.read_bytes() for p in sorted(supply.glob('train.part*')))
    assert len(data)==m['artifacts']['train']['bytes'] and hashlib.sha256(data).hexdigest()==m['artifacts']['train']['sha256']
    with (supply/'train.jsonl.gz').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    assert hashlib.sha256((supply/'train.jsonl.gz').read_bytes()).hexdigest()==m['artifacts']['train']['sha256']
    runtime=root/'runtime-t450';runtime.mkdir()
    command=[sys.executable,'-m','pip','install','--target',str(runtime),'--no-deps','transformers==4.50.0','tokenizers==0.21.4','huggingface_hub==0.29.3']
    with (out/'pip.stdout').open('x') as stdout,(out/'pip.stderr').open('x') as stderr:
        subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120,check=True)
    project=root.parent.parent;environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','USE_TF':'0','USE_FLAX':'0','OMP_NUM_THREADS':'1',
        'PYTHONPATH':':'.join(map(str,[runtime,project/'case-retrieval/round2/training-runtime',project/'case-retrieval/round2/runtime']))}
    code="""import json,torch,transformers,inspect,hashlib
from transformers import LongT5Config,LongT5ForConditionalGeneration
assert transformers.__version__=='4.50.0'
config=LongT5Config.from_pretrained(__import__('sys').argv[1],local_files_only=True)
with torch.device('meta'): model=LongT5ForConditionalGeneration(config)
parameters=sum(p.numel() for p in model.parameters());assert parameters==247587456
source=inspect.getsourcefile(LongT5ForConditionalGeneration)
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'parameters':parameters,'CUDA_initialized':torch.cuda.is_initialized(),'source':source,'source_sha256':hashlib.sha256(open(source,'rb').read()).hexdigest()}))
"""
    asset=root/'assets-base'
    with (out/'assets.stdout').open('x') as stdout,(out/'assets.stderr').open('x') as stderr:
        subprocess.run([sys.executable,str(application/'asset_prepare.py'),'--plan',str(application/'probe-plan.json'),'--output',str(asset)],stdout=stdout,stderr=stderr,timeout=240,check=True)
    result=subprocess.run([sys.executable,'-c',code,str(asset)],env=environment,capture_output=True,text=True,timeout=60)
    put(out/'runtime.json',dict(rc=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0
    put(out/'complete.json',dict(status='COMPLETE_NATIVE_CPU_RUNTIME_AND_SINGLE_BASE_ASSETS',seconds=time.monotonic()-start,
        python=sys.executable,environment={k:environment[k] for k in ('PYTHONPATH','USE_TF','USE_FLAX','OMP_NUM_THREADS')},
        runtime=json.loads(result.stdout),assets=str(asset/'complete.json'),supply=m['artifacts']['train'],GPU_calls=0))
except BaseException:
    import traceback
    put(out/'failure.json',dict(seconds=time.monotonic()-start,traceback=traceback.format_exc()));raise
