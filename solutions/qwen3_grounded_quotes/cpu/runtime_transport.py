"""One bounded CPU-only MCP deployment. No automatic retries or GPU controls."""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import zipfile
from common import ref,write

P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
SOURCE_NAMES=['common.py','token_census.py','native_cpu.py','reuse/native_tokenizer.py']

def main():
    p=argparse.ArgumentParser();p.add_argument('--deployment',required=True);a=p.parse_args()
    dep=json.loads(Path(a.deployment).read_text());assert re.fullmatch('[0-9a-f]{40}',dep['source_commit'])
    out=Path(dep['local']);out.mkdir(parents=True,exist_ok=False)
    write(out/'deployment.json',dep);remote=dep['remote'];W=Path(__file__).resolve().parent
    payload={n:(W/n).read_bytes() for n in SOURCE_NAMES}
    for name,item in dep['data'].items():
        assert ref(item['path'])==item
        payload[name+'.gz']=gzip.compress(Path(item['path']).read_bytes(),mtime=0)
    def identity(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    plan=dict(route_id='qwen3-grounded-quotes-v1',model=dep['model'],model_dir=dep['model_dir'],model_assets=dep['model_assets'],
        runtime_versions=dep['runtime_versions'],template_kind=dep['template_kind'],template_file=dep['template_file'],
        eos_token_id=dep['eos_token_id'],output=remote+'/actual',
        compressed_inputs={k:dict(path=remote+'/'+k+'.gz',raw_sha256=dep['data'][k]['sha256']) for k in ['train','held_input']},
        expected_local_native=dep.get('expected_local_native'),
        files={remote+'/'+n:dict(path=remote+'/'+n,**identity(b)) for n,b in payload.items()})
    payload['cpu-plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    manifest={n:identity(b) for n,b in payload.items()};buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    archive=buffer.getvalue();(out/'deployment.zip').write_bytes(archive)
    write(out/'package.json',dict(source=ref(__file__),source_commit=dep['source_commit'],files=manifest,archive=ref(out/'deployment.zip'),remote=remote,plan=plan))
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    c=Client();c.initialize();idx=0
    def call(tool,args):
        nonlocal idx
        idx+=1;tag=f'{idx:03d}-{tool}';safe=dict(args)
        if 'content' in safe:safe['content']='[frozen archive bytes omitted]'
        write(out/(tag+'.intent.json'),dict(tool=tool,args=safe,at=time.time()))
        try:r=c.request('tools/call',dict(name=tool,arguments=args))
        except Exception as e:
            write(out/(tag+'.UNKNOWN.json'),dict(error_type=type(e).__name__,no_automatic_retry=True));raise
        if r.get('isError'):
            write(out/(tag+'.ERROR.json'),r);raise RuntimeError('MCP tool error; preserve original intent, no replay')
        v=r.get('structuredContent')
        if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
        meta={k:v for k,v in v.items() if tool!='read_file' or k!='content'}
        write(out/(tag+'.response.json'),meta);return v
    rel=remote[len(P)+1:];chunks=[]
    # Each chunk is immutable in this unique run directory; ambiguous write stops.
    for offset in ([] if dep.get('reuse_payload_root') else range(0,len(archive),32768)):
        block=archive[offset:offset+32768];name=f'package-{offset:09d}.chunk'
        call('write_file',dict(mount_id=dep['mount'],path=rel+'/'+name,content=base64.b64encode(block).decode(),encoding='base64',create_parents=True))
        chunks.append(dict(name=name,**identity(block)))
    spec=dict(root=remote,chunks=chunks,archive_sha256=hashlib.sha256(archive).hexdigest(),files=manifest,
        python=dep['python'],pythonpath=dep['pythonpath'],budget=50,grace=5)
    if dep.get('reuse_payload_root'):
        spec.update(reuse_root=dep['reuse_payload_root'],new_plan_b64=base64.b64encode(payload['cpu-plan.json']).decode())
    code='SPEC='+repr(spec)+'\n'+REMOTE
    # External timeout remains effective if the caller loses connectivity.
    value=call('run_command',dict(mount_id=dep['mount'],cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=60))
    assert value['exit_code']==0,'CPU wrapper failed; preserve raw logs; no restart'
    remote_result=json.loads(value['stdout']);write(out/'remote-complete.json',remote_result)
    for name,item in remote_result['files'].items():
        target=out/'collected'/name;target.parent.mkdir(parents=True,exist_ok=True);data=bytearray()
        for offset in range(0,item['bytes'],131072):
            v=call('read_file',dict(mount_id=dep['mount'],path=rel+'/'+name,offset=offset,limit=min(131072,item['bytes']-offset)))
            b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode();data.extend(b)
        assert identity(data)==item,name
        with target.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
        assert identity(target.read_bytes())==item
    if remote_result['returncode']!=0:
        write(out/'failed-endpoint.json',dict(status='CPU_FAILED_RAW_COLLECTED',returncode=remote_result['returncode'],
            stderr=ref(out/'collected/stderr.log'),wrapper=ref(out/'collected/wrapper-complete.json'),no_automatic_retry=True))
        raise RuntimeError('CPU endpoint failed; logs and original receipt collected; no restart')
    actual=json.loads((out/'collected/actual/result.json').read_text())
    assert actual['status']=='PASS_QWEN3_GROUNDED_126_NATIVE_CPU_NO_MODEL' and actual['counts']['train']['rows']==87 and actual['counts']['held']['rows']==39
    final=dict(status='COLLECTED_ACTUAL_CPU_PASS',deployment=ref(a.deployment),package=ref(out/'package.json'),
        remote_result=ref(out/'remote-complete.json'),result=ref(out/'collected/actual/result.json'),
        model=dep['model'],no_model_GPU_quality=True)
    print(json.dumps(write(out/'collection.json',final)),flush=True)

REMOTE=r'''
import os,json,time,hashlib,zipfile,io,subprocess,base64
from pathlib import Path,PurePosixPath
root=Path(SPEC['root'])
def identity(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def save(p,v):put(p,(json.dumps(v,indent=2)+'\n').encode())
assert not (root/'launch-intent.json').exists() and not (root/'actual').exists()
if SPEC.get('reuse_root'):
 prior=Path(SPEC['reuse_root']);members={n:(base64.b64decode(SPEC['new_plan_b64']) if n=='cpu-plan.json' else (prior/n).read_bytes()) for n in SPEC['files']}
else:
 data=b''
 for chunk in SPEC['chunks']:
  b=(root/chunk['name']).read_bytes();assert identity(b)=={k:chunk[k] for k in ['bytes','sha256']};data+=b
 assert hashlib.sha256(data).hexdigest()==SPEC['archive_sha256']
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  assert set(z.namelist())==set(SPEC['files']) and len(z.namelist())==len(set(z.namelist()))
  members={n:z.read(n) for n in SPEC['files']}
for name,item in SPEC['files'].items():
 part=PurePosixPath(name);assert not part.is_absolute() and '..' not in part.parts
 b=members[name];assert identity(b)==item;put(root/name,b);assert identity((root/name).read_bytes())==item
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=SPEC['pythonpath']+':'+str(root))
started=time.time();save(root/'launch-intent.json',dict(at_unix=started,TERM_unix=started+50,KILL_unix=started+55,budget_seconds=50,grace_seconds=5,CUDA_VISIBLE_DEVICES='',no_model_GPU_quality=True))
with (root/'stdout.log').open('xb') as out,(root/'stderr.log').open('xb') as err:
 child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=5s','50s',SPEC['python'],str(root/'native_cpu.py'),'--plan',str(root/'cpu-plan.json')],env=env,cwd=root,stdout=out,stderr=err)
 save(root/'child-started.json',dict(pid=child.pid,at_unix=time.time()))
 rc=child.wait()
 for h in [out,err]:h.flush();os.fsync(h.fileno())
save(root/'wrapper-complete.json',dict(returncode=rc,seconds=time.time()-started,pid=child.pid,no_automatic_retry=True))
files={p.relative_to(root).as_posix():identity(p.read_bytes()) for p in root.rglob('*') if p.is_file() and (p.relative_to(root).parts[0]=='actual' or p.name in ['stdout.log','stderr.log','wrapper-complete.json','launch-intent.json','child-started.json']) and 'inputs' not in p.relative_to(root).parts and '__pycache__' not in p.relative_to(root).parts}
print(json.dumps(dict(returncode=rc,files=files)))
'''

if __name__=='__main__':main()
