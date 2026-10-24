#!/usr/bin/env python3
"""One bounded Qwen SFT32 training-readiness native CPU operation. inspect/collect never launch the CPU job.

This controller is prepared locally; only the owner executes run after source backup.
MCP imports and credential access are delayed until an explicitly selected operation.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
import zipfile
sys.dont_write_bytecode = True
F = Path(__file__).resolve().parent
MOUNT = '69b991ee-183f-4add-af1a-579af92479c9'
ROOT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL = 'ontology-round2/qwen35-sft32-full-report-training-cpu-v1'
REMOTE = ROOT + '/' + REL
PYTHON = ROOT + '/ontology-round2/qwen-venv/bin/python'
ENV = {'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':ROOT+'/case-retrieval/qwen35-preparation/a800-migration/runtime',
       'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
       'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
UPLOAD_CHUNK = 65536
READBACK_CHUNK = 49152
PLAN_SHA = 'a85cc2698c28b26db8069c5e1f7a30b4a154c64202ddfe3f27996f6e44a9ca04'
INSPECT_SHA = '54d32fe3644120456e473cd0299746bf52384c53401b35eff544781d9c1425b7'

INIT = r'''
from pathlib import Path
import os,json
p=Path(PARAM['root'])
assert os.getuid()==1045
p.mkdir(parents=True,exist_ok=False,mode=0o700)
assert p.stat().st_uid==1045
os.chmod(p,0o700)
with (p/'deployment-intent.json').open('x') as f:
 json.dump(PARAM['intent'],f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
print(json.dumps({'status':'EXCLUSIVE_PRIVATE_REMOTE_DEPLOYMENT_INTENT','uid':os.getuid(),'root':str(p)}))
'''

DEPLOY = r'''
from pathlib import Path
import os,json,hashlib,tarfile,io
p=Path(PARAM['root']);m=PARAM['manifest']
assert os.getuid()==p.stat().st_uid==1045
assert not (p/'CPU-start-intent.json').exists() and not (p/'native-result.json').exists()
parts=[]
for part in m['parts']:
 b=(p/part['name']).read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];parts.append(b)
b=b''.join(parts)
assert len(b)==m['bundle']['bytes'] and hashlib.sha256(b).hexdigest()==m['bundle']['sha256']
with (p/'source-inputs.tar.gz').open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tar:
 entries=tar.getmembers();assert len(entries)==5==len(m['files'])
 assert len({e.name for e in entries})==5 and set(e.name for e in entries)==set(m['files'])
 for e in entries:
  assert e.isfile() and not e.issym() and not e.islnk() and e.name==Path(e.name).name and '..' not in Path(e.name).parts
  data=tar.extractfile(e).read();expected=m['files'][e.name]
  assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
  target=p/e.name
  with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
for name in m['files']:
 if name.endswith('.py'):compile((p/name).read_text(),name,'exec')
result={'status':'ACTUAL_5_SOURCE_INPUT_SHA_COMPILE_FSYNC_CPU_ONLY','files':m['files'],'uid':os.getuid()}
with (p/'deployment-complete.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
print(json.dumps(result))
'''

CPU = r'''
from pathlib import Path
import subprocess,time,json,os,hashlib
p=Path(PARAM['root']);m=PARAM['manifest']
assert os.getuid()==p.stat().st_uid==1045 and os.environ['CUDA_VISIBLE_DEVICES']==''
assert (p/'deployment-complete.json').exists() and not (p/'native-result.json').exists()
for name,r in m['files'].items():
 b=(p/name).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
intent={'at_epoch':time.time(),'cap_seconds':60,'term_grace_seconds':5,'plan_sha256':PARAM['plan_sha256'],'controller_sha256':PARAM['controller_sha256'],'only_CPU_tokenizer':True}
with (p/'CPU-start-intent.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
started=time.monotonic()
with (p/'stdout.log').open('x') as out,(p/'stderr.log').open('x') as err:
 child=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=5s','60s',PARAM['python'],'-u',str(p/'native_cpu.py'),'--plan',str(p/'cpu-plan.json'),'--base','/dev/shm/patientphex-qwen35-1045-c20223623576'],cwd=p,stdout=out,stderr=err,start_new_session=True)
 with (p/'CPU-child.json').open('x') as f:json.dump({'pid':child.pid,'pgid':child.pid},f);f.write('\n');f.flush();os.fsync(f.fileno())
 rc=child.wait()
 out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
result={'returncode':rc,'wrapper_seconds':time.monotonic()-started,'cap_seconds':60,'term_grace_seconds':5,'plan_sha256':PARAM['plan_sha256'],'source_sha256':m['files']['native_cpu.py']['sha256'],'controller_sha256':PARAM['controller_sha256'],'only_CPU_tokenizer':True}
with (p/'CPU-complete.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
print(json.dumps(result))
'''

INSPECT = r'''
from pathlib import Path
import os,json,hashlib
p=Path(PARAM['root']);assert os.getuid()==1045
out={'root_exists':p.exists(),'path':str(p),'uid':os.getuid()}
if p.exists():
 assert p.stat().st_uid==1045
 for n in ['deployment-intent.json','deployment-complete.json','CPU-start-intent.json','CPU-child.json','CPU-complete.json']:
  if (p/n).is_file():out[n]=json.loads((p/n).read_text())
 for n in ['native-result.json','native-row-audit.json','stdout.log','stderr.log']:
  if (p/n).is_file():
   b=(p/n).read_bytes();out[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps(out))
'''

COLLECT = r'''
from pathlib import Path
import io,os,json,hashlib,zipfile
p=Path(PARAM['root']);assert os.getuid()==p.stat().st_uid==1045
complete=json.loads((p/'CPU-complete.json').read_text())
assert complete['returncode']==0 and complete['cap_seconds']==60 and complete['term_grace_seconds']==5
assert complete['plan_sha256']==PARAM['plan_sha256'] and complete['source_sha256']==PARAM['inspect_sha256']
assert complete['controller_sha256']==PARAM['controller_sha256']
names=['native-result.json','native-row-audit.json','CPU-complete.json'];meta={};buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
 for name in names:
  b=(p/name).read_bytes();assert len(b)<=200000
  meta[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
  info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16
  archive.writestr(info,b)
payload=buffer.getvalue();assert len(payload)<=1048576
z=p/'CPU-readback.zip'
if z.exists():assert z.read_bytes()==payload
else:
 with z.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
print(json.dumps({'status':'COMPLETE_CPU_ORIGINAL_BYTES_READY','files':meta,'zip':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()},'chunk_bytes':49152}))
'''

READ_PART = r'''
from pathlib import Path
import os,json,hashlib,base64
p=Path(PARAM['root']);assert os.getuid()==p.stat().st_uid==1045
b=(p/'CPU-readback.zip').read_bytes();assert len(b)==PARAM['zip']['bytes'] and hashlib.sha256(b).hexdigest()==PARAM['zip']['sha256']
offset=PARAM['offset'];size=PARAM['size'];assert 0<=offset<len(b) and 0<size<=49152
part=b[offset:offset+size]
print(json.dumps({'offset':offset,'bytes':len(part),'sha256':hashlib.sha256(part).hexdigest(),'base64':base64.b64encode(part).decode()}))
'''


def sha(data): return hashlib.sha256(data).hexdigest()
def fsync_parent(path):
    fd=os.open(str(Path(path).parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def exact_write(path,data):
    path=Path(path)
    if path.exists():
        assert path.read_bytes()==data, 'Existing recovery artifact differs: '+str(path)
        return
    with path.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
    fsync_parent(path)
def save(name,value):
    with (F/name).open('x') as out:json.dump(value,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
    fsync_parent(F/name)
def unpack(result):
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])
def script(body,parameters):
    value='PARAM='+repr(parameters)+'\n'+body
    compile(value,'frozen_remote_CPU_control','exec')
    return value


def client():
    sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
    from mcp_client import Client
    instance=Client();instance.initialize();return instance


def command(connection,kind,body,parameters,seconds):
    code=script(body,parameters)
    filename='remote-'+kind+'-'+str(time.time_ns())+'.py'
    exact_write(F/filename,code.encode())
    result=connection.request('tools/call',{'name':'run_command','arguments':
        dict(mount_id=MOUNT,command='python3 -c '+shlex.quote(code),timeout_seconds=seconds,env=ENV)})
    save(kind+'-'+str(time.time_ns())+'-tool.json',result)
    assert not result.get('isError'), 'UNKNOWN remote outcome; preserve original intent and inspect, never rerun start'
    parsed=unpack(result)
    assert parsed['exit_code']==0, 'Remote control exited nonzero; inspect the original namespace'
    return json.loads(parsed['stdout'])


def settings():
    manifest=json.loads((F/'deployment-manifest.json').read_text())
    assert sha(Path(__file__).read_bytes())==manifest['controller_sha256']
    assert manifest['mount_id']==MOUNT and manifest['remote_relative']==REL and manifest['remote_uid']==1045
    assert manifest['CPU_seconds']==60 and manifest['term_grace_seconds']==5 and manifest['MCP_command_seconds']==75
    assert manifest['files']['cpu-plan.json']['sha256']==PLAN_SHA and manifest['files']['native_cpu.py']['sha256']==INSPECT_SHA
    assert len(manifest['files'])==5
    for name,record in manifest['files'].items():
        data=(F/name).read_bytes();assert len(data)==record['bytes'] and sha(data)==record['sha256']
    return manifest


def collect(connection):
    params={'root':REMOTE,'plan_sha256':PLAN_SHA,'inspect_sha256':INSPECT_SHA,'controller_sha256':sha(Path(__file__).read_bytes())}
    receipt=command(connection,'collect',COLLECT,params,30)
    save('readback-ready-'+str(time.time_ns())+'.json',receipt)
    pieces=[]
    for offset in range(0,receipt['zip']['bytes'],READBACK_CHUNK):
        part=command(connection,'readback-part',READ_PART,dict(params,zip=receipt['zip'],offset=offset,size=READBACK_CHUNK),30)
        data=base64.b64decode(part['base64'],validate=True)
        assert part['offset']==offset and len(data)==part['bytes'] and sha(data)==part['sha256']
        pieces.append(data)
    data=b''.join(pieces)
    assert len(data)==receipt['zip']['bytes'] and sha(data)==receipt['zip']['sha256']
    exact_write(F/'CPU-readback.zip',data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        assert len(names)==3 and set(names)=={'native-result.json','native-row-audit.json','CPU-complete.json'}
        for name in names:
            value=archive.read(name);expected=receipt['files'][name]
            assert len(value)==expected['bytes'] and sha(value)==expected['sha256']
            exact_write(F/name,value)
    result=json.loads((F/'native-result.json').read_text());audit=json.loads((F/'native-row-audit.json').read_text())
    assert result['plan_sha256']==PLAN_SHA and result['source_sha256']==INSPECT_SHA
    assert result['status']=='PASS_NATIVE_CPU_TRAIN124_FULL_REPORT_COST_AND_MASK_ONLY_NO_TRAINING'
    assert result['rows']==124 and result['training_documents']==32 and len(audit)==124
    assert result['original']['total']==402594 and result['original']['maximum']==5523
    assert result['original']['supervised_tokens']==result['expanded']['supervised_tokens']==7170
    assert result['original_encoded_sha256']=='7fa9ef2b46a5600497887b594d44474d27aaa4165056650958626b454641bcb3'
    assert all(a['assistant_supervised_IDs_exact'] and a['native_Auto_template_and_decode_exact'] for a in audit)
    assert len({a['row_id'] for a in audit})==124 and {a['index'] for a in audit}==set(range(124))
    assert result['all_assistant_supervision_IDs_unchanged'] and result['all124_published_native_Auto_template_decode_exact']
    assert result['validation_gold_read'] is False and result['model_loaded'] is False and result['training_performed'] is False and result['cuda_initialized'] is False and result['truncation'] is False
    assert result['runtime']['torch']=='2.7.1+cu126' and result['runtime']['transformers']=='5.20.2' and result['runtime']['tokenizers']=='0.23.1'
    exact_write(F/'actual-result.json',(F/'native-result.json').read_bytes())
    save('readback-verified-'+str(time.time_ns())+'.json',dict(status='ORIGINAL_RESULT_AND_ROW_AUDIT_BYTES_SHA_VERIFIED',files=receipt['files'],zip=receipt['zip'],no_rerun=True))
    return result


def run():
    assert not (F/'deployment-intent.json').exists(), 'Existing deployment intent: use inspect/collect, never repeat run'
    assert not (F/'CPU-start-intent.json').exists() and not (F/'native-result.json').exists()
    manifest=settings();payload=(F/'source-inputs.tar.gz').read_bytes()
    assert len(payload)==manifest['bundle']['bytes'] and sha(payload)==manifest['bundle']['sha256']
    intent=dict(at_epoch=time.time(),manifest_sha256=sha((F/'deployment-manifest.json').read_bytes()),
                controller_sha256=sha(Path(__file__).read_bytes()),target=REL,mount_id=MOUNT,only_CPU_tokenizer=True)
    save('deployment-intent.json',intent)
    connection=client()
    remote=command(connection,'init',INIT,dict(root=REMOTE,intent=intent),30)
    save('remote-directory.json',remote)
    for index,expected in enumerate(manifest['parts']):
        part=payload[index*UPLOAD_CHUNK:(index+1)*UPLOAD_CHUNK]
        assert len(part)==expected['bytes'] and sha(part)==expected['sha256']
        response=connection.request('tools/call',{'name':'write_file','arguments':dict(mount_id=MOUNT,
            path=REL+'/'+expected['name'],content=base64.b64encode(part).decode(),encoding='base64',create_parents=False)})
        save('upload-%03d-tool.json'%index,response)
        assert not response.get('isError'), 'UNKNOWN upload outcome: keep intent and inspect, do not rerun run'
    deployment=command(connection,'deploy',DEPLOY,dict(root=REMOTE,manifest=manifest),30)
    save('deployment-result.json',deployment)
    cpu_intent=dict(at_epoch=time.time(),cap_seconds=60,term_grace_seconds=5,plan_sha256=PLAN_SHA,controller_sha256=intent['controller_sha256'])
    save('CPU-start-intent.json',cpu_intent)
    wrapper=command(connection,'CPU',CPU,dict(root=REMOTE,manifest=manifest,python=PYTHON,plan_sha256=PLAN_SHA,
                                           controller_sha256=intent['controller_sha256']),75)
    save('CPU-command-result.json',wrapper)
    assert wrapper['returncode']==0, 'Original CPU failed/timed out; preserve evidence, no repeat'
    return collect(connection)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['run','inspect','collect'],required=True)
    mode=parser.parse_args().mode
    if mode=='run':result=run()
    elif mode=='inspect':
        result=command(client(),'inspect',INSPECT,{'root':REMOTE},30)
        save('inspection-'+str(time.time_ns())+'.json',result)
    else:
        assert (F/'CPU-start-intent.json').exists(), 'No local original CPU intent; inspect and reconcile identity first'
        settings();result=collect(client())
    print(json.dumps(result))


if __name__=='__main__':main()
