"""Owned 8B campaign transport; identities and unique starts survive disconnects."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import time
import zipfile
from common import ref,write

P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','start','inspect']);p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.package);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((source/'candidate-plan.json').read_text());root=str(Path(plan['run_root']).parent)
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    c=Client();c.initialize()
    def call(tool,args,tag):
        prefix=out/f'{time.time_ns()}-{tag}';safe=dict(args)
        if 'content' in safe:safe['content']='[frozen source archive bytes omitted]'
        write(str(prefix)+'.intent.json',dict(tool=tool,args=safe,source=ref(__file__)))
        try:r=c.request('tools/call',dict(name=tool,arguments=args))
        except Exception as exc:
            write(str(prefix)+'.UNKNOWN.json',dict(error_type=type(exc).__name__,no_automatic_retry=True));raise
        if r.get('isError'):
            write(str(prefix)+'.ERROR.json',r);raise RuntimeError('MCP error preserved; do not replay mutation; explicit LUNA requires185s')
        v=r.get('structuredContent')
        if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
        write(str(prefix)+'.response.json',v);return v
    def command(code,tag):
        v=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=45),tag)
        assert v['exit_code']==0,(tag,v.get('stderr'))
        return json.loads(v['stdout'])
    if a.stage=='prepare':
        assert not (out/'prepared.json').exists() and not (out/'start-intent.json').exists()
        expected={}
        for phase in ['train','held']:
            raw=gzip.decompress((source/(phase+'.encoded.jsonl.gz')).read_bytes())
            expected[phase]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        prior=P+'/research/canonical-names-v1/qwen3-native-cpu-v1'
        # Reconstruct only the container/identity metadata from the already
        # verified actual encoded IDs. Never retokenize or read new labels.
        spec=dict(root=root,prior=prior,expected=expected,original_encoded=plan['cpu_result'])
        derived=command('SPEC='+repr(spec)+'\n'+DERIVE,'derive-same-IDs')
        cpu=json.loads((source/'cpu.json').read_text())
        for phase in ['train','held']:
            assert derived[phase]['raw']==expected[phase]
            cpu['counts'][phase]['encoded']=derived[phase]['encoded']
            plan['counts'][phase]['encoded']=derived[phase]['encoded']
        cpu_bytes=(json.dumps(cpu,ensure_ascii=False,indent=2)+'\n').encode()
        plan['cpu_result']=dict(path=root+'/data/cpu.json',bytes=len(cpu_bytes),sha256=hashlib.sha256(cpu_bytes).hexdigest())
        write(out/'cpu-actual.json',cpu);write(out/'candidate-plan-actual.json',plan)
        payload={'data/cpu.json':cpu_bytes,'candidate-plan.json':(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode()}
        own=Path(__file__).resolve().parent
        backend=Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations/full64_component')
        for label,item in plan['source_refs'].items():
            src=(own if label.startswith('own/') else backend)/label.split('/')[-1]
            local=ref(src);assert local['bytes']==item['bytes'] and local['sha256']==item['sha256']
            payload[str(Path(item['path']).relative_to(root))]=src.read_bytes()
        buff=io.BytesIO()
        with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as z:
            for n,b in payload.items():z.writestr(n,b)
        archive=buff.getvalue();(out/'source-package.zip').write_bytes(archive)
        members={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in payload.items()};chunks=[]
        for pos in range(0,len(archive),32768):
            block=archive[pos:pos+32768];name=f'source-{pos:08d}.chunk'
            call('write_file',dict(mount_id=MOUNT,path=root[len(P)+1:]+'/'+name,content=base64.b64encode(block).decode(),encoding='base64',create_parents=True),'upload-source')
            chunks.append(dict(name=name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
        spec=dict(root=root,chunks=chunks,archive_sha256=hashlib.sha256(archive).hexdigest(),members=members)
        ack=command('SPEC='+repr(spec)+'\n'+DEPLOY,'deploy-source')
        assert ack['members']==members
        prepared=dict(status='DEPLOYED_NOT_STARTED',remote=root,plan=ref(out/'candidate-plan-actual.json'),cpu=ref(out/'cpu-actual.json'),
            members=members,derived=derived,remote_ack=ack,archive=ref(out/'source-package.zip'))
        print(json.dumps(write(out/'prepared.json',prepared)),flush=True)
    elif a.stage=='start':
        assert (out/'prepared.json').exists() and not (out/'start-intent.json').exists()
        actual=json.loads((out/'candidate-plan-actual.json').read_text());prepared=json.loads((out/'prepared.json').read_text())
        assert ref(out/'candidate-plan-actual.json')==prepared['plan']
        command('ROOT='+repr(root)+'\n'+PRESTART,'prestart-owned-inventory')
        args=dict(mount_id=MOUNT,cwd=root[len(P)+1:],command=shlex.quote(actual['python'])+' source/phase_driver.py --plan candidate-plan.json',
            env={**actual['environment'],'CUDA_VISIBLE_DEVICES':actual['gpu_uuid']})
        write(out/'start-intent.json',dict(at_unix=time.time(),args=args,plan=prepared['plan'],no_automatic_retry=True))
        result=call('start_process',args,'start-unique-campaign')
        print(json.dumps(write(out/'managed-start.json',result)),flush=True)
    else:
        value=command('ROOT='+repr(root)+'\n'+INSPECT,'inspect-original')
        path=out/f'observation-{time.time_ns()}.json';print(json.dumps(write(path,value)),flush=True)

DERIVE=r'''
import gzip,json,hashlib,os
from pathlib import Path
root=Path(SPEC['root']);prior=Path(SPEC['prior']);assert not (root/'run').exists()
data=root/'data';data.mkdir(parents=True,exist_ok=True)
def identity(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
payload=json.loads(gzip.decompress((prior/'actual/native/encoded.json.gz').read_bytes()))
result={}
for phase,inputname in [('train','train'),('held','held_input')]:
 targets={r['row_id']:r for r in map(json.loads,(prior/'actual/inputs'/f'{inputname}.jsonl').read_text().splitlines())}
 rows=[{**r,'pmc_id':targets[r['row_id']]['pmc_id'],'target_patient_id':targets[r['row_id']]['target_patient_id']} for r in payload[phase]]
 raw=b''.join((json.dumps(r,separators=(',',':'))+'\n').encode() for r in rows);assert identity(raw)==SPEC['expected'][phase]
 path=data/f'{phase}.encoded.jsonl.gz';assert not path.exists();packed=gzip.compress(raw,mtime=0)
 with path.open('xb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
 assert gzip.decompress(path.read_bytes())==raw
 result[phase]=dict(raw=identity(raw),encoded=dict(path=str(path),**identity(path.read_bytes())))
fd=os.open(data,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(result))
'''

DEPLOY=r'''
import os,json,hashlib,zipfile,io
from pathlib import Path,PurePosixPath
root=Path(SPEC['root']);assert not (root/'run').exists()
def identity(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
data=b''
for item in SPEC['chunks']:
 b=(root/item['name']).read_bytes();assert identity(b)=={k:item[k] for k in ['bytes','sha256']};data+=b
assert hashlib.sha256(data).hexdigest()==SPEC['archive_sha256']
with zipfile.ZipFile(io.BytesIO(data)) as z:
 assert set(z.namelist())==set(SPEC['members']) and len(z.namelist())==len(set(z.namelist()))
 for name,item in SPEC['members'].items():
  p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
  b=z.read(name);assert identity(b)==item;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert identity(dest.read_bytes())==item
for p in sorted([q for q in root.rglob('*') if q.is_dir()]+[root],key=lambda q:len(q.parts),reverse=True):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(members=SPEC['members'],all_SHA_fsync_reopen=True)))
'''

PRESTART=r'''
from pathlib import Path
import json
root=Path(ROOT);assert not (root/'run').exists();print(json.dumps(dict(run_absent=True,root=str(root))))
'''

INSPECT=r'''
from pathlib import Path
import json
root=Path(ROOT)/'run';result={'run_exists':root.exists(),'files':{}}
for name in ['driver-start.json','driver-result.json','closed-before-formal.json']:
 p=root/name
 if p.exists():result['files'][name]=json.loads(p.read_text())
for phase in ['probe','formal','infer']:
 for name in [f'{phase}-child-start.json',f'{phase}-child-result.json',f'{phase}/complete.json',f'{phase}/failure.json',f'{phase}/progress.json',f'{phase}/generation/progress.json']:
  p=root/name
  if p.exists():result['files'][name]=json.loads(p.read_text())
print(json.dumps(result))
'''

if __name__=='__main__':main()
