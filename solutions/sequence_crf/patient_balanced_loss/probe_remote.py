"""Deploy/observe one probe; UNKNOWN never retries a model start."""
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

MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
REL='sequence-round2/patient-balanced-alpha05/probe-v1'
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REL


def ref(p):
    p=Path(p).resolve();raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def put(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    parent=p.resolve().parent
    while True:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if parent.parent==parent:break
        parent=parent.parent


def save(p,x):put(p,(json.dumps(x,indent=2)+'\n').encode())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pack','deploy','start','observe'])
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--h5-release',type=Path)
    args=ap.parse_args();out=args.directory.resolve();here=Path(__file__).resolve().parent;os.umask(0o077)
    if args.stage=='pack':
        plan=json.loads((here/'probe-plan.json').read_text());sources=json.loads((here/'probe-package-inputs.json').read_text())
        for name,r in sources.items():
            assert ref(r['path'])==r and {k:r[k] for k in ['bytes','sha256']}==plan['remote_files'][name]
        assert plan['operations_source']==ref(__file__)
        items=dict(sources,**{'run-plan.json':ref(here/'probe-plan.json')})
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
            for n,r in sorted(items.items()):assert Path(n).name==n;z.writestr(n,Path(r['path']).read_bytes())
        out.mkdir(parents=True,exist_ok=False);put(out/'deployment.zip',raw.getvalue())
        save(out/'package.json',dict(archive=ref(out/'deployment.zip'),members=items,plan=ref(here/'probe-plan.json'),source=ref(__file__),remote=ROOT,mount=MOUNT))
        print(json.dumps(ref(out/'package.json')));return
    cfg=json.loads((out/'package.json').read_text());assert cfg['source']==ref(__file__)
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(tool,arguments,tag):
        p=out/'transport'/(tag+'-'+str(time.time_ns()));safe=dict(arguments)
        if 'content' in safe:safe['content']='[frozen bytes omitted]'
        save(p.with_suffix('.intent.json'),dict(tool=tool,arguments=safe))
        try:result=client.request('tools/call',dict(name=tool,arguments=arguments))
        except Exception as e:
            save(p.with_suffix('.error.json'),dict(UNKNOWN=True,error_type=type(e).__name__,automatic_retry=False));raise RuntimeError('UNKNOWN; inventory, never replay start') from None
        text=json.dumps(result)
        for key in ['MCP_TOKEN','MCP_URL']:
            if os.environ.get(key):text=text.replace(os.environ[key],'[REDACTED]')
        result=json.loads(text)
        if result.get('isError'):
            save(p.with_suffix('.error.json'),result);raise RuntimeError('Remote tool error; stop; LUNA requires unified >=180s backoff')
        actual=result.get('structuredContent')
        if actual is None:actual=json.loads(next(x['text'] for x in result['content'] if x['type']=='text'))
        save(p.with_suffix('.response.json'),actual);return actual
    def cmd(code,tag):
        r=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=40),tag)
        assert r['exit_code']==0;return json.loads(r['stdout'])
    if args.stage=='deploy':
        inventory=cmd('from pathlib import Path;import json,hashlib;p=Path('+repr(ROOT)+');print(json.dumps({str(q.relative_to(p)):dict(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in p.iterdir() if q.is_file()} if p.exists() else {}))','inventory')
        assert 'driver-started.json' not in inventory and 'started.json' not in inventory
        assert ref(out/'deployment.zip')==cfg['archive'];raw=(out/'deployment.zip').read_bytes()
        chunks=[]
        for offset in range(0,len(raw),32768):
            data=raw[offset:offset+32768];name='deployment-%06d.part'%offset
            r=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());chunks.append(dict(name=name,**r))
            if name in inventory:assert inventory[name]==r;continue
            got=call('write_file',dict(mount_id=MOUNT,path=REL+'/'+name,content=base64.b64encode(data).decode(),encoding='base64',create_parents=True),'part-'+str(offset))
            assert got['sha256']==r['sha256'] and got['size']==r['bytes']
        code='''from pathlib import Path
import json,hashlib,zipfile,io,os
root=Path(ROOT);cfg=CFG;chunks=CHUNKS;raw=b''
assert not (root/'driver-started.json').exists()
for item in chunks:
 b=(root/item['name']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'];raw+=b
assert len(raw)==cfg['archive']['bytes'] and hashlib.sha256(raw).hexdigest()==cfg['archive']['sha256']
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert set(z.namelist())==set(cfg['members'])
 for name,item in cfg['members'].items():
  b=z.read(name);assert Path(name).name==name and len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'];p=root/name
  if p.exists():assert p.read_bytes()==b
  else:
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  with p.open('rb') as f:os.fsync(f.fileno())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
  if name.endswith('.py'):compile(b,str(p),'exec')
p=root
while True:
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if p.parent==p:break
 p=p.parent
print(json.dumps(dict(members=len(cfg['members']),SHA_fsync_reopen=True,root=str(root),no_model_import=True)))
'''.replace('ROOT',repr(ROOT)).replace('CFG',repr(cfg)).replace('CHUNKS',repr(chunks))
        actual=cmd(code,'verify');save(out/'deployment.json',dict(actual=actual,package=ref(out/'package.json')))
    elif args.stage=='start':
        assert (out/'deployment.json').exists() and not (out/'start-intent.json').exists()
        assert args.h5_release and args.h5_release.is_file(),'Explicit completed H5 release receipt required before this allocated window'
        code='''from pathlib import Path
import json,os
root=Path(ROOT);matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
  names={'probe_driver.py','probe.py',str(root/'probe_driver.py'),str(root/'probe.py')}
  if any(a in names for a in argv) and (p/'cwd').resolve()==root:matches.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(driver_started=(root/'driver-started.json').exists(),started=(root/'started.json').exists(),matching_pids=matches)))
'''.replace('ROOT',repr(ROOT))
        inventory=cmd(code,'before-start');assert not inventory['driver_started'] and not inventory['started'] and not inventory['matching_pids']
        plan=json.loads((here/'probe-plan.json').read_text());env=plan['environment']
        command=shlex.join([plan['python'],'-u','probe_driver.py'])
        save(out/'start-intent.json',dict(command=command,cwd=REL,environment=env,h5_release=ref(args.h5_release),inventory=inventory,no_retry=True))
        actual=call('start_process',dict(mount_id=MOUNT,cwd=REL,command=command,env=env),'start');save(out/'started.json',actual)
    elif args.stage=='observe':
        code='''from pathlib import Path
import json
p=Path(ROOT);names=['driver-started.json','child-started.json','wrapper-complete.json','complete.json','failure.json'];stages=sorted(p.glob('stage-*-after.json'))
if stages:names.append(stages[-1].name)
print(json.dumps({n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}))
'''.replace('ROOT',repr(ROOT))
        actual=cmd(code,'observe');save(out/('observation-'+str(time.time_ns())+'.json'),actual);print(json.dumps(actual))
    else:
        raise RuntimeError('Collect only completed original output files via bounded small reads; no model replay or broad command output')
    print(json.dumps(dict(stage=args.stage,complete=True,directory=str(out))))


if __name__=='__main__':main()
