"""Deploy/start one immutable arm or recover one terminal archive; never auto-retry."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile
import time
import zipfile


def sha(data): return hashlib.sha256(data).hexdigest()
def unpack(v): return v.get('structuredContent') or json.loads(v['content'][0]['text'])
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode', choices=['deploy','start','collect'])
    ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();d=a.directory.resolve()
    dep=json.loads((d/'deployment.json').read_text());plan=json.loads((d/'plan.json').read_text())
    sys.path.insert(0,dep['mcp_scripts']);from mcp_client import Client
    client=Client();client.initialize()
    def call(name,**args):
        value=client.request('tools/call',{'name':name,'arguments':dict(mount_id=dep['mount_id'],**args)})
        if value.get('isError'): raise RuntimeError('MCP '+name+' returned error; preserve stage, no retry')
        return value
    remote=dep['remote_relative'];absolute=dep['remote_absolute']
    if a.mode=='deploy':
        save(d/'deployment-intent.json',dict(at_epoch=time.time(),source_sha256=sha(Path(__file__).read_bytes())))
        names=list(plan['local_files'])+['plan.json'];assert len(set(names))==len(names)
        payload=io.BytesIO()
        with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name in names:
                p=d/name;assert p.is_file() and '/' not in name and not p.is_symlink()
                if name in plan['local_files']:
                    r=plan['local_files'][name];assert r['bytes']==p.stat().st_size and r['sha256']==sha(p.read_bytes())
                z.writestr(name,p.read_bytes())
        data=payload.getvalue();(d/'deployment.zip').write_bytes(data)
        parts=[data[i:i+240000] for i in range(0,len(data),240000)]
        for i,part in enumerate(parts):
            response=call('write_file',path=remote+'/deployment.chunk%04d'%i,content=base64.b64encode(part).decode(),encoding='base64',create_parents=True)
            save(d/('deployment-chunk%04d.json'%i),response)
        code='''from pathlib import Path
import json,hashlib,io,zipfile,os
p=Path(ABSOLUTE);data=b''.join((p/('deployment.chunk%04d'%i)).read_bytes() for i in range(COUNT))
assert hashlib.sha256(data).hexdigest()==ZIP_SHA
z=zipfile.ZipFile(io.BytesIO(data));assert z.namelist()==NAMES
for info in z.infolist():
 target=p/info.filename;assert '/' not in info.filename and not target.exists();b=z.read(info.filename)
 with target.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
plan=json.loads((p/'plan.json').read_text())
for n,r in plan['local_files'].items():
 q=p/n;assert q.stat().st_size==r['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='DEPLOYED_SHA_VERIFIED',zip_sha256=ZIP_SHA,files=len(z.namelist()))))'''.replace('ABSOLUTE',repr(absolute)).replace('COUNT',str(len(parts))).replace('ZIP_SHA',repr(sha(data))).replace('NAMES',repr(names))
        result=call('run_command',command='python3 -c '+shlex.quote(code),timeout_seconds=30);save(d/'deployment-verification.json',result)
        assert unpack(result)['exit_code']==0;print(json.dumps(unpack(result)))
    elif a.mode=='start':
        assert unpack(json.loads((d/'deployment-verification.json').read_text()))['exit_code']==0
        save(d/'execution-intent.json',dict(at_epoch=time.time(),plan_sha256=sha((d/'plan.json').read_bytes()),single_run=True))
        command=shlex.quote(plan['python'])+' -u '+shlex.quote(absolute+'/launch_arm.py')+' --plan '+shlex.quote(absolute+'/plan.json')
        save(d/'command.json',dict(command=command,environment=dep['environment']))
        result=call('start_process',command=command,env=dep['environment']);save(d/'managed-process.json',result)
        print(json.dumps(unpack(result)))
    else:
        assert not (d/'recovered').exists()
        save(d/'collection-intent.json',dict(at_epoch=time.time(),scope='Only terminal archive; no new inference or restart'))
        code='''from pathlib import Path
import json,hashlib,tarfile,os
p=Path(ABSOLUTE);t=p/'terminal.json';assert t.is_file();terminal=json.loads(t.read_text())
members=dict(terminal['files']);members['terminal.json']=dict(bytes=t.stat().st_size,sha256=hashlib.sha256(t.read_bytes()).hexdigest())
a=p/'terminal-artifacts.tar.gz';assert not a.exists()
with tarfile.open(a,'w:gz') as archive:
 for n,r in members.items():
  q=p/n;assert q.is_file() and not q.is_symlink() and '..' not in Path(n).parts
  assert q.stat().st_size==r['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
  archive.add(q,arcname=n,recursive=False)
with a.open('rb') as f:os.fsync(f.fileno())
fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
metadata=dict(status='FROZEN_TERMINAL_ARCHIVE',bytes=a.stat().st_size,sha256=hashlib.sha256(a.read_bytes()).hexdigest(),members=members,terminal_exit_code=terminal['exit_code'])
manifest=p/'terminal-artifacts.manifest.json'
with manifest.open('x') as f:json.dump(metadata,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(bytes=manifest.stat().st_size,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))'''.replace('ABSOLUTE',repr(absolute))
        result=call('run_command',command='python3 -c '+shlex.quote(code),timeout_seconds=45);save(d/'archive-command.json',result)
        terminal_result=unpack(result);assert terminal_result['exit_code']==0
        manifest_ref=json.loads(terminal_result['stdout'])
        manifest_result=unpack(call('read_file',path=remote+'/terminal-artifacts.manifest.json',offset=0,limit=1000000))
        manifest_bytes=manifest_result['content'].encode();assert len(manifest_bytes)==manifest_ref['bytes'] and sha(manifest_bytes)==manifest_ref['sha256']
        metadata=json.loads(manifest_bytes);save(d/'archive-manifest.json',metadata)
        data=bytearray()
        for offset in range(0,metadata['bytes'],500000):
            response=unpack(call('read_file',path=remote+'/terminal-artifacts.tar.gz',offset=offset,limit=min(500000,metadata['bytes']-offset)))
            chunk=base64.b64decode(response['content']) if response.get('encoding')=='base64' else response['content'].encode()
            assert len(chunk)==min(500000,metadata['bytes']-offset)
            data.extend(chunk)
        assert len(data)==metadata['bytes'] and sha(data)==metadata['sha256']
        (d/'terminal-artifacts.tar.gz').write_bytes(data)
        recovered=d/'recovered';recovered.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            members=archive.getmembers();assert len({m.name for m in members})==len(members)==len(metadata['members'])
            assert {m.name for m in members}==set(metadata['members'])
            for m in members:
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                b=archive.extractfile(m).read();r=metadata['members'][m.name]
                assert len(b)==r['bytes'] and sha(b)==r['sha256']
                p=recovered/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        save(d/'collection.json',dict(status='ALL_TERMINAL_MEMBERS_RESTORED_EXACT_SIZE_SHA',archive_sha256=sha(data),members=len(metadata['members']),exit_code=metadata['terminal_exit_code']))
        print((d/'collection.json').read_text())


if __name__=='__main__':main()
