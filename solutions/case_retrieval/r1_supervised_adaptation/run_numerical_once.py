"""Unique MCP deployment/start, or read-only collection of the independent R1 numerical diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(v):return v.get('structuredContent') or json.loads(v['content'][0]['text'])
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['start','collect']);ap.add_argument('--directory',required=True);a=ap.parse_args()
    d=Path(a.directory);dep=json.loads((d/'deployment.json').read_text())
    sys.path.insert(0,dep['mcp_scripts']);from mcp_client import Client
    c=Client();c.initialize()
    def call(name,**kw):return c.request('tools/call',{'name':name,'arguments':dict(mount_id=dep['mount_id'],**kw)})
    if a.mode=='start':
        save(d/'execution-intent.json',dict(at_epoch=time.time(),controller_sha256=sha(__file__),deployment_sha256=sha(d/'deployment.json'),authorization='One GPU1 R1 numerical diagnostic180+TERM30; no optimizer/quality'))
        for name,ref in dep['upload_files'].items():
            path=Path(ref['path']);assert sha(path)==ref['sha256'] and path.stat().st_size==ref['bytes']
            response=call('write_file',path=dep['remote_relative']+'/'+name,content=path.read_text(),encoding='utf8',create_parents=True)
            save(d/(name+'.upload.json'),response);assert not response.get('isError')
        expected={n:r['sha256'] for n,r in dep['upload_files'].items()}
        code='from pathlib import Path;import hashlib;p=Path('+repr(dep['remote_absolute'])+');e='+repr(expected)+';assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in e.items());assert not (p/"launch-started.json").exists();assert not (p/"actual").exists()'
        response=call('run_command',command='python3 -c '+shlex.quote(code),timeout_seconds=15)
        save(d/'deployment-verification.json',response);assert unpack(response)['exit_code']==0
        command=shlex.quote(dep['python'])+' -u '+shlex.quote(dep['remote_absolute']+'/launch_probe.py')+' --plan '+shlex.quote(dep['remote_absolute']+'/plan.json')
        save(d/'command.json',dict(command=command,environment=dep['environment']))
        response=call('start_process',command=command,env=dep['environment']);save(d/'managed-process.json',response);assert not response.get('isError')
        print(json.dumps(unpack(response)))
    else:
        response=call('read_file',path=dep['remote_relative']+'/terminal.json',offset=0,limit=1000000)
        if response.get('isError'):raise RuntimeError('No terminal yet; preserve running job')
        item=unpack(response);data=item['content'].encode();assert sha256(data)==item['sha256'];terminal=json.loads(data)
        recovered=d/'recovered';recovered.mkdir(exist_ok=False);(recovered/'terminal.json').write_bytes(data)
        for name,ref in terminal['files'].items():
            assert '..' not in Path(name).parts
            if name.endswith('.pt'):continue
            assert ref['bytes']<1000000
            response=call('read_file',path=dep['remote_relative']+'/'+name,offset=0,limit=1000000);item=unpack(response);data=item['content'].encode()
            assert len(data)==ref['bytes'] and sha256(data)==ref['sha256']==item['sha256']
            p=recovered/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        save(d/'collection.json',dict(status='ALL_SMALL_DIAGNOSTIC_FILES_RESTORED_SIZE_AND_SHA_TENSORS_REMAIN_ON_SERVER',exit_code=terminal['exit_code'],terminal_sha256=sha(recovered/'terminal.json'),members=len(terminal['files']),remote_tensor_members={n:r for n,r in terminal['files'].items() if n.endswith('.pt')},wall_seconds=terminal['wall_seconds']))
        print((d/'collection.json').read_text())


def sha256(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':main()
