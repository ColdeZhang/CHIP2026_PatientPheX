#!/usr/bin/env python3
"""Bounded own-directory MCP deployment/start/collection, no automatic retry."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['deploy','start','collect'])
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--mcp-client-directory',type=Path,required=True)
    a=ap.parse_args();b=a.directory
    sys.path.insert(0,str(a.mcp_client_directory))
    from mcp_client import Client
    c=Client();c.initialize()
    p=json.loads((b/'plan.json').read_text());mount=p['mount'];remote=p['remote_directory']
    def call(name,args):
        r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=mount,**args)))
        assert not r.get('isError'),r
        return r['structuredContent']
    def save(name,value):
        target=b/name;assert not target.exists()
        target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if a.mode=='deploy':
        assert not (b/'deployment.json').exists()
        code="from pathlib import Path; p=Path("+repr(remote)+"); assert not p.exists(); p.mkdir(parents=True)"
        result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10))
        assert result['exit_code']==0,result
        receipts={}
        for name,digest in p['local_files'].items():
            value=(b/name).read_bytes()
            assert len(value)<=512*1024 and hashlib.sha256(value).hexdigest()==digest
            receipts[name]=call('write_file',dict(path=remote+'/'+name,encoding='base64',content=base64.b64encode(value).decode()))
        receipts['plan.json']=call('write_file',dict(path=remote+'/plan.json',encoding='utf8',content=(b/'plan.json').read_text()))
        code="from pathlib import Path; import json,hashlib; p=json.loads(Path('plan.json').read_text()); assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in p['local_files'].items()); [compile(Path(n).read_text(),n,'exec') for n in p['local_files'] if n.endswith('.py')]; print('source hash and actual Python syntax PASS')"
        result=call('run_command',dict(command=shlex.quote(p['python'])+' -c '+shlex.quote(code),cwd=remote,env=p['env'],timeout_seconds=15))
        assert result['exit_code']==0,result
        save('deployment.json',dict(receipts=receipts,actual_cpu_source_syntax=result))
        print(json.dumps(dict(deployed=True,files=len(receipts))))
    elif a.mode=='start':
        assert (b/'deployment.json').exists() and not (b/'job.json').exists()
        code="""import json,subprocess,shutil,datetime
target=GPU_UUID
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
row=[s for s in g.splitlines() if target in s];assert len(row)==1 and target not in apps
assert int(row[0].split(',')[2].strip())<=512
free=shutil.disk_usage('.').free;assert free>2*1024**3
print(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_row=row[0],target_empty=True,free_bytes=free)))
""".replace('GPU_UUID',repr(p['gpu_uuid']))
        result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=15))
        assert result['exit_code']==0,result
        save('fresh-GPU.json',result)
        argv=['timeout','--signal=TERM','--kill-after=5s','115s',p['python'],'probe_entities.py',
              '--directory','.', '--model',p['model']]
        args=dict(command=shlex.join(argv),cwd=remote,env=p['env'])
        result=call('start_process',args)
        save('job.json',dict(arguments=args,result=result,plan_sha256=hashlib.sha256((b/'plan.json').read_bytes()).hexdigest(),
                             training_authorized=False,total_probe_cap_seconds=120))
        print(json.dumps(result))
    else:
        job=json.loads((b/'job.json').read_text())
        output={s:call('read_process_output',dict(process_id=job['result']['id'],stream=s,cursor=0,limit=30000)) for s in ['stdout','stderr']}
        if any(x['running'] for x in output.values()):
            print(json.dumps(dict(running=True,output=output)));return
        save('process-terminal.json',output)
        code="""from pathlib import Path
import json,hashlib,subprocess,datetime
p=json.loads(Path('plan.json').read_text());target=p['gpu_uuid']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
assert target not in apps
Path('gpu-release.json').write_text(json.dumps(dict(target=target,target_empty=True,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\\n')
files=[]
for name in ['run_manifest.json','equivalence.json','shortest-step.json','longest-step.json','heldout-shape-only.json','complete.json','gpu-release.json']:
 f=Path(name)
 if f.exists():files.append(dict(name=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(files))
"""
        result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=15))
        assert result['exit_code']==0,result
        items=json.loads(result['stdout']);target=b/'terminal';target.mkdir(exist_ok=True)
        for item in items:
            reply=call('read_file',dict(path=remote+'/'+item['name'],offset=0,limit=item['bytes']))
            value=base64.b64decode(reply['content']) if reply['encoding']=='base64' else reply['content'].encode()
            assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
            (target/item['name']).write_bytes(value)
        save('collection.json',dict(items=items,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        print(json.dumps(dict(running=False,complete=(target/'complete.json').exists(),files=items)))


if __name__=='__main__': main()
