#!/usr/bin/env python3
"""Own-directory deployment/managed run/collection; never implicit resume."""
from pathlib import Path
import argparse,base64,hashlib,json,shlex,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['deploy','start','collect']);ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--checkpoint');ap.add_argument('--release-evidence',type=Path);a=ap.parse_args()
    b=a.directory;p=json.loads((b/'plan.json').read_text());remote=p['remote']
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(name,args):
        r=client.request('tools/call',dict(name=name,arguments=dict(mount_id=p['mount'],**args)));assert not r.get('isError'),r
        return r['structuredContent']
    def save(name,x):
        with (b/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
    if a.mode=='deploy':
        code='from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();(p/"legacy").mkdir(parents=True)'
        r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10));assert r['exit_code']==0
        for name,digest in p['files'].items():
            raw=(b/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
            parts=[]
            for i,start in enumerate(range(0,len(raw),512*1024)):
                part=remote+'/'+name+f'.part{i:03d}';parts.append(part)
                call('write_file',dict(path=part,encoding='base64',content=base64.b64encode(raw[start:start+512*1024]).decode()))
            code='from pathlib import Path;import hashlib;parts='+repr(parts)+';v=b"".join(Path(x).read_bytes() for x in parts);assert hashlib.sha256(v).hexdigest()=='+repr(digest)+';Path('+repr(remote+'/'+name)+').write_bytes(v);[Path(x).unlink() for x in parts]'
            r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10));assert r['exit_code']==0
        call('write_file',dict(path=remote+'/plan.json',encoding='utf8',content=(b/'plan.json').read_text()))
        save('deployment.json',dict(files=p['files'],plan_sha256=hashlib.sha256((b/'plan.json').read_bytes()).hexdigest()))
    elif a.mode=='start':
        assert a.checkpoint and len(a.checkpoint)==40 and a.release_evidence and a.release_evidence.exists()
        assert not (b/'job.json').exists() and (b/'deployment.json').exists() and (b/'adapter-transfer.json').exists()
        code='''import json,subprocess,shutil,hashlib,datetime
from pathlib import Path
p=json.loads(Path('plan.json').read_text());assert hashlib.sha256(Path('plan.json').read_bytes()).hexdigest()==PLAN_SHA
assert not Path('responses.jsonl').exists()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
target=[x for x in gpu.splitlines() if p['gpu_uuid'] in x];assert len(target)==1 and p['gpu_uuid'] not in apps
assert int(target[0].split(',')[1])<512 and shutil.disk_usage('.').free>2*1024**3
for n,h in p['adapter_files'].items():assert hashlib.sha256((Path(p['adapter'])/n).read_bytes()).hexdigest()==h
print(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_empty=True,gpu=target[0],free_bytes=shutil.disk_usage('.').free)))
'''.replace('PLAN_SHA',repr(hashlib.sha256((b/'plan.json').read_bytes()).hexdigest()))
        r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=20));assert r['exit_code']==0;save('fresh-GPU.json',r)
        args=dict(command=shlex.join(['timeout','--signal=TERM','--kill-after=30s','600s',p['python'],'launch.py']),cwd=remote,env=p['env'])
        result=call('start_process',args);save('job.json',dict(arguments=args,result=result,checkpoint=a.checkpoint,
            release_evidence=dict(path=str(a.release_evidence),sha256=hashlib.sha256(a.release_evidence.read_bytes()).hexdigest()),cap=600,grace=30))
        print(json.dumps(result))
    else:
        job=json.loads((b/'job.json').read_text());records={s:call('read_process_output',dict(process_id=job['result']['id'],stream=s,cursor=0,limit=30000)) for s in ['stdout','stderr']}
        if any(x['running'] for x in records.values()):print(json.dumps(dict(running=True,output={k:v['content'][-2000:] for k,v in records.items()})));return
        save('process-terminal.json',records)
        code='''from pathlib import Path
import json,subprocess,hashlib,datetime
p=json.loads(Path('plan.json').read_text());apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);assert p['gpu_uuid'] not in apps
Path('gpu-release.json').write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=p['gpu_uuid'],target_empty=True)))
files=[]
for name in ['actual-preflight.json','runner-exit.json','responses.jsonl','responses.manifest.json','responses.checkpoint.json','responses.loading.json','complete.json','gpu-release.json']:
 x=Path(name)
 if x.exists():files.append(dict(name=name,bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()))
print(json.dumps(files))
'''
        r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=15));assert r['exit_code']==0
        items=json.loads(r['stdout']);out=b/'terminal';out.mkdir(exist_ok=False)
        for it in items:
            chunks=[]
            for offset in range(0,it['bytes'],512*1024):
                x=call('read_file',dict(path=remote+'/'+it['name'],offset=offset,limit=min(512*1024,it['bytes']-offset)))
                chunks.append(base64.b64decode(x['content']) if x['encoding']=='base64' else x['content'].encode())
            raw=b''.join(chunks);assert len(raw)==it['bytes'] and hashlib.sha256(raw).hexdigest()==it['sha256'];(out/it['name']).write_bytes(raw)
        save('collection.json',dict(items=items,complete=(out/'complete.json').exists()))
        print(json.dumps(dict(running=False,complete=(out/'complete.json').exists(),items=len(items))))

if __name__=='__main__':main()
