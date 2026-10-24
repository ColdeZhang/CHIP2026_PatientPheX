#!/usr/bin/env python3
"""Own-directory bounded deployment/probe; never an implicit retry or training run."""
import argparse,base64,datetime,hashlib,json,shlex,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cpu','start','collect']);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--checkpoint');a=ap.parse_args();b=a.directory
    resource=json.loads((b/'resource.json').read_text());remote=resource['remote']
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(name,**args):
        r=client.request('tools/call',dict(name=name,arguments=dict(mount_id=resource['mount'],**args)));assert not r.get('isError'),r
        return r['structuredContent']
    def save(name,x):
        with (b/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
    def run(code,timeout=30,env=None):return call('run_command',command=resource['python']+' -c '+shlex.quote(code),cwd=remote,env=env or resource['CPU_env'],timeout_seconds=timeout)
    def read(name):
        x=call('read_file',path=remote+'/'+name,offset=0,limit=512*1024)
        return base64.b64decode(x['content']) if x['encoding']=='base64' else x['content'].encode()
    if a.mode=='cpu':
        assert not (b/'cpu-job.json').exists()
        prep=json.loads((b/'preparation.json').read_text())
        expected={name:r['sha256'] for name,r in prep['files'].items()}
        code='from pathlib import Path;import hashlib;p=Path('+repr(remote)+');expected='+repr(expected)+';p.mkdir(parents=True,exist_ok=True);assert all(x.is_file() and x.name in expected and hashlib.sha256(x.read_bytes()).hexdigest()==expected[x.name] for x in p.iterdir())'
        r=call('run_command',command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10);assert r['exit_code']==0
        for name,ref in prep['files'].items():
            raw=(b/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'] and len(raw)==ref['bytes']
            parts=[]
            for i,start in enumerate(range(0,len(raw),256*1024)):
                part=name+f'.part{i:03d}';parts.append(part)
                call('write_file',path=remote+'/'+part,encoding='base64',content=base64.b64encode(raw[start:start+256*1024]).decode())
            code='from pathlib import Path;import hashlib;parts='+repr(parts)+';raw=b"".join(Path(x).read_bytes() for x in parts);assert hashlib.sha256(raw).hexdigest()=='+repr(ref['sha256'])+';Path('+repr(name)+').write_bytes(raw);[Path(x).unlink() for x in parts]'
            r=run(code);assert r['exit_code']==0
        args=dict(command=shlex.join([resource['python'],'torch_contracts.py','--output','cpu-result.json']),cwd=remote,env=resource['CPU_env'],timeout_seconds=30)
        r=call('run_command',**args);save('cpu-job.json',dict(arguments=args,result=r));assert r['exit_code']==0,r
        raw=read('cpu-result.json');(b/'cpu-result.json').write_bytes(raw);v=json.loads(raw);assert not v['CUDA_initialized'] and v['status']=='ACTUAL_CPU_AUTOGRAD_PASS'
        print(json.dumps(dict(status=v['status'],seconds=v['seconds'],sha256=hashlib.sha256(raw).hexdigest())))
    elif a.mode=='start':
        assert a.checkpoint and len(a.checkpoint)==40 and not (b/'job.json').exists()
        p=json.loads((b/'probe-plan.json').read_text());digest=hashlib.sha256((b/'probe-plan.json').read_bytes()).hexdigest()
        assert p['operations_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        call('write_file',path=remote+'/probe-plan.json',encoding='utf8',content=(b/'probe-plan.json').read_text())
        code='''import json,hashlib,subprocess,shutil,datetime
from pathlib import Path
p=json.loads(Path('probe-plan.json').read_text());assert hashlib.sha256(Path('probe-plan.json').read_bytes()).hexdigest()==PLAN
assert not Path('probe-result.json').exists() and not Path('probe-partial.json').exists()
for name,h in p['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
assert hashlib.sha256(Path('cpu-result.json').read_bytes()).hexdigest()==p['cpu_result_sha256']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
target=[x for x in gpu.splitlines() if p['gpu_uuid'] in x];assert len(target)==1 and p['gpu_uuid'] not in apps and int(target[0].split(',')[1])<512
assert shutil.disk_usage('.').free>2*1024**3
print(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_empty=True,gpu=target[0],free_bytes=shutil.disk_usage('.').free,plan_sha256=PLAN)))
'''.replace('PLAN',repr(digest))
        r=run(code);assert r['exit_code']==0,r;save('fresh-GPU.json',r)
        args=dict(command=shlex.join(['timeout','--signal=TERM','--kill-after=15s','180s',resource['python'],'probe.py']),cwd=remote,env=resource['GPU_env'])
        r=call('start_process',**args);save('job.json',dict(arguments=args,result=r,checkpoint=a.checkpoint,plan_sha256=digest,cap=180,grace=15));print(json.dumps(r))
    else:
        job=json.loads((b/'job.json').read_text());records={s:call('read_process_output',process_id=job['result']['id'],stream=s,cursor=0,limit=30000) for s in ['stdout','stderr']}
        if any(x['running'] for x in records.values()):print(json.dumps(dict(running=True,output={k:v['content'][-1600:] for k,v in records.items()})));return
        save('process-terminal.json',records)
        code='''from pathlib import Path
import json,hashlib,subprocess,datetime
p=json.loads(Path('probe-plan.json').read_text());apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);assert p['gpu_uuid'] not in apps
Path('gpu-release.json').write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=p['gpu_uuid'],target_empty=True)))
print(json.dumps([dict(name=n,bytes=Path(n).stat().st_size,sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest()) for n in ['cpu-result.json','probe-partial.json','probe-result.json','gpu-release.json'] if Path(n).exists()]))
'''
        r=run(code);assert r['exit_code']==0,r;items=json.loads(r['stdout']);(b/'terminal').mkdir(exist_ok=False)
        for it in items:
            raw=read(it['name']);assert len(raw)==it['bytes'] and hashlib.sha256(raw).hexdigest()==it['sha256'];(b/'terminal'/it['name']).write_bytes(raw)
        save('collection.json',dict(items=items,complete=(b/'terminal/probe-result.json').exists()));print(json.dumps(dict(running=False,complete=(b/'terminal/probe-result.json').exists())))

if __name__=='__main__':main()
