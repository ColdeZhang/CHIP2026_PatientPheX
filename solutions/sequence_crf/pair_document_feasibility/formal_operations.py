#!/usr/bin/env python3
"""Deploy/start/recover one bounded formal run without duplicating a live job."""
import argparse,base64,datetime,hashlib,json,shlex,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['deploy','start','collect']);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--checkpoint');a=ap.parse_args();b=a.directory
    resource=json.loads((b/'resource.json').read_text());remote=resource['remote'];plan=json.loads((b/'formal-plan.json').read_text())
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(name,**args):
        r=client.request('tools/call',dict(name=name,arguments=dict(mount_id=resource['mount'],**args)));assert not r.get('isError'),r
        return r['structuredContent']
    def save(name,x):
        with (b/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
    def run(code,timeout=30):return call('run_command',command=resource['python']+' -c '+shlex.quote(code),cwd=remote,env=resource['CPU_env'],timeout_seconds=timeout)
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.mode=='deploy':
        assert not (b/'deployment.json').exists() and not (b/'job.json').exists()
        files={**plan['remote_files'],'formal-plan.json':{'sha256':digest(b/'formal-plan.json'),'bytes':(b/'formal-plan.json').stat().st_size}}
        code='from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();p.mkdir(parents=True)'
        r=call('run_command',command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10);assert r['exit_code']==0,r
        for name,ref in files.items():
            raw=(b/name).read_bytes();assert digest(b/name)==ref['sha256'] and len(raw)==ref['bytes'];parts=[]
            for i,start in enumerate(range(0,len(raw),256*1024)):
                part=name+f'.part{i:03d}';parts.append(part);call('write_file',path=remote+'/'+part,encoding='base64',content=base64.b64encode(raw[start:start+256*1024]).decode())
            code='from pathlib import Path;import hashlib;parts='+repr(parts)+';raw=b"".join(Path(x).read_bytes() for x in parts);assert hashlib.sha256(raw).hexdigest()=='+repr(ref['sha256'])+';Path('+repr(name)+').write_bytes(raw);[Path(x).unlink() for x in parts]'
            r=run(code);assert r['exit_code']==0,r
        code='import py_compile,json;[py_compile.compile(x,doraise=True) for x in '+repr([n for n in files if n.endswith('.py')])+'];print(json.dumps(dict(compiled=True,training_started=False)))'
        r=run(code);assert r['exit_code']==0,r;save('deployment.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'actual_cpu_compile':r,'GPU_started':False});print(json.dumps({'deployed_files':len(files),'GPU_started':False}))
    elif a.mode=='start':
        assert a.checkpoint and len(a.checkpoint)==40 and (b/'deployment.json').exists() and not (b/'job.json').exists()
        review=json.loads((b/'formal-entry-review.json').read_text());assert review['status'].startswith('PASS')
        code='''from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
p=json.loads(Path('formal-plan.json').read_text());assert hashlib.sha256(Path('formal-plan.json').read_bytes()).hexdigest()==PLAN
assert not Path('uniform').exists() and not Path('pair').exists() and not Path('training-complete.json').exists()
for name,r in p['remote_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==r['sha256']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
target=[x for x in gpu.splitlines() if p['gpu_uuid'] in x];assert len(target)==1 and p['gpu_uuid'] not in apps and int(target[0].split(',')[1])<512
assert shutil.disk_usage('.').free>8*1024**3
print(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_empty=True,gpu=target[0],free_bytes=shutil.disk_usage('.').free,plan_sha256=PLAN)))
'''.replace('PLAN',repr(digest(b/'formal-plan.json')))
        r=run(code);assert r['exit_code']==0,r;save('fresh-GPU.json',r)
        argv=['timeout','--signal=TERM','--kill-after=30s','1800s',resource['python'],'-u','train_pair_document.py']
        args=dict(command=shlex.join(argv),cwd=remote,env=resource['GPU_env']);r=call('start_process',**args)
        save('job.json',dict(arguments=args,result=r,checkpoint=a.checkpoint,plan_sha256=digest(b/'formal-plan.json'),review_sha256=digest(b/'formal-entry-review.json'),cap=1800,grace=30));print(json.dumps(r))
    else:
        assert not (b/'collection.json').exists(),'Already collected: use existing immutable result'
        job=json.loads((b/'job.json').read_text());records={s:call('read_process_output',process_id=job['result']['id'],stream=s,cursor=0,limit=30000) for s in ['stdout','stderr']}
        if any(x['running'] for x in records.values()):print(json.dumps(dict(running=True,output={k:v['content'][-1800:] for k,v in records.items()})));return
        save('process-terminal.json',records)
        code='''from pathlib import Path
import json,hashlib,subprocess,datetime
p=json.loads(Path('formal-plan.json').read_text());apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);assert p['gpu_uuid'] not in apps
Path('gpu-release.json').write_text(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=p['gpu_uuid'],target_empty=True)))
names=['training-complete.json','gpu-release.json']+[mode+'/'+n for mode in ['uniform','pair'] for n in ['progress.json','complete.json','failure.json','probabilities.json']]
print(json.dumps([dict(name=n,bytes=Path(n).stat().st_size,sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest()) for n in names if Path(n).exists()]))
'''
        r=run(code);assert r['exit_code']==0,r;items=json.loads(r['stdout']);dest=b/'terminal';dest.mkdir(exist_ok=False)
        for it in items:
            parts=[]
            for offset in range(0,it['bytes'],512*1024):
                x=call('read_file',path=remote+'/'+it['name'],offset=offset,limit=min(512*1024,it['bytes']-offset));parts.append(base64.b64decode(x['content']) if x['encoding']=='base64' else x['content'].encode())
            raw=b''.join(parts);assert len(raw)==it['bytes'] and hashlib.sha256(raw).hexdigest()==it['sha256'];p=dest/it['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        save('collection.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),items=items,complete=(dest/'training-complete.json').exists(),remote=resource,model_and_optimizer_states_retained_remote=True));print(json.dumps(dict(running=False,complete=(dest/'training-complete.json').exists())))

if __name__=='__main__':main()
