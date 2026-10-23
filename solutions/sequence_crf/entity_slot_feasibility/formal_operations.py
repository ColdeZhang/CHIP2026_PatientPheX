#!/usr/bin/env python3
"""Managed fresh training and dependent paired inference; no automatic retries."""
from pathlib import Path
import argparse,base64,hashlib,json,shlex,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['deploy','start-train','start-infer','collect-train','collect-infer'])
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--mcp-client-directory',type=Path,required=True)
    ap.add_argument('--checkpoint',help='actual pushed Git SHA required for any start')
    a=ap.parse_args();b=a.directory;p=json.loads((b/'plan.json').read_text());remote=p['remote_directory']
    sys.path.insert(0,str(a.mcp_client_directory));from mcp_client import Client
    c=Client();c.initialize()
    def call(name,args):
        result=c.request('tools/call',dict(name=name,arguments=dict(mount_id=p['mount'],**args)))
        assert not result.get('isError'),result
        return result['structuredContent']
    def save(name,value):
        target=b/name;assert not target.exists();target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if a.mode=='deploy':
        result=call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path; p=Path('+repr(remote)+'); assert not p.exists();p.mkdir(parents=True)'),cwd='',timeout_seconds=10))
        assert result['exit_code']==0
        receipts={}
        for name,digest in p['local_files'].items():
            raw=(b/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
            parts=[]
            for i,offset in enumerate(range(0,len(raw),512*1024)):
                part=remote+'/'+name+f'.part{i:03d}';parts.append(part)
                call('write_file',dict(path=part,encoding='base64',content=base64.b64encode(raw[offset:offset+512*1024]).decode()))
            code='from pathlib import Path;import hashlib;p=Path('+repr(remote+'/'+name)+');parts='+repr(parts)+';v=b"".join(Path(x).read_bytes() for x in parts);assert hashlib.sha256(v).hexdigest()=='+repr(digest)+';p.write_bytes(v);[Path(x).unlink() for x in parts]'
            result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd='',timeout_seconds=10));assert result['exit_code']==0
            receipts[name]=dict(bytes=len(raw),sha256=digest,parts=len(parts))
        call('write_file',dict(path=remote+'/plan.json',encoding='utf8',content=(b/'plan.json').read_text()))
        code="from pathlib import Path;import json,hashlib; p=json.loads(Path('plan.json').read_text());assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in p['local_files'].items());[compile(Path(n).read_text(),n,'exec') for n in p['local_files'] if n.endswith('.py')];print('actual source syntax/hash PASS')"
        result=call('run_command',dict(command=shlex.quote(p['python'])+' -c '+shlex.quote(code),cwd=remote,env=p['env'],timeout_seconds=15));assert result['exit_code']==0
        save('deployment.json',dict(receipts=receipts,actual_cpu=result));print('DEPLOYED')
    elif a.mode.startswith('start-'):
        stage=a.mode.split('-')[1];assert a.checkpoint and len(a.checkpoint)==40 and (b/'deployment.json').exists()
        # Refuse re-entry before any remote mutation (the final save is too late).
        assert not (b/(stage+'-job.json')).exists(), 'Existing managed job; inspect it, never launch a duplicate'
        if stage=='infer':
            terminal=json.loads((b/'terminal-train/training_complete.json').read_text());assert terminal['global_step']==160 and terminal['epochs']==2
            code="""from pathlib import Path
import json,hashlib
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected=EXPECTED_SHA
assert sha('plan.json')==expected
complete=json.loads(Path('training_complete.json').read_text())
checkpoint=json.loads(Path('epoch-002/checkpoint.json').read_text())
manifest=json.loads(Path('run_manifest.json').read_text())
assert complete['plan_sha256']==checkpoint['plan_sha256']==manifest['plan_sha256']==expected
assert complete['manifest_sha256']==checkpoint['training_manifest_sha256']==sha('run_manifest.json')
assert complete['checkpoint_sha256']==sha('epoch-002/checkpoint.json')
assert not Path('inference').exists()
print('frozen plan/training/checkpoint closed; inference output absent')
""".replace('EXPECTED_SHA',repr(hashlib.sha256((b/'plan.json').read_bytes()).hexdigest()))
            result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=15));assert result['exit_code']==0
            save('infer-dependency-preflight.json',result)
        code="""import json,subprocess,shutil,datetime
target=GPU_UUID
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
rows=[x for x in g.splitlines() if target in x];assert len(rows)==1 and target not in apps
assert int(rows[0].split(',')[2])<=512
free=shutil.disk_usage('.').free;assert free>=MIN_FREE
print(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=target,row=rows[0],target_empty=True,free_bytes=free)))
""".replace('GPU_UUID',repr(p['gpu_uuid'])).replace('MIN_FREE',str(p['budgets']['minimum_free_bytes']))
        result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=15));assert result['exit_code']==0
        save(stage+'-fresh-GPU.json',result)
        script='train_entities.py' if stage=='train' else 'infer_entities.py';limit=2200 if stage=='train' else 600
        args=dict(command=shlex.join(['timeout','--signal=TERM','--kill-after=30s',str(limit)+'s',p['python'],script,'--directory','.','--model',p['model']]),cwd=remote,env=p['env'])
        result=call('start_process',args)
        save(stage+'-job.json',dict(arguments=args,result=result,source_checkpoint=a.checkpoint,
            plan_sha256=hashlib.sha256((b/'plan.json').read_bytes()).hexdigest(),main_cap=limit,grace=30))
        print(json.dumps(result))
    else:
        stage=a.mode.split('-')[1];job=json.loads((b/(stage+'-job.json')).read_text())
        records={s:call('read_process_output',dict(process_id=job['result']['id'],stream=s,cursor=0,limit=30000)) for s in ['stdout','stderr']}
        if any(x['running'] for x in records.values()):
            print(json.dumps(dict(running=True,output={k:v['content'][-3000:] for k,v in records.items()})));return
        save(stage+'-process-terminal.json',records)
        code="""from pathlib import Path
import json,hashlib,subprocess,datetime
p=json.loads(Path('plan.json').read_text());target=p['gpu_uuid'];stage=STAGE
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);assert target not in apps
Path(stage+'-gpu-release.json').write_text(json.dumps(dict(target=target,target_empty=True,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\\n')
items=[]
for path in sorted(Path('.').rglob('*')):
 if path.is_file() and path.suffix in ['.json','.jsonl'] and str(path) not in p['local_files'] and path.name!='plan.json':
  assert path.stat().st_size<=2*1024*1024
  items.append(dict(name=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(items))
""".replace('STAGE',repr(stage))
        result=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=remote,timeout_seconds=20));assert result['exit_code']==0
        items=json.loads(result['stdout']);target=b/('terminal-'+stage);target.mkdir(exist_ok=False)
        for item in items:
            parts=[]
            for offset in range(0,item['bytes'],512*1024):
                out=call('read_file',dict(path=remote+'/'+item['name'],offset=offset,limit=min(512*1024,item['bytes']-offset)))
                parts.append(base64.b64decode(out['content']) if out['encoding']=='base64' else out['content'].encode())
            value=b''.join(parts);assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
            path=target/item['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        save(stage+'-collection.json',dict(items=items,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        print(json.dumps(dict(running=False,collected=len(items),training_complete=(target/'training_complete.json').exists(),inference_complete=(target/'inference/complete.json').exists())))

if __name__=='__main__':main()
