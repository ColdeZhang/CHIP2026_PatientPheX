"""Deploy exact frozen inputs/source and start the sole authorized FLA80 job; no retries."""
import argparse,base64,hashlib,io,json,os,shlex,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE='research/fla-fulltext-normal-quality-20261024/full80-attempt2'
def digest(data):return hashlib.sha256(data).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def unpack(result):
    assert not result.get('isError'), 'MCP failure retained; no automatic retry'
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args()
    out=a.directory.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent;plan=json.loads((source/'plan.json').read_text())
    assert plan['execution_ready'] and plan['fresh_rows']==80 and plan['cached_original_indices']==[]
    data={n:(source/n).read_bytes() for n in plan['sources']}
    data['plan.json']=(source/'plan.json').read_bytes()
    data['model-source-manifest.json']=(source/'model-source-manifest.json').read_bytes()
    for key,v in plan['local_input_references'].items():
        b=Path(v['path']).read_bytes();assert len(b)==v['bytes'] and digest(b)==v['sha256']
        data[Path(plan[key]['path']).name]=b
    members={n:dict(bytes=len(b),sha256=digest(b)) for n,b in data.items()}
    for n,v in plan['sources'].items():assert members[n]==v
    with zipfile.ZipFile(out/'deployment.zip','x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(data.items()):z.writestr(n,b)
    blob=(out/'deployment.zip').read_bytes()
    save(out/'deployment-plan.json',dict(remote=REMOTE,members=members,archive_sha256=digest(blob),archive_bytes=len(blob),source_sha256=digest(Path(__file__).read_bytes()),at_utc=datetime.now(timezone.utc).isoformat()))
    sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
    c=Client();c.initialize()
    def call(name,arguments,record):
        try:r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=MOUNT,**arguments)))
        except BaseException as e:
            save(out/(record+'-exception.json'),dict(error_type=type(e).__name__,state_unknown=True,no_retry=True));raise
        save(out/(record+'.json'),r);return unpack(r)
    # Existence check must precede any mutation, especially starting a model.
    result=call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(REMOTE)+').exists();print("NEW_NAMESPACE_ABSENT")'),timeout_seconds=10),'predeploy')
    assert result['exit_code']==0
    parts=[]
    for i,start in enumerate(range(0,len(blob),65536)):
        name=REMOTE+'/upload-parts/%04d.bin'%i;parts.append(name)
        call('write_file',dict(path=name,content=base64.b64encode(blob[start:start+65536]).decode(),encoding='base64',create_parents=True),'upload-%04d'%i)
    code='''from pathlib import Path
import json,hashlib,zipfile,io,os
p=Path(REMOTE);members=MEMBERS
b=b''.join(Path(n).read_bytes() for n in PARTS);assert hashlib.sha256(b).hexdigest()==ARCHIVE
with zipfile.ZipFile(io.BytesIO(b)) as z:
 assert set(z.namelist())==set(members)
 for n in sorted(members):
  assert Path(n).name==n
  data=z.read(n);v=members[n];assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
  with (p/n).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert (p/n).read_bytes()==data
root=Path.cwd().resolve();q=p.resolve()
while True:
 fd=os.open(str(q),os.O_RDONLY);os.fsync(fd);os.close(fd)
 if q==root:break
 assert root in q.parents;q=q.parent
print(json.dumps(dict(status='DEPLOYED_EXACT_NEW_FLA80_NO_START',members=len(members),plan_sha256=members['plan.json']['sha256'])))
'''.replace('REMOTE',repr(REMOTE)).replace('MEMBERS',repr(members)).replace('PARTS',repr(parts)).replace('ARCHIVE',repr(digest(blob)))
    result=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=30),'deployment-receipt');assert result['exit_code']==0
    command=shlex.join([plan['python'],'-u',plan['project_root']+'/'+REMOTE+'/launch.py','--plan',plan['project_root']+'/'+REMOTE+'/plan.json'])
    save(out/'execution-intent.json',dict(command=command,plan_sha256=members['plan.json']['sha256'],unique_start=True,cap_seconds=1200,grace_seconds=30,at_utc=datetime.now(timezone.utc).isoformat()))
    result=call('start_process',dict(command=command,env=plan['env']),'managed-process')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
