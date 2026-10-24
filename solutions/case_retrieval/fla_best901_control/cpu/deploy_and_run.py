"""Deploy exact four old prompt streams and execute one CPU-only native audit."""
import argparse,base64,hashlib,io,json,os,shlex,sys,zipfile
from pathlib import Path
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE='research/fla-best901-originaldev-20261024/input-cpu-v1'
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent;plan=json.loads((source/'plan.json').read_text())
    data={n:(source/n).read_bytes() for n in ['token_audit.py','run_all.py','plan.json']}
    for n,r in plan['sources'].items():assert len(data[n])==r['bytes'] and sha(data[n])==r['sha256']
    for name,flow in plan['flows'].items():
        r=flow['prompts'];b=Path(r['path']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'];data[name+'.prompts.jsonl']=b
    members={n:dict(bytes=len(b),sha256=sha(b)) for n,b in data.items()}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(data.items()):z.writestr(n,b)
    blob=buf.getvalue();(out/'deployment.zip').write_bytes(blob)
    save(out/'intent.json',dict(source_sha256=sha(Path(__file__).read_bytes()),archive_sha256=sha(blob),archive_bytes=len(blob),members=members,remote=REMOTE,cpu_seconds=60,no_GPU=True))
    sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
    c=Client();c.initialize()
    def call(name,label,**kw):
        r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=MOUNT,**kw)));save(out/(label+'.json'),r);assert not r.get('isError');return r.get('structuredContent') or json.loads(r['content'][0]['text'])
    r=call('run_command','predeploy',command='python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(REMOTE)+').exists();print("ABSENT")'),timeout_seconds=10);assert r['exit_code']==0
    parts=[]
    for i,offset in enumerate(range(0,len(blob),65536)):
        n=REMOTE+'/parts/%04d.bin'%i;parts.append(n);call('write_file','upload%04d'%i,path=n,content=base64.b64encode(blob[offset:offset+65536]).decode(),encoding='base64',create_parents=True)
    code='''from pathlib import Path
import json,hashlib,zipfile,io,os
p=Path(REMOTE);b=b''.join(Path(n).read_bytes() for n in PARTS);assert hashlib.sha256(b).hexdigest()==SHA
with zipfile.ZipFile(io.BytesIO(b)) as z:
 assert set(z.namelist())==set(MEMBERS)
 for n,r in MEMBERS.items():
  b=z.read(n);assert Path(n).name==n and len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  with (p/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
q=p.resolve();root=Path.cwd().resolve()
while True:
 fd=os.open(q,os.O_RDONLY);os.fsync(fd);os.close(fd)
 if q==root:break
 assert root in q.parents;q=q.parent
print('EXACT_DEPLOYED_NO_EXECUTION')
'''.replace('REMOTE',repr(REMOTE)).replace('PARTS',repr(parts)).replace('SHA',repr(sha(blob))).replace('MEMBERS',repr(members))
    r=call('run_command','deployment-receipt',command='python3 -c '+shlex.quote(code),timeout_seconds=20);assert r['exit_code']==0
    command=shlex.join(['/usr/bin/timeout','--signal=TERM','--kill-after=5s','65',plan['python'],'-u',ROOT+'/'+REMOTE+'/run_all.py','--plan',ROOT+'/'+REMOTE+'/plan.json'])
    env=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH=ROOT+'/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:'+ROOT+'/case-retrieval/round2/runtime',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
    save(out/'execution-intent.json',dict(command=command,env=env,unique_CPU_execution=True))
    r=call('run_command','actual',command=command,env=env,timeout_seconds=70);assert r['exit_code']==0
    for name in ['result.json']+[x+'.native.json' for x in plan['flow_order']]:
        v=call('read_file','read-'+name,path=REMOTE+'/'+name,offset=0,limit=500000)
        b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode()
        with (out/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    result=json.loads((out/'result.json').read_text());assert result['status']=='PASS_ALL234_NATIVE_IDS_INPUTS_OLD_LENGTHS_NO_GPU_GOLD'
    for name,row in result['flows'].items():
        b=(out/(name+'.native.json')).read_bytes();assert len(b)==row['audit']['bytes'] and sha(b)==row['audit']['sha256']
    save(out/'collection.json',dict(status='ALL_FOUR_NATIVE_AUDITS_LOCAL_SHA_VERIFIED',result_sha256=sha((out/'result.json').read_bytes()),remote=REMOTE,no_model_or_gold_or_score=True));print(json.dumps(dict(status=result['status'],rows=result['rows'],tokens=result['tokens'],seconds=result['elapsed_seconds'],result_sha256=sha((out/'result.json').read_bytes()))))
if __name__=='__main__':main()
