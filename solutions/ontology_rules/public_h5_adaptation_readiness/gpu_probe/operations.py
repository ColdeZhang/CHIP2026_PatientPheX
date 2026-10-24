"""Prepared local GPU entry; all remote mutations need later explicit activation."""
from pathlib import Path
import base64,datetime,hashlib,io,json,shlex,sys,zipfile
S=Path(__file__).parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/public-h5-train32-probe-readiness');REMOTE='public-h5-train32-gpu-probe-v1';M='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def activation():
 a=json.loads((D/'activation.json').read_text());assert a['official_submission_accepted'] is True and a['execute_this_probe'] is True and a['plan_sha256']==sha(S/'plan.json');return a
mode=sys.argv[1]
if mode=='prepare':
 D.mkdir(exist_ok=True);plan=json.loads((S/'plan.json').read_text());files={n:S/n for n in ['probe.py','loss.py','bounded_driver.py','plan.json','sampling.json']};files['training-queries.jsonl.gz']=Path(plan['dataset']['path']);files['variable-coverage.json']=Path(plan['prior_variable_coverage']['path'])
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.writestr(n,p.read_bytes())
 archive=D/'prepared.zip'
 with archive.open('xb') as f:f.write(buf.getvalue())
 with zipfile.ZipFile(archive) as z:
  for n,p in files.items():assert z.read(n)==p.read_bytes()
 write(D/'ready.json',dict(status='PREPARED_ONLY_NO_REMOTE_OR_GPU',archive=ref(archive),files={n:ref(p) for n,p in files.items()},execution_authorized=False));print(json.dumps(dict(status='PREPARED_ONLY',archive=ref(archive))));raise SystemExit(0)
if mode in ['deploy','start']:a=activation()
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 r=c.request('tools/call',dict(name=n,arguments=dict(mount_id=M,**args)));assert not r.get('isError');return r['structuredContent']
def command(code):
 r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=30));assert r['exit_code']==0,r.get('stderr','')[-800:];return json.loads(r['stdout'])
if mode=='deploy':
 ready=json.loads((D/'ready.json').read_text());assert sha(D/'prepared.zip')==ready['archive']['sha256'];b=(D/'prepared.zip').read_bytes();h=hashlib.sha256(b).hexdigest();write(D/'deploy-intent.json',dict(archive_sha256=h,activation=a))
 for i,off in enumerate(range(0,len(b),196608)):call('write_file',dict(path=REMOTE+'-upload/part-%03d'%i,content=base64.b64encode(b[off:off+196608]).decode(),encoding='base64',create_parents=True))
 code="""from pathlib import Path
import hashlib,io,json,zipfile,os
root=Path(REMOTE);assert not root.exists();b=b''.join(p.read_bytes() for p in sorted(Path(REMOTE+'-upload').glob('part-*')));assert hashlib.sha256(b).hexdigest()==HASH;root.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all('/' not in n for n in z.namelist());z.extractall(root);(root/'activation.json').write_text(json.dumps(ACTIVATION,indent=2)+'\\n');result={}
for p in root.iterdir():
 with p.open('rb') as f:os.fsync(f.fileno())
 result[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(result))
""".replace('REMOTE',repr(REMOTE)).replace('HASH',repr(h)).replace('ACTIVATION',repr(a));actual=command(code)
 for n,r in ready['files'].items():assert actual[n]['sha256']==r['sha256'] and actual[n]['bytes']==r['bytes']
 write(D/'deployment.json',dict(files=actual,archive_sha256=h));print(json.dumps(dict(status='DEPLOYED_AFTER_ACTIVATION')))
elif mode=='start':
 assert (D/'deployment.json').exists();write(D/'start-intent.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(S/'plan.json'),activation=a));p=json.loads((S/'plan.json').read_text());j=call('start_process',dict(command=p['native_root']+'/.venv/bin/python bounded_driver.py',cwd=REMOTE,env=dict(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')));write(D/'job.json',j);print(json.dumps(j))
elif mode=='status':
 j=json.loads((D/'job.json').read_text());m=call('read_process_output',dict(process_id=j['id'],stream='stdout',cursor=0,limit=2048));state=command("from pathlib import Path;import json;p=Path("+repr(REMOTE+'/state.json')+");print(p.read_text() if p.exists() else '{}')");(D/'latest-status.json').write_text(json.dumps(dict(managed=m,state=state),indent=2)+'\n');print(json.dumps(dict(running=m.get('running'),state=state)))
elif mode=='collect':
 manifest=command("from pathlib import Path;print(Path("+repr(REMOTE+'/terminal-manifest.json')+").read_text())");assert manifest['state']['status']!='running';write(D/'terminal-manifest.json',manifest);a=manifest['archive'];chunks=[]
 for off in range(0,a['bytes'],196608):
  x=command("from pathlib import Path;import json,base64;f=Path("+repr(a['path'])+").open('rb');f.seek("+str(off)+");print(json.dumps(dict(data=base64.b64encode(f.read(196608)).decode())))");chunks.append(base64.b64decode(x['data']))
 b=b''.join(chunks);assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'];(D/'terminal.zip').write_bytes(b);out=D/'terminal';out.mkdir(exist_ok=False)
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert all('..' not in n and not n.startswith('/') for n in z.namelist());assert json.loads(z.read('members.json'))==manifest['members'];z.extractall(out)
 for n,r in manifest['members'].items():assert sha(out/n)==r['sha256'] and (out/n).stat().st_size==r['bytes']
 write(D/'collection.json',dict(archive=a,all_members_SHA_verified=True));print(json.dumps(dict(status='COLLECTED',archive=a,state=manifest['state'])))
else:raise ValueError(mode)
