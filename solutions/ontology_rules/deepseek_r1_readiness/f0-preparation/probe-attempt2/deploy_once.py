"""Deploy new owned five-source namespace only; same original metadata, no GPU."""
from pathlib import Path
import base64,json,hashlib,shlex,sys,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9';rel='research/deepseek-r1-0528-qwen3-8b-readiness/f0-longest-v2';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel;b=(P/'sources.tar.gz').read_bytes();plan=json.loads((P/'plan.json').read_text());expected=plan['sources'];digest=hashlib.sha256(b).hexdigest()
def save(n,x):
 with (P/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(x):assert not x.get('isError'),x;return x.get('structuredContent') or json.loads(x['content'][0]['text'])
save('deployment-intent.json',{'at':time.time(),'bundle_sha256':digest,'files':expected,'no_GPU':True});c=Client();c.initialize();r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/sources.tar.gz',content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)});save('deployment-upload.json',r);un(r)
code="""from pathlib import Path
import json,hashlib,tarfile,io,os
p=Path(REMOTE);assert os.getuid()==p.stat().st_uid==1045;os.chmod(p,0o700);b=(p/'sources.tar.gz').read_bytes();assert hashlib.sha256(b).hexdigest()==SHA;expected=EXPECTED
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 members=t.getmembers();assert {m.name for m in members}==set(expected) and len(members)==len(expected)
 for m in members:
  assert m.isfile() and Path(m.name).name==m.name;b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name];q=p/m.name
  if q.exists():assert q.read_bytes()==b
  else:
   with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for n,d in expected.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==d;compile((p/n).read_text(),n,'exec')
assert not (p/'wrapper-started.json').exists() and not (p/'terminal').exists();print(json.dumps({'status':'ACTUAL_ATTEMPT2_SOURCE_SHA_COMPILE_NO_GPU','files':expected,'uid':os.getuid()}))
""".replace('REMOTE',repr(remote)).replace('EXPECTED',repr(expected)).replace('SHA',repr(digest))
r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=20)});save('actual-deployment.json',r);o=un(r);assert o['exit_code']==0,o;print(o['stdout'])
