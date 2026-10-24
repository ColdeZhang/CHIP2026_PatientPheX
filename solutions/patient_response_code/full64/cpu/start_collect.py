"""One bounded CPU census in existing frozen runtime; no GPU/model/quality."""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
P=argparse.ArgumentParser();P.add_argument('action',choices=['start','inspect-collect']);P.add_argument('--directory',type=Path,required=True);a=P.parse_args();B=a.directory
plan=json.loads((B/'plan.json').read_text());M=plan['mount_id'];rel=plan['remote'];root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel
python=plan['python'];base=plan['model'];env=plan['env']
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,j):
 with (B/n).open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
cl=Client();cl.initialize()
def call(name,args,label):
 r=cl.request('tools/call',dict(name=name,arguments=dict(mount_id=M,**args)));save(label,r)
 if r.get('isError'):raise RuntimeError('Original CPU operation unknown/error; preserve evidence, do not blind repeat start')
 return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def remote(code,params,label):
 r=call('run_command',dict(command='python3 -c '+shlex.quote('PARAM='+repr(params)+'\n'+code),timeout_seconds=30),label);assert r['exit_code']==0;return json.loads(r['stdout'])
if a.action=='start':
 assert (B/'deployment-result.json').is_file() and not (B/'start-intent.json').exists()
 assert json.loads((B/'structural-result.json').read_text())['status']=='PASS_FULL64_SHORT265_FULLTEXT64_RESPONSE_CODE_NO_DEV_GOLD'
 driver='''import json,os,subprocess,time
from pathlib import Path
r=Path.cwd();p=r/'driver-started.json'
with p.open('x') as f:json.dump(dict(pid=os.getpid(),started_unix=time.time(),CPU_only=True),f);f.flush();os.fsync(f.fileno())
s=time.time()
with open('stdout.log','xb') as o,open('stderr.log','xb') as e:
 c=subprocess.Popen(COMMAND,stdout=o,stderr=e);rc=c.wait()
with open('terminal.json','x') as f:json.dump(dict(returncode=rc,seconds=time.time()-s,completed_unix=time.time(),CPU_only=True,child_pid=c.pid),f);f.flush();os.fsync(f.fileno())
raise SystemExit(rc)
'''
 cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s','120s',python,str(Path(root)/'native_cpu.py'),'--plan',root+'/cpu-plan.json','--base',base]
 save('start-intent.json',dict(at_unix=time.time(),cap_seconds=120,grace_seconds=5,command=cmd,environment=env,source_sha256=sha(Path(__file__).read_bytes()),no_retry_unknown=True))
 result=call('start_process',dict(command=shlex.join([python,'-u','-c','COMMAND='+repr(cmd)+'\n'+driver]),cwd=rel,env=env),'start-tool.json');save('job.json',result);print(json.dumps(result))
else:
 code='''from pathlib import Path
import json,hashlib,base64,io,zipfile
r=Path(PARAM['root']);out={}
if not (r/'terminal.json').exists():print(json.dumps(dict(complete=False)));raise SystemExit(0)
names=['driver-started.json','terminal.json','stdout.log','stderr.log']
names += [n for n in ['native-intent.json','native-row-audit.json','native-result.json'] if (r/n).exists()]
buf=io.BytesIO();refs={}
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names:
  b=(r/n).read_bytes();refs[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());z.writestr(n,b)
b=buf.getvalue();assert len(b)<200000
print(json.dumps(dict(complete=True,terminal=json.loads((r/'terminal.json').read_text()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),files=refs,b64=base64.b64encode(b).decode())))
'''
 r=remote(code,dict(root=root),'inspect-collect-%d-tool.json'%time.time_ns())
 if not r['complete']:print(json.dumps(r))
 else:
  import io,zipfile
  b=base64.b64decode(r.pop('b64'));assert len(b)==r['bytes'] and sha(b)==r['sha256'];(B/'native-terminal.zip').write_bytes(b);D=B/'native-terminal';D.mkdir(exist_ok=False)
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   for n,f in r['files'].items():
    q=z.read(n);assert len(q)==f['bytes'] and sha(q)==f['sha256'];(D/n).write_bytes(q)
  save('native-terminal-manifest.json',r);print(json.dumps(dict(complete=True,terminal=r['terminal'],zip_sha256=r['sha256'],members=len(r['files']))))
