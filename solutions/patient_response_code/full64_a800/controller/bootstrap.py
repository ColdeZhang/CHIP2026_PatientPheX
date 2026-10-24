"""One guarded original formal entry after exact-piece transport recovery; no recipe changes."""
from pathlib import Path
import argparse,ast,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def constant(source,name):
 values=[n.value.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)];assert len(values)==1;return values[0]
BOOT='''from pathlib import Path
import contextlib,hashlib,io,json,os,subprocess,time,traceback
p=Path(P['root']);assert p.is_dir() and not p.is_symlink();control=p/'bootstrap-control';control.mkdir(exist_ok=False)
def save(name,x):
 with (control/name).open('x') as f:json.dump(x,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
 fd=os.open(str(control),os.O_RDONLY);os.fsync(fd);os.close(fd)
def run_source(source,args,label):
 output=io.StringIO()
 with contextlib.redirect_stdout(output):exec(compile(source,label,'exec'),args)
 value=json.loads(output.getvalue());save(label+'.json',value);return value
save('claim.json',dict(pid=os.getpid(),at_unix=time.time(),campaign=P['campaign'],plan_sha256=P['campaign']['plan_sha256'],bootstrap_sha256=P['bootstrap_sha256'],model_started=False))
try:
 assert time.time()<P['campaign']['absolute_deadline_unix']
 run_source(P['deploy_source'],{'P':P['deploy_parameters']},'actual-deployment')
 plan=json.loads((p/'controller-plan.json').read_text());assert hashlib.sha256((p/'controller-plan.json').read_bytes()).hexdigest()==P['campaign']['plan_sha256']
 run_source(P['preflight_source'],{'PARAM':{'root':str(p)}},'actual-preflight')
 raw=(json.dumps(P['campaign'],indent=2)+chr(10)).encode();q=p/'campaign.json'
 if q.exists():assert q.read_bytes()==raw
 else:
  with q.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert time.time()<P['campaign']['absolute_deadline_unix'];argv=[plan['python'],'-u',str(p/'epoch_driver.py'),'--plan',str(p/'controller-plan.json'),'--campaign',str(q),'--epoch','1'];save('original-driver-launch.json',dict(at_unix=time.time(),argv=argv,original_epoch_driver_sha256=hashlib.sha256((p/'epoch_driver.py').read_bytes()).hexdigest(),no_deadline_reset=True))
 child=subprocess.Popen(argv,cwd=p);save('original-driver-child.json',dict(pid=child.pid));rc=child.wait();save('complete.json',dict(at_unix=time.time(),returncode=rc,no_deadline_reset=True));raise SystemExit(rc)
except Exception as error:
 save('failure.json',dict(at_unix=time.time(),error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),original_clock_unchanged=True));raise
'''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['target-stage','prepare','start']);ap.add_argument('--controller',type=Path,required=True);a=ap.parse_args();C=a.controller;S=Path(__file__).resolve().parent;plan=json.loads((C/'controller-plan.json').read_text());target=plan['persistent_mount_id'];source=plan['mount_id'];cl=Client();cl.initialize();T=C/'bootstrap-transport';T.mkdir(exist_ok=True)
 def call(mount,n,args,label):
  p=T/(label+'-'+str(time.time_ns()));save(p.with_suffix('.intent.json'),dict(tool=n,arguments=args));r=cl.request('tools/call',dict(name=n,arguments=dict(mount_id=mount,**args)));save(p.with_suffix('.tool.json'),r);assert not r.get('isError'),str(r);return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 def remote(mount,code,param,label):
  r=call(mount,'run_command',dict(command='python3 -c '+shlex.quote('P='+repr(param)+'\n'+code),timeout_seconds=30),label);assert r['exit_code']==0;return json.loads(r['stdout'])
 if a.action=='target-stage':
  code='''from pathlib import Path
import hashlib,json,os,shutil
p=Path(P['root']);assert os.getuid()==1006;p.mkdir(parents=True,exist_ok=True,mode=0o700);assert p.stat().st_uid==1006 and not p.is_symlink();os.chmod(p,0o700);assert shutil.disk_usage(p).free>=2*1024**3;assert not (p/'campaign.json').exists()
for n,content in P['files'].items():
 b=content.encode();q=p/n
 if q.exists():assert q.read_bytes()==b
 else:
  with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 with q.open('rb') as f:os.fsync(f.fileno())
 if n.endswith('.py'):compile(b,n,'exec')
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ACTUAL_PERSISTENT_FETCH_SOURCE_FSYNC',free=shutil.disk_usage(p).free,files={n:dict(bytes=(p/n).stat().st_size,sha256=hashlib.sha256((p/n).read_bytes()).hexdigest()) for n in P['files']})))
'''
  actual=remote(target,code,dict(root=plan['persistent_remote_absolute'],files={n:(C/n).read_text() for n in ['controller-plan.json','fetch_checkpoint.py','io_driver.py']}),'target-stage');save(C/'target-deployment.json',actual);print(json.dumps(actual))
 elif a.action=='prepare':
  deploy=constant((S/'deploy.py').read_text(),'verify');tree=ast.parse((S/'operations.py').read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and any(isinstance(q,ast.Constant) and q.value=='preflight' for q in n.test.comparators)];assert len(nodes)==1;sub=ast.Module(body=nodes[0].body,type_ignores=[]);preflight=constant(ast.unparse(sub),'code')
  manifest=json.loads((C/'deployment-manifest.json').read_text());b=Path(manifest['archive']['path']).read_bytes();assert sha(b)==manifest['archive']['sha256'];parts=[dict(name='part-%03d'%i,bytes=len(b[j:j+32768]),sha256=sha(b[j:j+32768])) for i,j in enumerate(range(0,len(b),32768))]
  payload=dict(root=plan['remote_absolute'],deploy_source=deploy,preflight_source=preflight,deploy_parameters=dict(root=plan['remote_absolute'],manifest=manifest,parts=parts,decompress=plan['decompress'],shm=str(Path(plan['output_dir']).parent),output=plan['output_dir']),bootstrap_sha256=ref(__file__)['sha256'])
  save(C/'bootstrap-payload.json',payload);save(C/'bootstrap-entry.json',dict(source=ref(__file__),payload=ref(C/'bootstrap-payload.json'),original_entry=ref(C/'entry.json'),original_plan=ref(C/'controller-plan.json'),archive=manifest['archive'],operations=ref(S/'operations.py'),watcher=ref(S/'watch_authorized_campaign.py'),no_method_or_data_change=True,unique_epoch1_start=True,shared5400_from_original_managed_bootstrap_start=True,source_verify_and_preflight_reused_exact_strings=True));print(json.dumps(ref(C/'bootstrap-entry.json')))
 else:
  assert (C/'target-deployment.json').is_file() and not (C/'epoch-001-start-intent.json').exists() and not (C/'campaign.json').exists();entry=json.loads((C/'bootstrap-entry.json').read_text());assert entry['source']==ref(__file__) and entry['payload']==ref(C/'bootstrap-payload.json');auth=json.loads((C/'GPU-AUTHORIZATION.json').read_text());assert auth['authorized'] and auth['budget_seconds']==5400 and auth['controller_plan_sha256']==ref(C/'controller-plan.json')['sha256']
  started=time.time();campaign=dict(plan_sha256=ref(C/'controller-plan.json')['sha256'],cap_seconds=5400,grace_seconds=30,started_unix=started,absolute_deadline_unix=started+5400,no_budget_reset=True);save(C/'campaign.json',campaign)
  code='''from pathlib import Path
import json,os,hashlib
p=Path(P['root']);b=(json.dumps(P['value'],indent=2)+chr(10)).encode();q=p/'campaign.json'
if q.exists():assert q.read_bytes()==b
else:
 with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd);print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))
'''
  remote(target,code,dict(root=plan['persistent_remote_absolute'],value=campaign),'target-campaign')
  payload=json.loads((C/'bootstrap-payload.json').read_text());payload['campaign']=campaign;body='P='+repr(payload)+'\n'+BOOT;compile(body,'bootstrap-remote.py','exec');assert len(body.encode())<60000
  command=shlex.join(['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(max(1,int(campaign['absolute_deadline_unix']-time.time())))+'s','python3','-u','-c',body]);save(C/'epoch-001-start-intent.json',dict(at_epoch=time.time(),plan_sha256=campaign['plan_sha256'],campaign=campaign,epoch=1,bootstrap_entry=ref(C/'bootstrap-entry.json'),bootstrap_command_sha256=sha(command.encode()),no_retry=True))
  actual=call(source,'start_process',dict(command=command,cwd=plan['remote_relative'],env=plan['env']),'original-epoch1-start');save(C/'epoch-001-job.json',actual);print(json.dumps({k:actual[k] for k in ['id','pid','started_at']}))
if __name__=='__main__':main()
