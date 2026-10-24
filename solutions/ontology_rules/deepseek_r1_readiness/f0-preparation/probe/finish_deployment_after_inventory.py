"""Resolve UNKNOWN upload by actual fullSHA inventory; run remaining deploy check once, never reupload."""
from pathlib import Path
import ast,json,hashlib,os,sys,shlex,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9'
def save(n,x):
 with (P/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(x):assert not x.get('isError'),x;return x.get('structuredContent') or json.loads(x['content'][0]['text'])
j=json.loads((P/'deployment-manifest.json').read_text());remote=j['remote'];relative=remote.split('/CHIP2026_PatientPheX/',1)[1];b=(P/'deployment.tar.gz').read_bytes();sha=hashlib.sha256(b).hexdigest()
x=un(json.loads((P/'post-UNKNOWN-actual-inventory.json').read_text()));assert x['exit_code']==0;inventory=json.loads(x['stdout']);assert inventory['archive']=={'exists':True,'bytes':len(b),'sha256':sha} and inventory['probe_started'] is False
expected={v['name']:{k:v['local'][k] for k in ['bytes','sha256']} for v in j['files']}
source=(P/'deploy_sources_once.py').read_text();assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='code');ns=dict(remote=remote,expected=expected,b=b,hashlib=hashlib);exec(compile(ast.Module(body=[assignment],type_ignores=[]),'original-deploy-code','exec'),ns);code=ns['code']
plan=json.loads((P/'plan.pending.json').read_text());save('finish-deployment-intent.json',{'at':time.time(),'actual_archive_SHA':sha,'no_reupload_GPU_model':True,'original_source_SHA':hashlib.sha256(source.encode()).hexdigest(),'command_code_SHA':hashlib.sha256(code.encode()).hexdigest()})
c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=shlex.quote(plan['python'])+' -c '+shlex.quote(code),cwd=relative,env={'CUDA_VISIBLE_DEVICES':'',**plan['GPU_environment'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},timeout_seconds=45)});save('actual-deployment-tool.json',r);x=un(r);assert x['exit_code']==0,x;v=json.loads(x['stdout']);save('actual-deployment.json',v);print(json.dumps({k:v[k] for k in ['status','runtime','CUDA_initialized','SHM_free','copy_progress','copy_ACK_exists']}))
