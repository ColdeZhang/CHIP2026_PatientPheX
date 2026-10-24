"""Deploy only immutable pending F0 sources/two compressed inputs; pure CPU actual contract, no GPU."""
from pathlib import Path
import json,hashlib,os,sys,shlex,base64,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9';j=json.loads((P/'deployment-manifest.json').read_text());remote=j['remote'];rel=remote.split('/CHIP2026_PatientPheX/',1)[1];b=(P/'deployment.tar.gz').read_bytes();expected={v['name']:{k:v['local'][k] for k in ['bytes','sha256']} for v in j['files']};params={'remote':remote,'bundle_sha256':hashlib.sha256(b).hexdigest(),'expected':expected}
def save(n,x):
 with (P/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(x):assert not x.get('isError'),x;return x.get('structuredContent') or json.loads(x['content'][0]['text'])
code='import json\nparams=json.loads('+repr(json.dumps(params))+')\n'+'''from pathlib import Path
import os,hashlib,tarfile,io,sys,copy,time,shutil
root=Path(params['remote']);assert os.getuid()==root.stat().st_uid==1045;os.chmod(root,0o700);blob=(root/'deployment.tar.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==params['bundle_sha256'];expected=params['expected']
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
 members=tar.getmembers();assert {m.name for m in members}==set(expected) and len(members)==len(expected)
 for member in members:
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  data=tar.extractfile(member).read();e=expected[member.name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==data,'Never replace different deployed bytes'
  else:
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
for name,e in expected.items():
 path=root/name;assert path.stat().st_size==e['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
 if name.endswith('.py'):compile(path.read_text(),name,'exec')
assert not (root/'wrapper-started.json').exists() and not (root/'terminal').exists() and not (root/'plan.json').exists()
plan=json.loads((root/'plan.pending.json').read_text());assert plan['execution_ready'] is False and plan['conditional_F1_gate']['status'].startswith('PENDING')
assert plan['process_budget_seconds']==6000 and plan['TERM_grace_seconds']==30 and plan['cached_indices']==[30] and len(plan['new_indices'])==57
sys.path.insert(0,str(root));from full58_contract import load_inputs
shadow=copy.deepcopy(plan);shadow['execution_ready']=True;start=time.monotonic();rows,encoded,audit,cached=load_inputs(shadow);seconds=time.monotonic()-start
assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'tokenizers' not in sys.modules
out={'status':'PASS_REMOTE_ACTUAL_58_INPUT_ONE_EXACT_CACHE_NO_GPU_PENDING_F1','uid':os.getuid(),'plan_sha256':hashlib.sha256((root/'plan.pending.json').read_bytes()).hexdigest(),'sources':plan['sources'],'all_deployment_files':expected,'rows':len(rows),'tokens':sum(len(e['input_ids']) for e in encoded),'cache_indices':sorted(cached),'cache_sha256':plan['cached_rows']['30']['sha256'],'program_seconds':seconds,'execution_ready_on_disk':False,'CPU_shadow_scope':'Only in-memory pure load_inputs receives execution_ready true; no launcher/native imports/model/gold/quality; original conditional plan remains false','disk_free_bytes':shutil.disk_usage(root).free,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'no_native_model_modules':True}
assert out['disk_free_bytes']>32*1024*1024
with (root/'actual-input-contract.json').open('x') as f:json.dump(out,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
print(json.dumps(out))
'''
compile(code,'remote_pending_deployment','exec');save('deployment-intent.json',{'at':time.time(),**params,'generated_command_sha256':hashlib.sha256(code.encode()).hexdigest(),'generated_command_compiled':True,'no_GPU':True})
(P/'remote-deployment-command.py').write_text(code)
c=Client();c.initialize();r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/deployment.tar.gz',content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)});save('deployment-upload.json',r);un(r)
r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},timeout_seconds=30)});save('actual-input-contract-tool.json',r);v=un(r);assert v['exit_code']==0,v;out=json.loads(v['stdout']);save('actual-input-contract.json',out);print(json.dumps({k:out[k] for k in ['status','rows','tokens','cache_indices','program_seconds','execution_ready_on_disk','disk_free_bytes']}))
