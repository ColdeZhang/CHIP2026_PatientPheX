"""Deploy/status/one start of a source-bound pilot; no auto retries after UNKNOWN."""
import argparse,base64,datetime,hashlib,importlib.util,json,os,shlex,time
from pathlib import Path

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def unpack(x):return x.get('structuredContent') or json.loads(x['content'][0]['text'])
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('deploy','status','start'));p.add_argument('--bind-dir',required=True);p.add_argument('--mcp-client',required=True);p.add_argument('--output',required=True);p.add_argument('--release');a=p.parse_args();b=Path(a.bind_dir).resolve();e=json.loads((b/'execution.json').read_text());o=Path(a.output);o.mkdir(parents=True,exist_ok=False);remote=Path(e['references']['inputs_gz']['path']).parent;root=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX');rel=str(remote.relative_to(root));mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
 s=importlib.util.spec_from_file_location('pilot_mcp',a.mcp_client);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);c=m.Client();c.initialize()
 def call(tool,args,label):
  result=c.request('tools/call',dict(name=tool,arguments=args));save(o/(label+'.json'),result)
  if result.get('isError'):raise RuntimeError('Recorded MCP outcome; inspect original state, do not repeat '+label)
  return unpack(result)
 def remote_code(code,label):
  v=call('run_command',dict(mount_id=mount,command=shlex.quote(e['python'])+' -c '+shlex.quote(code),cwd=rel,env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30),label);assert v['exit_code']==0;return json.loads(v['stdout'])
 if a.mode=='deploy':
  expected={}
  for f in sorted(b.iterdir()):
   if not f.is_file() or f.name=='binding.json':continue
   x=ref(f);expected[f.name]={'bytes':x['bytes'],'sha256':x['sha256']};r=call('write_file',dict(mount_id=mount,path=rel+'/'+f.name,content=base64.b64encode(f.read_bytes()).decode(),encoding='base64',create_parents=True),'upload-'+f.name);assert r['sha256']==x['sha256']
  code="import pathlib,json,hashlib; ex="+repr(expected)+"; actual={};\nfor name,w in ex.items():\n p=pathlib.Path(name); b=p.read_bytes(); a={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}; assert a==w,name;actual[name]=a\nprint(json.dumps({'status':'PASS_ALL_DEPLOYED_FILES','files':actual}))"
  result=remote_code(code,'verification');save(o/'complete.json',dict(result,source=ref(__file__),execution=ref(b/'execution.json'),no_GPU=True));print(json.dumps({'status':result['status'],'files':len(expected)}));return
 if a.mode=='status':
  code="import pathlib,json,subprocess; p=pathlib.Path('.');r={};\nfor n in ('launch.accepted.json','driver-start.json','child.json','driver-complete.json','terminal/progress.json','terminal/result.json'):\n f=p/n\n if f.exists():\n  x=json.loads(f.read_text());r[n]=x if n not in ('terminal/progress.json','terminal/result.json') else {k:x.get(k) for k in ('status','phase','rows','error','program_seconds','mechanical_counts')}\nr['gpu']=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout;r['apps']=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout\nprint(json.dumps(r))"
  result=remote_code(code,'observation');save(o/'result.json',result);print(json.dumps(result));return
 assert a.release,'Explicit preceding route fullchain release receipt required'
 release=json.loads(Path(a.release).read_text());assert release['whole_chain_including_Sap_released'] is True and release['gpu_uuid']==e['gpu_uuid']
 # Source/file/fresh GPU checks before creating the sole launch intention.
 code="import pathlib,json,hashlib,subprocess; p=pathlib.Path('execution.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(ref(b/'execution.json')['sha256'])+"; e=json.loads(p.read_text());\nfor n,h in e['sources'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==h\nr={'gpu':subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout,'apps':subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout};print(json.dumps(r))"
 fresh=remote_code(code,'fresh');gpu={l.split(',')[0].strip():[int(x) for x in l.split(',')[1:]] for l in fresh['gpu'].splitlines()};assert gpu[e['gpu_uuid']][0]<=64 and gpu[e['gpu_uuid']][1]*1024**2>e['empirical_memory_proxy_bytes'];assert not any(l.split(',')[0].strip()==e['gpu_uuid'] for l in fresh['apps'].splitlines())
 t=time.time();admission=dict(valid_from_unix=t,valid_until_unix=t+300,execution_sha256=ref(b/'execution.json')['sha256'],prior_owner_release=ref(a.release));save(o/'admission.json',admission)
 # Exclusive local intent prevents repeating a start on a second invocation with another outputdir.
 lock=b/'unique-start-intent.json';save(lock,dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),admission=ref(o/'admission.json'),release=ref(a.release),remote_execution=str(remote/'execution.json')))
 v=call('write_file',dict(mount_id=mount,path=rel+'/admission.json',content=(o/'admission.json').read_text(),encoding='utf8',create_parents=False),'admission-upload');assert v['sha256']==ref(o/'admission.json')['sha256']
 assert time.time()<admission['valid_until_unix']
 cmd=shlex.quote(e['python'])+' -u pilot_driver.py --execution execution.json --admission admission.json'
 save(o/'start-intent.json',dict(command=cmd,cwd=rel,admission=admission,execution=ref(b/'execution.json'),source=ref(__file__)))
 started=call('start_process',dict(mount_id=mount,command=cmd,cwd=rel,env={'PYTHONPATH':e['PYTHONPATH'],'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false'}),'start-response');print(json.dumps(started))
if __name__=='__main__':main()
