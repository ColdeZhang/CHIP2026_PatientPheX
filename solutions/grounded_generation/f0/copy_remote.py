"""One bounded private same-byte adapter copy, recover original intents after UNKNOWN."""
import argparse,fcntl,hashlib,json,os,shlex,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/F0-copy-v1')
REL='grounded-generation-v1/F0-copy-v1';P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/'+REL

def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 q=p.parent.resolve()
 while True:
  fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if q==q.parent:break
  q=q.parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','deploy','serve','observe-source','fetch','observe-destination']);a=ap.parse_args();B.mkdir(exist_ok=True);os.umask(0o077)
 lock=(B/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=json.loads((S/'copy-plan.json').read_text());assert ref(S/'adapter_copy.py')==p['source']
 for backoff in B.glob('*.backoff.json'):assert time.time()>=json.loads(backoff.read_text())['not_before_unix']
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize()
 def call(tool,args,tag):
  dest=B/(tag+'-'+str(time.time_ns()));safe=dict(args)
  if 'content' in safe:safe['content']='[frozen source bytes omitted]'
  save(dest.with_suffix('.intent.json'),dict(tool=tool,arguments=safe))
  try:r=c.request('tools/call',dict(name=tool,arguments=args))
  except Exception as error:save(dest.with_suffix('.error.json'),dict(UNKNOWN=True,type=type(error).__name__));raise RuntimeError('UNKNOWN; inventory original namespace, no replay') from None
  text=json.dumps(r)
  for k in ['MCP_URL','MCP_TOKEN']:
   if os.environ.get(k):text=text.replace(os.environ[k],'[REDACTED]')
  r=json.loads(text)
  if r.get('isError'):
   save(dest.with_suffix('.error.json'),r)
   if 'LUNA_OFFLINE' in text:save(dest.with_suffix('.backoff.json'),dict(not_before_unix=time.time()+185))
   raise RuntimeError('Tool error preserved; do not replay start')
  v=r.get('structuredContent')
  if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
  save(dest.with_suffix('.response.json'),v);return v
 def cmd(mount,code,tag):
  v=call('run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=40),tag);assert v['exit_code']==0;return json.loads(v['stdout'])
 def inventory(mount,source):
  code="import hashlib,json,os,subprocess;from pathlib import Path;root=Path("+repr(ROOT)+");p="+repr(p)+";base=Path(p['source_directory' if "+repr(source)+" else 'destination']);print(json.dumps(dict(addresses=subprocess.check_output(['hostname','-I'],text=True).split(),free=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize,files={name:dict(bytes=(base/name).stat().st_size,sha256=hashlib.sha256((base/name).read_bytes()).hexdigest()) for name in p['files'] if (base/name).is_file()},markers=[n for n in ['serve-intent.json','fetch-intent.json','source.stdout.log','fetch.stdout.log'] if (root/n).exists()],deployment={n:dict(bytes=(root/n).stat().st_size,sha256=hashlib.sha256((root/n).read_bytes()).hexdigest()) for n in ['adapter_copy.py','copy-plan.json'] if (root/n).exists()})))"
  return cmd(mount,code,'inventory-source' if source else 'inventory-destination')
 if a.stage=='inventory':
  src=inventory(p['source_mount'],True);dst=inventory(p['destination_mount'],False)
  assert p['source_private_address'] in src['addresses'] and src['files']==p['files'] and dst['free']>=p['minimum_free_bytes']
  save(B/'inventory.json',dict(source=src,destination=dst,plan=ref(S/'copy-plan.json')))
 elif a.stage=='deploy':
  assert (B/'inventory.json').exists()
  for mount,source in [(p['source_mount'],True),(p['destination_mount'],False)]:
   actual=inventory(mount,source);assert not actual['markers']
   for name in ['adapter_copy.py','copy-plan.json']:
    wanted=ref(S/name);expected={k:wanted[k] for k in ['bytes','sha256']}
    if name in actual['deployment']:assert actual['deployment'][name]==expected;continue
    v=call('write_file',dict(mount_id=mount,path=REL+'/'+name,content=(S/name).read_text(),encoding='utf8',create_parents=True),'deploy-'+('source-' if source else 'dest-')+name);assert v['sha256']==wanted['sha256'] and v['size']==wanted['bytes']
  save(B/'deployed.json',dict(plan=ref(S/'copy-plan.json'),source=ref(S/'adapter_copy.py')))
 elif a.stage in ['serve','fetch']:
  assert (B/'deployed.json').exists();source=a.stage=='serve';mount=p['source_mount'] if source else p['destination_mount'];intent=B/(a.stage+'-intent.json');assert not intent.exists()
  actual=inventory(mount,source);assert not actual['markers']
  for name in ['adapter_copy.py','copy-plan.json']:assert actual['deployment'][name]=={k:ref(S/name)[k] for k in ['bytes','sha256']}
  argv=['timeout','--signal=TERM','--kill-after=10','180','python3','-u','adapter_copy.py',a.stage,'--plan','copy-plan.json']
  if not source:
   observed=json.loads((B/'source-ready.json').read_text());ready=observed['ready'];argv+=['--url','http://'+p['source_private_address']+':'+str(ready['port']),'--manifest-sha',ready['manifest_sha256'],'--deadline',str(time.time()+180)]
  command=shlex.join(argv)+' > '+('source.stdout.log' if source else 'fetch.stdout.log')+' 2> '+('source.stderr.log' if source else 'fetch.stderr.log')
  claim=dict(command=command,at_unix=time.time(),no_retry=True,CPU_only=True,cap_seconds=180,grace_seconds=10)
  save(intent,claim)
  claimcode='import json,os;from pathlib import Path;p=Path('+repr(ROOT+'/'+a.stage+'-intent.json')+');p.parent.mkdir(parents=True,exist_ok=True);f=p.open("x");json.dump('+repr(claim)+',f);f.flush();os.fsync(f.fileno());f.close();fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print("{}")'
  cmd(mount,claimcode,'claim-'+a.stage)
  result=call('start_process',dict(mount_id=mount,cwd=REL,command=command,env={'CUDA_VISIBLE_DEVICES':''}),'start-'+a.stage);save(B/(a.stage+'-started.json'),result);print(json.dumps(result))
 else:
  source=a.stage=='observe-source';mount=p['source_mount'] if source else p['destination_mount'];name='source' if source else 'fetch'
  code="import json;from pathlib import Path;r=Path("+repr(ROOT)+");name="+repr(name)+";out=r/(name+'.stdout.log');err=r/(name+'.stderr.log');print(json.dumps(dict(stdout=out.read_text() if out.exists() else None,stderr=err.read_text() if err.exists() else None,copy_ack=json.loads(Path("+repr(p['destination']+'/copy-ack.json')+").read_text()) if "+repr(not source)+" and Path("+repr(p['destination']+'/copy-ack.json')+").is_file() else None)))"
  actual=cmd(mount,code,a.stage);save(B/(a.stage+'-'+str(time.time_ns())+'.json'),actual)
  if source and actual['stdout']:
   ready=json.loads(actual['stdout'].splitlines()[0]);assert ready['ready'] and ready['manifest']['files']==p['files'];save(B/'source-ready.json',dict(ready=ready,plan=ref(S/'copy-plan.json')))
  elif not source and actual['copy_ack']:
   ack=actual['copy_ack'];assert ack['files']==p['files'] and ack['destination_all_files_fsync_and_reopen_SHA'];save(B/'copy-complete.json',dict(ack=ack,source_intent=ref(B/'serve-intent.json'),fetch_intent=ref(B/'fetch-intent.json'),plan=ref(S/'copy-plan.json')))
  print(json.dumps(actual))
if __name__=='__main__':main()
