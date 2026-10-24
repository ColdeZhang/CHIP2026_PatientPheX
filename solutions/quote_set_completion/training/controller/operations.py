"""Explicit once-only stages. Inspect unknown status; never regenerate model work."""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,shlex,sys,time,tarfile
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','preflight','start-epoch','inspect','start-server','start-fetch','collect-export']);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--epoch',type=int,choices=[1],default=1);a=ap.parse_args();B=a.directory;plan=json.loads((B/'controller-plan.json').read_text());epoch=a.epoch;assert plan['formal_training_authorized'] is True and plan['configuration']['epochs']==1
sha=lambda b:hashlib.sha256(b).hexdigest();plan_sha=sha((B/'controller-plan.json').read_bytes());source=plan['mount_id'];target=plan['persistent_mount_id'];root=plan['remote_absolute'];dest=plan['persistent_remote_absolute'];client=Client();client.initialize()
def save(name,x):
 with (B/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(B),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def call(mount,name,params,label):
 result=client.request('tools/call',dict(name=name,arguments=dict(mount_id=mount,**params)));save(label,result)
 if result.get('isError'):raise RuntimeError('Unknown/error '+name+'; inspect original intent/job, do not repeat mutation')
 return result.get('structuredContent') or json.loads(result['content'][0]['text'])
def remote(mount,code,params,label,timeout=30):
 body='PARAM='+repr(params)+'\n'+code;compile(body,'remote.py','exec');r=call(mount,'run_command',dict(command='python3 -c '+shlex.quote(body),timeout_seconds=timeout),label)
 assert r['exit_code']==0, 'Remote stage failed; inspect saved tool receipt'
 return json.loads(r['stdout'])
def write(mount,relative,raw,label):
 return call(mount,'write_file',dict(path=relative,content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True),label)
def publish_json(mount,path,obj,label):
 code='''from pathlib import Path
import os,json,hashlib
p=Path(PARAM['path']);b=(json.dumps(PARAM['value'],indent=2)+chr(10)).encode()
if p.exists():assert p.read_bytes()==b
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
directory=p.parent
while True:
 fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if directory.parent==directory:break
 directory=directory.parent
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))
'''
 return remote(mount,code,dict(path=path,value=obj),label)
def load_campaign():
 d=json.loads((B/'campaign.json').read_text());assert d['plan_sha256']==plan_sha and d['absolute_deadline_unix']==d['started_unix']+plan['cap_seconds']
 return d
if a.action=='deploy':
 assert not (B/'deployment-intent.json').exists()
 manifest=json.loads((B/'deployment-manifest.json').read_text());payload=Path(manifest['archive']['path']).read_bytes();assert sha(payload)==manifest['archive']['sha256'];parts=[payload[i:i+65536] for i in range(0,len(payload),65536)];save('deployment-intent.json',dict(manifest=manifest,at_epoch=time.time(),no_model=True))
 refs=[]
 for i,data in enumerate(parts):
  name='part-%03d'%i;refs.append(dict(name=name,bytes=len(data),sha256=sha(data)));write(source,plan['remote_relative']+'/'+name,data,'deploy-part-%03d-tool.json'%i)
 code='''from pathlib import Path
import os,hashlib,json,tarfile
p=Path(PARAM['root']);assert os.getuid()==1045 and p.stat().st_uid==1045;os.chmod(p,0o700);m=PARAM['manifest'];parts=PARAM['parts']
assert not (p/'campaign.json').exists()
for r in parts:
 b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
archive=p/'deployment.tar.gz'
with archive.open('xb') as f:
 for r in parts:f.write((p/r['name']).read_bytes())
 f.flush();os.fsync(f.fileno())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['archive']['sha256']
with tarfile.open(archive) as t:
 assert len(t.getnames())==len(m['files']) and set(t.getnames())==set(m['files'])
 for item in t.getmembers():
  assert item.isfile() and item.name==Path(item.name).name;b=t.extractfile(item).read();r=m['files'][item.name];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  path=p/item.name
  with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  os.chmod(path,0o600)
  if path.suffix=='.py':compile(path.read_text(),item.name,'exec')
# Decompress only verified private training inputs; never truncate or mutate source.
import gzip
for gzname,item in PARAM['decompress'].items():
 b=gzip.decompress((p/gzname).read_bytes());assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
 with (p/item['name']).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
shm=Path(PARAM['shm']);assert not shm.exists();shm.mkdir(mode=0o700)
directory=p
while True:
 fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if directory.parent==directory:break
 directory=directory.parent
print(json.dumps(dict(status='ACTUAL_SOURCE_DATA_SHA_COMPILE_FSYNC_DEPLOYED',plan_sha256=hashlib.sha256((p/'controller-plan.json').read_bytes()).hexdigest(),files=len(m['files']),private_shm=str(shm))))
'''
 actual=remote(source,code,dict(root=root,manifest=manifest,parts=refs,decompress=plan['decompress'],shm=str(Path(plan['output_dir']).parent)),'source-deploy-tool.json',60);save('source-deployment.json',actual)
 # Only executable CPU fetch/controller and public config copied to persistent host.
 for name in ['fetch_checkpoint.py','io_driver.py','controller-plan.json']:
  raw=(B/name).read_bytes();write(target,plan['persistent_remote_relative']+'/'+name,raw,'target-deploy-'+name+'.json')
 code='''from pathlib import Path
import hashlib,json,os,shutil
p=Path(PARAM['root']);assert os.getuid()==1006 and p.stat().st_uid==1006;os.chmod(p,0o700);assert shutil.disk_usage(p).free>=2*1024**3
for n,r in PARAM['files'].items():
 f=p/n;assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
 with f.open('rb') as h:os.fsync(h.fileno())
 if f.suffix=='.py':compile(f.read_text(),n,'exec')
directory=p
while True:
 fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if directory.parent==directory:break
 directory=directory.parent
print(json.dumps(dict(status='ACTUAL_PERSISTENT_FETCH_SOURCE_FSYNC',free=shutil.disk_usage(p).free)))
'''
 files={n:dict(bytes=(B/n).stat().st_size,sha256=sha((B/n).read_bytes())) for n in ['fetch_checkpoint.py','io_driver.py','controller-plan.json']};actual=remote(target,code,dict(root=dest,files=files),'target-deploy-verify.json');save('target-deployment.json',actual);print(json.dumps(actual))
elif a.action=='preflight':
 assert not (B/'preflight-intent.json').exists();save('preflight-intent.json',dict(at_epoch=time.time(),no_model=True,plan_sha256=plan_sha))
 code='''from pathlib import Path
import hashlib,json,os,subprocess,time
p=Path(PARAM['root']);plan=json.loads((p/'controller-plan.json').read_text());model=Path(plan['model']);manifest=json.loads((p/'model-source-manifest.json').read_text());actual=[]
for item in manifest['files']:
 f=model/item['name'];h=hashlib.sha256()
 with f.open('rb') as reader:
  for block in iter(lambda:reader.read(4*1024*1024),b''):h.update(block)
 assert f.stat().st_size==item['size'] and h.hexdigest()==item['sha256'];actual.append(dict(name=item['name'],bytes=item['size'],sha256=item['sha256']))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
row=[s for s in gpu.splitlines() if s.split(',')[1].strip()==plan['gpu_uuid']];assert len(row)==1;idx,uuid,used,free,total=[x.strip() for x in row[0].split(',')];assert idx=='2' and int(used)<=100 and int(free)>=78000 and int(total)==81920 and uuid not in apps
shm=Path(plan['output_dir']).parent;assert shm.stat().st_uid==1045 and not Path(plan['output_dir']).exists();assert os.statvfs(shm).f_bavail*os.statvfs(shm).f_frsize>=4*1024**3
print(json.dumps(dict(status='ACTUAL_FRESH_GPU_BASE_SHA_SHM',at_epoch=time.time(),files=actual,gpu=row[0],no_process=True,plan_sha256=hashlib.sha256((p/'controller-plan.json').read_bytes()).hexdigest(),shm_free=os.statvfs(shm).f_bavail*os.statvfs(shm).f_frsize)))
'''
 d=remote(source,code,dict(root=root),'preflight-tool.json',60);save('preflight.json',d);print(json.dumps({k:v for k,v in d.items() if k!='files'}))
elif a.action=='start-epoch':
 assert not (B/('epoch-%03d-start-intent.json'%epoch)).exists()
 if epoch==1:
  proof=json.loads((B/'preflight.json').read_text());assert proof['plan_sha256']==plan_sha and time.time()-proof['at_epoch']<300
  assert not (B/'campaign.json').exists();started=time.time();campaign=dict(plan_sha256=plan_sha,cap_seconds=plan['cap_seconds'],grace_seconds=30,started_unix=started,absolute_deadline_unix=started+plan['cap_seconds'],no_budget_reset=True);save('campaign.json',campaign)
  publish_json(source,root+'/campaign.json',campaign,'source-campaign.json');publish_json(target,dest+'/campaign.json',campaign,'target-campaign.json')
 assert time.time()<campaign['absolute_deadline_unix'];save('epoch-%03d-start-intent.json'%epoch,dict(at_epoch=time.time(),plan_sha256=plan_sha,campaign=campaign,epoch=epoch))
 argv=[plan['python'],'-u',root+'/epoch_driver.py','--plan',root+'/controller-plan.json','--campaign',root+'/campaign.json','--epoch',str(epoch)]
 d=call(source,'start_process',dict(command=shlex.join(argv),cwd=plan['remote_relative'],env=plan['env']),'epoch-%03d-start-tool.json'%epoch);save('epoch-%03d-job.json'%epoch,d);print(json.dumps(d))
elif a.action=='inspect':
 stamp=str(time.time_ns());code='''from pathlib import Path
import json
p=Path(PARAM['root']);out={};output=Path(PARAM['output'])
for epoch in [1]:
 c=p/'control'/('epoch-%03d'%epoch);d={}
 for n in ['driver-started.json','wrapper-complete.json','gpu-release.json']:
  if (c/n).is_file():d[n]=json.loads((c/n).read_text())
 if (c/'stdout.log').is_file():d['stdout_tail']=(c/'stdout.log').read_text()[-2000:]
 out[str(epoch)]=d
for n in ['progress.json','training_complete.json','failure.json']:
 if (output/n).is_file():out[n]=json.loads((output/n).read_text())
print(json.dumps(out))
'''
 d=remote(source,code,dict(root=root,output=plan['output_dir']),'inspect-'+stamp+'-tool.json');save('inspect-'+stamp+'.json',d)
 for ep in [1]:
  value=d.get(str(ep),{});wrapper=value.get('wrapper-complete.json');release=value.get('gpu-release.json');path=B/('epoch-%03d-state.json'%ep)
  if wrapper and release and not path.exists():save(path.name,dict(wrapper=wrapper,release=release))
 print(json.dumps(d))
elif a.action=='start-server':
 campaign=load_campaign();state=json.loads((B/('epoch-%03d-state.json'%epoch)).read_text());assert state['wrapper']['returncode']==0 and not state['wrapper'].get('timed_out') and 'endpoint_error' not in state['wrapper'] and not state['release']['target_compute_rows']
 name='epoch-%03d-export-intent.json'%epoch;assert not (B/name).exists();started=time.time();deadline=min(started+600,campaign['absolute_deadline_unix']);assert deadline>started
 entry=dict(phase='server',epoch=epoch,plan_sha256=plan_sha,started_unix=started,absolute_deadline_unix=deadline);save(name,entry);publish_json(source,root+'/'+name,entry,'epoch-%03d-server-entry-copy.json'%epoch)
 argv=[plan['python'],'-u',root+'/io_driver.py','--plan',root+'/controller-plan.json','--campaign',root+'/campaign.json','--entry',root+'/'+name]
 d=call(source,'start_process',dict(command=shlex.join(argv),cwd=plan['remote_relative'],env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}),'epoch-%03d-server-start-tool.json'%epoch);save('epoch-%03d-server-job.json'%epoch,d);print(json.dumps(d))
elif a.action=='start-fetch':
 campaign=load_campaign();entry=json.loads((B/('epoch-%03d-export-intent.json'%epoch)).read_text());assert time.time()<entry['absolute_deadline_unix'];assert not (B/('epoch-%03d-fetch-intent.json'%epoch)).exists()
 code='''from pathlib import Path
import json
p=Path(PARAM['root'])/'control'/('export-server-epoch-%03d'%PARAM['epoch']);ready=None
if (p/'stdout.log').is_file():ready=next((json.loads(s) for s in (p/'stdout.log').read_text().splitlines() if s.startswith('{')),None)
print(json.dumps(dict(ready=ready,complete=json.loads((p/'wrapper-complete.json').read_text()) if (p/'wrapper-complete.json').is_file() else None)))
'''
 d=remote(source,code,dict(root=root,epoch=epoch),'epoch-%03d-server-ready-%s-tool.json'%(epoch,time.time_ns()));ready=d['ready']
 if ready is None:print(json.dumps(d));raise SystemExit(0)
 assert ready['ready'] and ready['manifest']['completed_epochs']==epoch and ready['manifest']['global_step']==plan['configuration']['steps_per_epoch']*epoch
 state=json.loads((B/('epoch-%03d-state.json'%epoch)).read_text());assert ready['manifest']['checkpoint_sha256']==state['wrapper']['checkpoint_sha256']
 save('epoch-%03d-server-ready.json'%epoch,ready);entry={**entry,'phase':'fetch','url':'http://'+plan['source_private_IP']+':'+str(ready['port']),'manifest_sha256':ready['manifest_sha256']};save('epoch-%03d-fetch-intent.json'%epoch,entry)
 name='epoch-%03d-fetch-entry.json'%epoch;publish_json(target,dest+'/'+name,entry,'epoch-%03d-fetch-entry-copy.json'%epoch)
 argv=[plan['persistent_python'],'-u',dest+'/io_driver.py','--plan',dest+'/controller-plan.json','--campaign',dest+'/campaign.json','--entry',dest+'/'+name]
 d=call(target,'start_process',dict(command=shlex.join(argv),cwd=plan['persistent_remote_relative'],env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}),'epoch-%03d-fetch-start-tool.json'%epoch);save('epoch-%03d-fetch-job.json'%epoch,d);print(json.dumps(d))
else:
 assert not (B/('epoch-%03d-export-ack.json'%epoch)).exists();stamp=str(time.time_ns())
 code='''from pathlib import Path
import hashlib,json,os
root=Path(PARAM['root']);control=root/'control'/('export-fetch-epoch-%03d'%PARAM['epoch']);wrapper=control/'wrapper-complete.json'
if not wrapper.exists():print(json.dumps(dict(complete=False)))
else:
 w=json.loads(wrapper.read_text());assert w['returncode']==0 and not w.get('timed_out')
 cp=Path(PARAM['checkpoint']);p=cp/('export-ack-epoch-%03d.json'%PARAM['epoch']);raw=p.read_bytes();ack=json.loads(raw)
 assert ack['status']=='persistent_checkpoint_verified' and ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files'] and ack['destination_directories_fsynced']
 for n,item in ack['files'].items():
  f=cp/n;assert f.stat().st_size==item['bytes'];h=hashlib.sha256()
  with f.open('rb') as reader:
   os.fsync(reader.fileno())
   for b in iter(lambda:reader.read(4*1024*1024),b''):h.update(b)
  assert h.hexdigest()==item['sha256']
 print(json.dumps(dict(complete=True,ack=ack,ack_sha256=hashlib.sha256(raw).hexdigest(),wrapper=w,all_files_reopened_SHA=True)))
'''
 d=remote(target,code,dict(root=dest,epoch=epoch,checkpoint=plan['persistent_checkpoint_root']+'/epoch-%03d'%epoch),'epoch-%03d-export-collect-%s-tool.json'%(epoch,stamp),60)
 if not d['complete']:print(json.dumps(d));raise SystemExit(0)
 ready=json.loads((B/('epoch-%03d-server-ready.json'%epoch)).read_text());ack=d['ack'];assert ack['files']==ready['manifest']['files'] and ack['transport_manifest_sha256']==ready['manifest_sha256'] and ack['completed_epochs']==epoch and ack['global_step']==plan['configuration']['steps_per_epoch']*epoch
 save('epoch-%03d-export-ack.json'%epoch,ack);save('epoch-%03d-export-verification.json'%epoch,d)
 published=publish_json(source,plan['output_dir']+'/export-ack-epoch-%03d.json'%epoch,ack,'epoch-%03d-source-ack-tool.json'%epoch);save('epoch-%03d-source-ack-copy.json'%epoch,published)
 # The bounded server observes its exact source ACK and exits itself; no assumed recursive stop_process cleanup.
 if epoch==1:save('campaign-complete.json',dict(status='ONE_COMPLETE_EPOCH_DURABLE_ACK_VERIFIED',at_epoch=time.time(),campaign=load_campaign(),epoch1_ack_sha256=sha((B/'epoch-001-export-ack.json').read_bytes()),no_quality=True))
 print(json.dumps(dict(complete=True,epoch=epoch,bytes=ack['total_bytes'],seconds=ack['seconds'],ack_sha256=d['ack_sha256'],persistent_directory=ack['persistent_directory'])))
