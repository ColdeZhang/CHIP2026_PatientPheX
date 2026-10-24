"""Read and fsync an exact existing full64 CPU deployment after unknown transport.
No archive fetch, content write, model start, or process mutation.
"""
from pathlib import Path
import argparse,json,hashlib,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();D=a.directory;plan=json.loads((D/'plan.json').read_text());m=json.loads((D/'deployment-manifest.json').read_text());stamp=str(time.time_ns());root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+plan['remote']
def save(name,x):
 with (D/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
code='''from pathlib import Path
import hashlib,json,os,tarfile
p=Path(P['root']);m=P['manifest'];out={'status':'OBSERVED_INCOMPLETE_OR_ABSENT_DEPLOYMENT','existing_files':{},'missing':[],'no_content_write':True,'no_model':True}
assert not (p/'driver-started.json').exists()
for n,r in m['files'].items():
 q=p/n
 if not q.exists():out['missing'].append(n);continue
 assert not q.is_symlink();b=q.read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];out['existing_files'][n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 if q.suffix=='.py':compile(q.read_text(),n,'exec')
 with q.open('rb') as f:os.fsync(f.fileno())
a=p/'deployment.tar.gz'
if not a.exists():out['missing'].append('deployment.tar.gz')
else:
 b=a.read_bytes();assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256']
 with tarfile.open(a) as t:
  assert set(t.getnames())==set(m['files'])
  for n,r in m['files'].items():b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 with a.open('rb') as f:os.fsync(f.fileno())
if not out['missing']:
 assert os.getuid()==1045 and p.stat().st_uid==1045
 fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
 out.update(status='ACTUAL_ALL_SOURCE_INPUT_TRAINING_ACK_SHA_COMPILE_FSYNC',files=len(m['files']),pieces=(m['archive']['bytes']+32767)//32768,plan_sha256=hashlib.sha256((p/'plan.json').read_bytes()).hexdigest(),uid=os.getuid(),no_model_loaded=True,adopted_existing_exact_deployment=True)
print(json.dumps(out))
'''
save('deployment-recovery-'+stamp+'-intent.json',dict(at_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_unknown_transport_preserved=True,no_content_write=True))
c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':plan['mount_id'],'command':'python3 -c '+shlex.quote('P='+repr(dict(root=root,manifest=m))+'\n'+code),'timeout_seconds':30}});save('deployment-recovery-'+stamp+'-tool.json',r)
assert not r.get('isError'),str(r);v=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert v['exit_code']==0,v.get('stderr');x=json.loads(v['stdout']);save('deployment-recovery-'+stamp+'.json',x)
if x['status']=='ACTUAL_ALL_SOURCE_INPUT_TRAINING_ACK_SHA_COMPILE_FSYNC':
 assert x['plan_sha256']==hashlib.sha256((D/'plan.json').read_bytes()).hexdigest();save('deployment-result.json',x)
print(json.dumps({k:v for k,v in x.items() if k!='existing_files'}))
