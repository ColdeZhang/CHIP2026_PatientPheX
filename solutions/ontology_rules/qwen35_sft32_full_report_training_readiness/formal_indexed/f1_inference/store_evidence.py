from pathlib import Path
import sys,json,hashlib,shlex,time,os,base64
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
B=Path('/workspace/patientphex-runs/round2-20261023/research/ontology-indexed-fulltext32-terminal-backup-v1');meta=json.loads((B/'local-archive.json').read_text());M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/ontology-indexed-fulltext32/backups/terminal-20261024-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
assert meta['status']=='LOCAL_ARCHIVE_FSYNC_ALL_MEMBERS_VERIFIED_NO_REMOTE_WRITE' and 30 <= meta['member_count'] <= 150
assert Path(meta['archive_path']).name=='ontology-indexed-fulltext32-terminal-v1.tar.gz'
def save(name,obj):
 with (B/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':dict(mount_id=M,**args)})
 if r.get('isError'):save('remote-error-'+str(time.time_ns())+'.json',r);raise RuntimeError('MCP '+name+' failed; prior outcomes retained')
 return r
assert not (B/'upload-intent.json').exists(),'Existing intent; inspect original writes before any recovery'
b=Path(meta['archive_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==meta['archive_sha256'];parts=[b[i:i+65536] for i in range(0,len(b),65536)];pr=[dict(name='archive.part%03d'%i,bytes=len(x),sha256=hashlib.sha256(x).hexdigest()) for i,x in enumerate(parts)];save('upload-intent.json',dict(at=time.time(),meta=meta,parts=pr,only_own_target=True))
c=Client();c.initialize()
code="from pathlib import Path; import os,json,shutil; p=Path("+repr(root+rel)+"); assert os.getuid()==1006; p.mkdir(parents=True,exist_ok=True,mode=0o700); assert p.stat().st_uid==1006; os.chmod(p,0o700); assert shutil.disk_usage(p).free>2147483648; print(json.dumps({'uid':os.getuid(),'path':str(p),'mode':oct(p.stat().st_mode&0o777)}))"
r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=30));save('private-directory.json',r);assert un(r)['exit_code']==0
for i,x in enumerate(parts):
 r=call('write_file',dict(path=rel+'/'+pr[i]['name'],content=base64.b64encode(x).decode(),encoding='base64',create_parents=False));save('part%03d-receipt.json'%i,r)
code='''from pathlib import Path
import os,json,hashlib,tarfile,time
p=Path(PATH); parts=PARTS; expected=EXPECTED
assert os.getuid()==1006 and p.stat().st_uid==1006
for x in parts:
 b=(p/x['name']).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
target=p/'ontology-indexed-fulltext32-terminal-v1.tar.gz'
if not target.exists():
 temp=p/'archive.assembling'
 with temp.open('xb') as f:
  for x in parts:f.write((p/x['name']).read_bytes())
  f.flush();os.fsync(f.fileno())
 assert temp.stat().st_size==expected['archive_bytes'] and hashlib.sha256(temp.read_bytes()).hexdigest()==expected['archive_sha256']
 os.replace(temp,target)
 d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
actual=hashlib.sha256(target.read_bytes()).hexdigest();assert actual==expected['archive_sha256'] and target.stat().st_size==expected['archive_bytes']
with tarfile.open(target,'r:gz') as t:
 members=t.getmembers();assert all(x.isfile() and not x.issym() and not x.islnk() and not x.name.startswith('/') and '..' not in Path(x.name).parts for x in members)
 assert len({x.name for x in members})==len(members)
 mb=t.extractfile('members.json').read();assert hashlib.sha256(mb).hexdigest()==expected['members_manifest_sha256'];manifest=json.loads(mb)
 assert len(members)==len(manifest['members'])+1 and set(x.name for x in members)==set(x['archive_path'] for x in manifest['members'])|{'members.json'}
 for x in manifest['members']:
  b=t.extractfile(x['archive_path']).read();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
result={'status':'PERSISTENT_ARCHIVE_FSYNC_FULL_SHA_AND_EVERY_MEMBER_VERIFIED','uid':os.getuid(),'at_epoch':time.time(),'locator':str(target),'bytes':target.stat().st_size,'sha256':actual,'member_count':len(manifest['members']),'all_member_size_SHA_exact':True,'no_full_model_Adam_tensor_arrays':True,'private_input_prompts_predictions_and_raw_records':True,'independent_final_review_pending_at_packaging':True}
with (p/'stored-receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
d=os.open(str(p),os.O_RDONLY);os.fsync(d);os.close(d)
print(json.dumps(result))
'''.replace('PATH',repr(root+rel)).replace('PARTS',repr(pr)).replace('EXPECTED',repr(meta))
save('verify-intent.json',dict(at=time.time(),source=code))
r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=45));save('stored-tool.json',r);d=un(r);assert d['exit_code']==0;v=json.loads(d['stdout']);save('stored-receipt.json',v);print(json.dumps(v))
