from pathlib import Path
import sys,json,shlex,time,zipfile,hashlib,base64,io
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
D=Path(__file__).parent;S=Path('/workspace/patientphex-worktrees/explore-patient-competition-calibration/solutions/patient_competition_calibration');P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';rel='research/patient-competition-calibration-v1';R=P+'/'+rel;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';c=Client();c.initialize()
def call(n,a,label):
 v=c.request('tools/call',{'name':n,'arguments':{'mount_id':M,**a}});(D/(label+'.json')).write_text(json.dumps(v,indent=2));assert not v.get('isError'),v;return v.get('structuredContent') or json.loads(v['content'][0]['text'])
files={}
for folder in ['calibration-v1','assembled-v1','quality-v1']:
 for p in (D/folder).rglob('*'):
  if p.is_file():files['quality/'+str(p.relative_to(D))]=p.read_bytes()
for name in ['quality-invocation.json','collection-receipt.json','terminal-ready.json','cpu-complete.json','start-intent.json']:files['quality/'+name]=(D/name).read_bytes()
files['previousPCL/terminal-manifest.json']=(D.parent/'patient-competition-loss-v1/terminal-manifest.json').read_bytes()
for p in S.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.md']:files['source/'+str(p.relative_to(S))]=p.read_bytes()
manifest={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()};buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for n,b in files.items():z.writestr(n,b)
 z.writestr('inventory.json',json.dumps(manifest,indent=2))
b=buf.getvalue();p=D/'quality-evidence.zip';p.write_bytes(b);expected={'path':R+'/quality-evidence.zip','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'members':len(manifest)};(D/'quality-backup-intent.json').write_text(json.dumps(expected,indent=2)+'\n');x=call('write_file',{'path':rel+'/quality-evidence.zip','content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True},'quality-upload-response');assert x['sha256']==expected['sha256']
code="""from pathlib import Path
import os,json,hashlib,zipfile
p=Path(%r);want=%r
with p.open('rb') as f:os.fsync(f.fileno())
fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);assert p.stat().st_size==want['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==want['sha256']
with zipfile.ZipFile(p) as z:
 inv=json.loads(z.read('inventory.json'));assert len(inv)==want['members']
 for n,r in inv.items():b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
print(json.dumps({'status':'QUALITY_ARCHIVE_FILE_PARENT_FSYNC_ALL_MEMBERS_SHA_PASS','archive':want}))
"""%(R+'/quality-evidence.zip',expected);x=call('run_command',{'command':'python3 -c '+shlex.quote(code),'timeout_seconds':30},'quality-storage-response');assert x['exit_code']==0
chunks=[]
for offset in range(0,len(b),524288):
 size=min(524288,len(b)-offset);x=call('read_file',{'path':rel+'/quality-evidence.zip','offset':offset,'limit':size},'quality-readback-'+str(offset));chunk=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(chunk)==size;chunks.append(chunk)
assert b''.join(chunks)==b;out={'status':'QUALITY_ARCHIVE_REMOTE_FILE_PARENT_FSYNC_AND_FULL_READBACK_EXACT','archive':expected,'all_members_verified':True,'full_readback_bytes':len(b),'checkpoint_loaded':json.loads((D/'collected/run/native/loaded.json').read_text())};(D/'quality-backup-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'archive':expected}),flush=True)
