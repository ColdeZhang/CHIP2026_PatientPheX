"""Private .100 CPU-only terminal backup; no GPU, training, or quality."""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,sys,tarfile,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE='research/patient-response-code/backups/F0-held58-terminal-v1'
ABS='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REMOTE

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();D=a.directory;m=json.loads((D/'archive-manifest.json').read_text());payload=Path(m['archive']['path']).read_bytes();assert sha(payload)==m['archive']['sha256'] and len(payload)==m['archive']['bytes'];assert not (D/'upload-intent.json').exists()
 def save(n,x):
  with (D/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 save('upload-intent.json',dict(at_unix=time.time(),manifest_sha256=sha((D/'archive-manifest.json').read_bytes()),remote=ABS,no_GPU=True));client=Client();client.initialize()
 def call(n,params,label):
  r=client.request('tools/call',dict(name=n,arguments=dict(mount_id=MOUNT,**params)));save(label,r)
  if r.get('isError'):raise RuntimeError('Unknown/error: inspect frozen original remote parts; do not overwrite different bytes')
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 parts=[]
 for i in range(0,len(payload),32768):
  b=payload[i:i+32768];name='part-%09d'%i;parts.append(dict(name=name,bytes=len(b),sha256=sha(b)))
  call('write_file',dict(path=REMOTE+'/'+name,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True),name+'-tool.json')
 params=dict(root=ABS,parts=parts,manifest=m)
 code='''from pathlib import Path
import os,json,hashlib,tarfile,time
p=Path(PARAM['root']);assert os.getuid()==1006 and p.stat().st_uid==1006;os.chmod(p,0o700)
for r in PARAM['parts']:
 b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
target=p/'terminal.tar.gz';assert not target.exists()
with target.open('xb') as f:
 for r in PARAM['parts']:f.write((p/r['name']).read_bytes())
 f.flush();os.fsync(f.fileno())
b=target.read_bytes();m=PARAM['manifest'];assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256']
with tarfile.open(target) as t:
 assert set(t.getnames())==set(m['members'])
 for n,r in m['members'].items():
  item=t.getmember(n);assert item.isfile() and not item.issym() and '..' not in Path(n).parts and not Path(n).is_absolute();b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
receipt=dict(status='PERSISTENT_PRIVATE_TERMINAL_ARCHIVE_FSYNC_REOPEN_ALL_SHA',at_unix=time.time(),path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),members=len(m['members']),no_model_or_optimizer_state=True,private_clinical_inputs_and_predictions=True,no_held_gold=True)
with (p/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(receipt))
'''
 r=call('run_command',dict(command='python3 -c '+shlex.quote('PARAM='+repr(params)+'\n'+code),timeout_seconds=60),'verify-tool.json');assert r['exit_code']==0;receipt=json.loads(r['stdout']);save('stored-receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
