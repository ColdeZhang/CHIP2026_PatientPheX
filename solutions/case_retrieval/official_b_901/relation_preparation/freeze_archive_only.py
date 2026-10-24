"""Freeze one completed old-runner archive; small metadata only, no generation.

Reuses the exact original archive body from controller7a170; changes stdout to
return an independently reopened archive/manifest inventory. Download is separate.
"""
import argparse,ast,base64,hashlib,json,os,shlex,sys
from pathlib import Path
PIN='7a170599af196e78b99164534de23acd3bca6201f99d5767de67145892d84c4e'

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def unpack(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();d=a.directory.resolve();dep=json.loads((d/'deployment.json').read_text())
 original=Path(__file__).with_name('run_arm_once.py');assert sha(original.read_bytes())==PIN
 strings=[n.value for n in ast.walk(ast.parse(original.read_text())) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'FROZEN_TERMINAL_ARCHIVE' in n.value];assert len(strings)==1;code=strings[0]
 old_print="print(json.dumps(dict(bytes=manifest.stat().st_size,sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))"
 assert code.count(old_print)==1;code=code.replace(old_print,'').replace('ABSOLUTE',repr(dep['remote_absolute']))
 before="""import json
from pathlib import Path
pre=Path(REMOTE)/'terminal.json';assert pre.is_file();state=json.loads(pre.read_text());assert state['exit_code']==0 and not state['timeout_triggered']
""".replace('REMOTE',repr(dep['remote_absolute']))
 after="""
from datetime import datetime,timezone
inventory={'at_utc':datetime.now(timezone.utc).isoformat(),'files':{}}
for name in ['terminal-artifacts.tar.gz','terminal-artifacts.manifest.json']:
 q=p/name;s=q.stat();inventory['files'][name]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
print(json.dumps(inventory))
"""
 assert not (d/'archive-manifest.json').exists();save(d/'archive-freeze-intent.json',dict(source_sha256=sha(Path(__file__).read_bytes()),original_controller_sha256=PIN,remote=dep['remote_absolute'],body_sha256=sha((before+code+after).encode()),no_model_or_new_inference=True))
 sys.path.insert(0,dep['mcp_scripts']);from mcp_client import Client
 c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':dep['mount_id'],'command':'python3 -c '+shlex.quote(before+code+after),'timeout_seconds':45}});save(d/'archive-freeze-inventory.json',r)
 assert not r.get('isError');u=unpack(r);assert u['exit_code']==0;inv=json.loads(u['stdout']);expected=inv['files']['terminal-artifacts.manifest.json']
 r=c.request('tools/call',{'name':'read_file','arguments':{'mount_id':dep['mount_id'],'path':dep['remote_relative']+'/terminal-artifacts.manifest.json','offset':0,'limit':expected['bytes']}})
 if r.get('isError'):save(d/'archive-manifest-read-error.json',r);raise RuntimeError('Manifest transport failed; existing remote archive retained, no retry')
 u=unpack(r);b=base64.b64decode(u['content']) if u.get('encoding')=='base64' else u['content'].encode();assert len(b)==expected['bytes'] and sha(b)==expected['sha256'];m=json.loads(b)
 assert m['bytes']==inv['files']['terminal-artifacts.tar.gz']['bytes'] and m['sha256']==inv['files']['terminal-artifacts.tar.gz']['sha256']
 with (d/'archive-manifest.json').open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(dict(status='ARCHIVE_AND_MANIFEST_FROZEN_REOPENED_METADATA_RECOVERED',archive_bytes=m['bytes'],archive_sha256=m['sha256'],members=len(m['members']),exit_code=m['terminal_exit_code'])))
if __name__=='__main__':main()
