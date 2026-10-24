"""Read-only four copied epoch members; no model/process mutation."""
from pathlib import Path
import hashlib,json,shlex,sys,time,os
sys.path.insert(0,'/workspace/patientphex-tools')
from mcp_client import Client
D=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1/probe-path-recovery-v1')
receipt=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/F0-copy-v1/copy-complete.json')
proof=json.loads(receipt.read_text())['ack']
spec=dict(root=proof['destination'],files=proof['files'])
code='SPEC='+repr(spec)+'''\nfrom pathlib import Path
import hashlib,json,os,time
root=Path(SPEC['root']);actual={}
for name,expected in SPEC['files'].items():
 p=root/name;assert p.is_file() and not p.is_symlink();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 actual[name]=dict(bytes=p.stat().st_size,sha256=h.hexdigest());assert actual[name]==expected
print(json.dumps(dict(at_unix=time.time(),root=str(root),actual_model_load_path=str(root/'adapter'),actual=actual,no_GPU=True,no_mutation=True)))
'''
c=Client();c.initialize();r=c.request('tools/call',dict(name='run_command',arguments=dict(mount_id='69b991ee-183f-4add-af1a-579af92479c9',command='python3 -c '+shlex.quote(code),timeout_seconds=20)))
assert not r.get('isError');a=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert a['exit_code']==0
out=json.loads(a['stdout']);out['source']=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());p=D/'actual-four-file-inventory.json'
with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),at_unix=out['at_unix'],four_files=len(out['actual']))))
