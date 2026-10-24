"""Read-only proof the original bounded four-file server and listener ended."""
import hashlib,json,os,shlex,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/copy-A30-v1')
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/grounded-sampled-consensus-v1/copy-A30-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 started=json.load(open(D/'serve-started.json'));ready=json.load(open(D/'source-ready.json'))['ready'];intent=json.load(open(D/'serve-intent.json'))
 assert time.time()>intent['at_unix']+190
 spec=dict(root=ROOT,pid=started['pid'],port=ready['port'])
 code='SPEC='+repr(spec)+'''\nfrom pathlib import Path
import os,json,time
root=Path(SPEC['root']);matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0));cwd=(p/'cwd').resolve()
  if cwd==root and any(x in ['adapter_copy.py',str(root/'adapter_copy.py')] for x in argv):
   matches.append(dict(pid=int(p.name),argv=argv,cwd=str(cwd),stat=(p/'stat').read_text()))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
listeners=[]
for f in ['/proc/net/tcp','/proc/net/tcp6']:
 for line in Path(f).read_text().splitlines()[1:]:
  fields=line.split()
  if fields[3]=='0A' and int(fields[1].split(':')[-1],16)==SPEC['port']:listeners.append(line)
print(json.dumps(dict(at_unix=time.time(),original_managed_pid_exists=Path('/proc/'+str(SPEC['pid'])).exists(),matching_original_namespace_processes=matches,original_port_listeners=listeners,read_only=True,closed=not matches and not listeners)))
'''
 stamp=D/('closure-'+str(time.time_ns()));save(stamp.with_suffix('.intent.json'),dict(source=ref(__file__),original_start=ref(D/'serve-started.json'),original_ready=ref(D/'source-ready.json'),spec=spec))
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize();r=c.request('tools/call',dict(name='run_command',arguments=dict(mount_id=started['mount_id'],cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=15)))
 text=json.dumps(r)
 for k in ['MCP_URL','MCP_TOKEN']:
  if os.environ.get(k):text=text.replace(os.environ[k],'[REDACTED]')
 r=json.loads(text);save(stamp.with_suffix('.tool.json'),r);assert not r.get('isError')
 v=r.get('structuredContent')
 if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 assert v['exit_code']==0;x=json.loads(v['stdout']);assert x['closed']
 save(D/'source-service-closed.json',dict(actual=x,source=ref(__file__),tool=ref(stamp.with_suffix('.tool.json')),original_start=ref(D/'serve-started.json'),no_signal_sent=True,no_GPU_or_model=True));print(json.dumps(ref(D/'source-service-closed.json')))
if __name__=='__main__':main()
