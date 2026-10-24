"""Short read-only recovery of the existing UNKNOWN probe deployment, never start."""
import hashlib,json,os,shlex,sys,time
from pathlib import Path
import probe_remote as ops

D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/probe-v1')
cfg=json.loads((D/'package.json').read_text());t=D/'transport';prefix=t/('short-inventory-'+str(time.time_ns()))
code="""from pathlib import Path
import hashlib,json
root=Path(ROOT)
def ref(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
r=root/'deployment-complete.json'
value=dict(receipt=json.loads(r.read_text()) if r.exists() else None,markers=[n for n in ['start-intent.json','driver-started.json','child-started.json','wrapper-complete.json','stdout.log'] if (root/n).exists()],members={})
for name in NAMES:
 p=root/name
 if p.exists():value['members'][name]=ref(p)
print(json.dumps(value))
"""
command='python3 -c '+shlex.quote('ROOT='+repr(ops.ROOT)+'\nNAMES='+repr(sorted(cfg['members']))+'\n'+code)
args=dict(mount_id=ops.MOUNT,cwd='.',command=command,timeout_seconds=20)
ops.save(prefix.with_suffix('.intent.json'),dict(tool='run_command',arguments=args,read_only=True))
sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
c=Client();c.initialize()
try:r=c.request('tools/call',dict(name='run_command',arguments=args))
except Exception as e:
 ops.save(prefix.with_suffix('.error.json'),dict(UNKNOWN=True,type=type(e).__name__));raise RuntimeError('Original unchanged; no start called') from None
text=json.dumps(r)
for k in ['MCP_TOKEN','MCP_URL']:
 if os.environ.get(k):text=text.replace(os.environ[k],'[REDACTED]')
r=json.loads(text)
if r.get('isError'):
 ops.save(prefix.with_suffix('.error.json'),r)
 if 'LUNA_OFFLINE' in text:ops.save(t/('offline-backoff-'+str(time.time_ns())+'.json'),dict(not_before_unix=time.time()+185))
 raise RuntimeError('Read-only recovery failed; preserve actual error')
v=r.get('structuredContent')
if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
ops.save(prefix.with_suffix('.response.json'),v);assert v['exit_code']==0
value=json.loads(v['stdout']);assert not value['markers']
assert value['members']=={n:ops.identity(i) for n,i in cfg['members'].items()}
receipt=value['receipt'];assert receipt['package_sha256']==ops.ref(D/'package.json')['sha256'] and receipt['SHA_fsync_reopen']
ops.save_exact(D/'deployment.json',dict(actual=receipt,package=ops.ref(D/'package.json')))
print(json.dumps(dict(original_deployment_complete=True,source=ops.ref(__file__),read_only=True,no_start=True)))
