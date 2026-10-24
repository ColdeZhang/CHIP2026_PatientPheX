"""CPU-only deploy/start/observe with durable local intents; no automatic start replay."""
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
from pathlib import Path
from synthesize import ref,save
MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL='grounded-weight-average-v1/F1/cpu-v1';ROOT=PROJECT+'/'+REL
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['deploy','start','observe']);ap.add_argument('--directory',required=True,type=Path);a=ap.parse_args()
a.directory.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
cl=Client();cl.initialize()
def call(name,args):
 tag=str(time.time_ns());safe=dict(args)
 if 'content' in safe:safe['content']='[fixed source archive omitted]'
 save(a.directory/(tag+'-intent.json'),dict(tool=name,args=safe))
 try:r=cl.request('tools/call',dict(name=name,arguments=args))
 except BaseException as e:
  save(a.directory/(tag+'-unknown.json'),dict(type=type(e).__name__,UNKNOWN=True,no_retry=True));raise
 save(a.directory/(tag+'-response.json'),r)
 assert not r.get('isError'),'Original response retained; recover read-only, do not replay start'
 return r.get('structuredContent') or json.loads(next(v['text'] for v in r['content'] if v['type']=='text'))
def command(code):
 r=call('run_command',dict(mount_id=MOUNT,command='python3 -c '+shlex.quote(code),timeout_seconds=30))
 assert r['exit_code']==0,r
 return json.loads(r['stdout'])
source=Path(__file__).parent
if a.stage=='deploy':
 assert not (a.directory/'start-intent.json').exists()
 names=['synthesize.py','delta_mean.py','cpu_driver.py','cpu-plan.json'];buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.writestr(name,(source/name).read_bytes())
 raw=buf.getvalue();digest=hashlib.sha256(raw).hexdigest();save(a.directory/'package.json',dict(files={n:ref(source/n) for n in names},bytes=len(raw),sha256=digest,remote=ROOT))
 r=call('write_file',dict(mount_id=MOUNT,path=REL+'/source.zip',content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True));assert r['sha256']==digest
 actual=command('from pathlib import Path\nimport json,zipfile,hashlib,os\np=Path('+repr(ROOT)+');z=zipfile.ZipFile(p/"source.zip");assert not (p/"driver-started.json").exists()\nfor n in z.namelist():\n assert Path(n).name==n\n data=z.read(n);q=p/n\n with q.open("xb") as h:h.write(data);h.flush();os.fsync(h.fileno())\n assert q.read_bytes()==data\n if n.endswith(".py"):compile(data,str(q),"exec")\nfd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\nprint(json.dumps(dict(deployed=True,files=z.namelist(),disk_free=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize)))')
 save(a.directory/'deployment.json',actual);print(json.dumps(actual))
elif a.stage=='start':
 assert (a.directory/'deployment.json').is_file() and not (a.directory/'start-intent.json').exists()
 p=json.loads((source/'cpu-plan.json').read_text());intent=dict(mount_id=MOUNT,cwd=REL,command=shlex.join([p['python'],'-u','cpu_driver.py']),env=p['environment'])
 save(a.directory/'start-intent.json',intent)
 r=call('start_process',intent);save(a.directory/'started.json',r);print(json.dumps(r))
else:
 code='from pathlib import Path\nimport json,os\np=Path('+repr(ROOT)+');names=["driver-started.json","child-started.json","wrapper-complete.json","output/complete.json","output/tiny-peft.json","output/failure.json"]\nfiles={n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}\nprint(json.dumps(dict(files=files,stdout=(p/"stdout.log").read_text()[-1500:] if (p/"stdout.log").exists() else None,stderr=(p/"stderr.log").read_text()[-4000:] if (p/"stderr.log").exists() else None)))'
 r=command(code);save(a.directory/('observation-'+str(time.time_ns())+'.json'),r);print(json.dumps(r))
