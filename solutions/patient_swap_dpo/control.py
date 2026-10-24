"""Recoverable local deployment/start/inspection; secrets stay in Client env."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

PREFIX='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
MOUNT='1593f159-19bc-41ca-8b4c-2e7455f5db99'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def localref(p):b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=digest(b))
def relative(p):assert p.startswith(PREFIX);return p[len(PREFIX):]


class Remote:
    def __init__(self):self.client=Client();self.client.initialize()
    def call(self,name,**args):
        r=self.client.request('tools/call',dict(name=name,arguments=dict(mount_id=MOUNT,**args)))
        if r.get('isError'):raise RuntimeError('MCP tool error; original intent retained')
        if 'structuredContent' in r:return r['structuredContent']
        return json.loads(next(x['text'] for x in r['content'] if x.get('type')=='text'))
    def put(self,remote,data):
        return self.call('write_file',path=relative(remote),content=base64.b64encode(data).decode(),encoding='base64',create_parents=True)
    def read(self,remote):
        r=self.call('read_file',path=relative(remote),offset=0,limit=1048576)
        return base64.b64decode(r['content']) if r.get('encoding')=='base64' else r['content'].encode()


def deploy(D,remote):
    items=json.loads((D/'deployment.json').read_text())
    for i,item in enumerate(items):
        src=Path(item['local']['path']);assert localref(src)==item['local']
        response=D/f'upload-{i:02d}.json';intent=D/f'upload-{i:02d}-intent.json'
        if response.exists():continue
        assert not intent.exists(),'Unknown upload: read original remote path and prove exact bytes before recovering receipt'
        save(intent,dict(at_utc=now(),**item))
        result=remote.put(item['remote'],src.read_bytes());save(response,result)
    code='''import json,hashlib,os
from pathlib import Path
items=ITEMS
for item in items:
 p=Path(item['remote']);b=p.read_bytes();assert len(b)==item['local']['bytes'] and hashlib.sha256(b).hexdigest()==item['local']['sha256']
 with p.open('rb') as f:os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'ALL_DEPLOYMENT_FILES_SHA_FSYNC_REOPEN','files':len(items)}))
'''.replace('ITEMS',repr(items))
    plan=json.loads((D/'disabled-plan.json').read_text())
    result=remote.call('run_command',command=shlex.join([plan['python'],'-c',code]),cwd='',timeout_seconds=30)
    save(D/'deployment-verification.json',result);assert result['exit_code']==0


def cpu(D,remote):
    p=json.loads((D/'disabled-plan.json').read_text());root=p['application'].rsplit('/',1)[0]
    intent=D/'target-cpu-intent.json';assert not intent.exists(),'Recover original target CPU; do not restart blindly'
    command=shlex.join(['timeout','--signal=TERM','--kill-after=5','50',p['python'],p['application']+'/runtime/input_gate.py','--plan',p['application']+'/disabled-plan.json','--output',root+'/cpu-target-v1'])
    args=dict(command=command,cwd=relative(root),env=dict(p['environment'],CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'),timeout_seconds=55)
    save(intent,dict(at_utc=now(),arguments=args));r=remote.call('run_command',**args);save(D/'target-cpu-response.json',r);assert r['exit_code']==0


def collect_cpu(D,remote):
    p=json.loads((D/'disabled-plan.json').read_text());root=p['application'].rsplit('/',1)[0]
    dest=D/'target-cpu';dest.mkdir(exist_ok=True)
    for name in ['complete.json','verified-assets.json','row-audit.json']:
        out=dest/name
        if out.exists():continue
        b=remote.read(root+'/cpu-target-v1/'+name)
        with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    completed=json.loads((dest/'complete.json').read_text());assert completed['status']=='PASS_ACTUAL_TARGET_HF_NATIVE_ALL148_AND_INHERITED_ASSET_STAT'
    for key,name in [('assets','verified-assets.json'),('row_audit','row-audit.json')]:
        r=localref(dest/name);assert (r['sha256'],r['bytes'])==(completed[key]['sha256'],completed[key]['bytes'])
    print(json.dumps(dict(status=completed['status'],complete=localref(dest/'complete.json'))))


def activate(D,remote):
    old=D/'disabled-plan.json';p=json.loads(old.read_text());root=p['application'].rsplit('/',1)[0]
    done=json.loads((D/'target-cpu/complete.json').read_text())
    assert done['plan']['sha256']==localref(old)['sha256']
    for name,dig in p['source_files'].items():
        assert digest((Path(__file__).parent/'runtime'/name).read_bytes())==dig,name
    p['execution_authorized']=True;p['CPU_ready']=dict(localref(D/'target-cpu/complete.json'),path=root+'/cpu-target-v1/complete.json')
    p['activated_at_utc']=now();p['authorization']='root fixed DPO reference900/probe600/formal1800/infer1200+30; one primary only'
    save(D/'execution-plan.json',p)
    intent=dict(at_utc=now(),source=localref(D/'execution-plan.json'),remote=p['application']+'/execution-plan.json')
    save(D/'activation-upload-intent.json',intent)
    result=remote.put(intent['remote'],(D/'execution-plan.json').read_bytes());save(D/'activation-upload-response.json',result)
    assert remote.read(intent['remote'])==(D/'execution-plan.json').read_bytes()
    save(D/'activated.json',dict(at_utc=now(),plan=localref(D/'execution-plan.json'),actual_remote_byteexact=True))


def start(D,remote):
    p=json.loads((D/'execution-plan.json').read_text());assert (D/'activated.json').exists()
    assert not (D/'GPU-start-intent.json').exists(),'Recover original managed process; never issue a second start'
    args=dict(command=shlex.join([p['python'],p['application']+'/runtime/driver.py','--plan',p['application']+'/execution-plan.json']),cwd=relative(p['application']),env=dict(p['environment'],CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'))
    save(D/'GPU-start-intent.json',dict(at_utc=now(),arguments=args,plan=localref(D/'execution-plan.json')))
    r=remote.call('start_process',**args);save(D/'GPU-start-response.json',r);print(json.dumps(r))


def inspect(D,remote):
    p=json.loads((D/'execution-plan.json').read_text())
    code='''import json
from pathlib import Path
r=Path(ROOT)
names=['driver-start.json','driver-progress.json','driver-result.json','reference/complete.json','reference/progress.json','probe/complete.json','probe/progress.json','formal/complete.json','formal/progress.json','formal/epoch-001-persistent-ack.json']
result={n:json.loads((r/n).read_text()) for n in names if (r/n).is_file()}
for phase in ['reference','probe','formal']:
 p=r/(phase+'.stderr.log')
 if p.exists() and p.stat().st_size:result[phase+'_stderr_tail']=p.read_text()[-2500:]
print(json.dumps(result))
'''.replace('ROOT',repr(p['run_root']))
    r=remote.call('run_command',command=shlex.join([p['python'],'-c',code]),cwd='',timeout_seconds=20)
    save(D/('inspect-'+str(__import__('time').time_ns())+'.json'),r)
    assert r['exit_code']==0;value=json.loads(r['stdout']);print(json.dumps(value))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True,type=Path);ap.add_argument('mode',choices=['deploy','cpu','collect_cpu','activate','start','inspect']);a=ap.parse_args()
    try:globals()[a.mode](a.directory,Remote())
    except Exception as e:
        save(a.directory/('failure-'+a.mode+'-'+str(__import__('time').time_ns())+'.json'),dict(at_utc=now(),error_type=type(e).__name__,original_intents_preserved=True))
        print(json.dumps(dict(error_type=type(e).__name__,mode=a.mode)));raise SystemExit(1)
