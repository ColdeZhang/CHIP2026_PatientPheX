"""One explicitly authorized atomic tombstone; never launches or kills a process."""
import json,sys,time,shlex,os
from pathlib import Path
from prepare import D,S,ref,save
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/copy-grounded-decode-v1/F0-proposer-v1/normalize-v1'
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
CODE=r'''
from pathlib import Path
import datetime,hashlib,json,os
r=Path(CFG['root']);claim=r/'start-intent.json'
def ident(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ident(claim)==CFG['claim']
assert ident(r/'job-plan.json')==CFG['plan']
assert ident(r/'job_driver.py')==CFG['driver']
for name in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log','fresh-resource-gate.json','fresh-resource-gate.json.pending']:
 assert not (r/name).exists(), 'Original path changed; do not fence or second-start'
assert not (r/'output').exists()
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  a=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0));cwd=(p/'cwd').resolve()
  assert not (cwd==r and any(x in ['job_driver.py','normalize_entities.py',str(r/'job_driver.py'),str(r/'normalize_entities.py')] for x in a)), 'Original process present'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
v=dict(schema='FENCED_UNKNOWN_LAUNCH',resource_PASS=False,fence_id=CFG['fence_id'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Root-authorized tombstone: original unknown start had no prebound admission expiry; late original driver must fail exclusive fresh-resource-gate publish before Popen/model',request=CFG['request'],claim=CFG['claim'],original_plan=CFG['plan'],original_driver=CFG['driver'],no_model_no_start_no_kill=True)
b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();pending=r/'fresh-resource-gate.json.pending';target=r/'fresh-resource-gate.json'
with pending.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.link(pending,target);pending.unlink()
q=r
while True:
 fd=os.open(q,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if q.parent==q:break
 q=q.parent
assert target.read_bytes()==b
# If original has crossed the marker before this fence, never claim sole admission.
assert not (r/'driver-started.json').exists() and not (r/'child-started.json').exists()
print(json.dumps(dict(status='FENCED_BEFORE_MODEL',path=str(target),**ident(target),value=v)))
'''
def main():
 out=D/'unknown-fence-v3';out.mkdir(exist_ok=False)
 req=D/'normalize-v1/transport/start-1792855867731525732.intent.json';p=json.loads((S/'normalize-plan-essential-v2.json').read_text());cfg=dict(root=ROOT,fence_id='F0-Sap-UNKNOWN-1792855867731525732-v3',request=ref(req),claim=dict(bytes=7152,sha256='370bbc840102169c22f95eada777b51167ba05f4e64f7b94bfe1591a15a45193'),plan={k:ref(S/'normalize-plan-essential-v2.json')[k] for k in ['bytes','sha256']},driver={k:ref(S/'job_driver.py')[k] for k in ['bytes','sha256']})
 save(out/'intent.json',dict(source=ref(__file__),configuration=cfg,explicit_root_authorization=True))
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize()
 def call(n,args):
  response=c.request('tools/call',dict(name=n,arguments=args));save(out/(n+'.response.json'),response)
  assert not response.get('isError'),'UNKNOWN/error: inspect original fence; never rerun this mutation'
  return response.get('structuredContent') or json.loads(next(x['text'] for x in response['content'] if x['type']=='text'))
 # Preserve only matched managed rows; never archive unrelated managed-process data.
 rr=c.request('tools/call',dict(name='list_processes',arguments=dict(mount_id=MOUNT)));assert not rr.get('isError');rows=rr.get('structuredContent') or json.loads(next(x['text'] for x in rr['content'] if x['type']=='text'));assert isinstance(rows,list)
 matched=[x for x in rows if 'F0-proposer-v1/normalize-v1' in json.dumps(x)];save(out/'registry.json',dict(records=len(rows),matches=matched));assert not matched
 code='CFG='+repr(cfg)+'\n'+CODE
 v=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=40));assert v['exit_code']==0
 result=json.loads(v['stdout']);save(out/'complete.json',dict(source=ref(__file__),intent=ref(out/'intent.json'),registry=ref(out/'registry.json'),actual=result));print(json.dumps(ref(out/'complete.json')))
if __name__=='__main__':main()
