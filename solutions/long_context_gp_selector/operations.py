"""Exclusive local operation intents; recover original remote namespace on UNKNOWN."""
import argparse,base64,hashlib,json,lzma,os,shlex,sys,time,zipfile
from pathlib import Path
from rpc import call
S=Path(__file__).resolve().parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1');P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';REL='research/long-context-gp-selector-v1/formal-v1';REMOTE=P+'/'+REL;MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def run(code,tag,timeout=30):
 v=call(D,'run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=timeout),tag);assert v.get('exit_code')==0,v;return json.loads(v['stdout'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['data','build','deploy','cpu','start','observe','pack','collect']);a=ap.parse_args()
 if a.action=='data':
  blob=(D/'data.transport.xz').read_bytes();parts=[blob[i:i+262144] for i in range(0,len(blob),262144)]
  existing=run("import json,hashlib;from pathlib import Path;r=Path("+repr(REMOTE)+");print(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.glob('part-*')}))",'data-inventory')
  for i,b in enumerate(parts):
   n='part-'+str(i);wanted=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
   if n in existing:assert existing[n]==wanted;continue
   call(D,'write_file',dict(mount_id=MOUNT,path=REL+'/'+n,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True),'data-part-'+str(i))
  save(D/'data-uploaded.json',dict(at=time.time(),parts=len(parts),transport=ref(D/'data.transport.xz')));return
 if a.action=='build':
  copy=json.load(open(D.parent/'asset-copy-v1/copy-complete.json'))['copy-complete.json'];assets=P+'/research/long-context-gp-selector-v1/transfer/assets';old=json.load(open('/workspace/patientphex-worktrees/research-biomedical-long-context/solutions/biomedical_long_context/formal/formal-plan.json'));jobs=json.load(open(S/'quality-plan.json'))
  env={'USE_TF':'0','USE_FLAX':'0','PYTHONPATH':P+'/research/seq2seq-patient-set-v1/runtime-t450:'+P+'/case-retrieval/round2/training-runtime:'+P+'/case-retrieval/round2/runtime','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
  cfg=dict(schema='bioclinical-gp103-f1-v1',model=assets,model_files={n:{k:r[k] for k in ['bytes','sha256']} for n,r in old['model_files'].items()},GP23_checkpoint=dict(path=assets+'/GP23-model.safetensors',**copy['files']['GP23-model.safetensors']),GP23_model_manifest=dict(path=assets+'/GP23-model-manifest.json',**copy['files']['GP23-model-manifest.json']),data=ref(D/'data.json'),transport=ref(D/'data.transport.xz'),source_files={n:ref(S/'runtime'/n) for n in ['common.py','model.py','native_cpu.py','driver.py','train_score.py']},F1_ids=jobs['outer1_held_ids'],python='/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',environment=env,cpu_receipt=REMOTE+'/cpu-complete.json',gpu_uuid='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539',budget_seconds=900,grace_seconds=30,fit_count=3,score_passes=4,total_new_updates=896,fit_configuration={'epochs':8,'LR':3e-5,'weight_decay':.01,'clip':1.,'seed':20261024,'batch':1,'warmup':'floor(.1*total),25/25/38'},cpu_budget_seconds=120,old_source='parent GP model.py byteexact; original train loop/state writer mechanical job/row count generalization',runtime_change='Torch2.7.0cu128 source→2.7.1cu126 destination, same TF4.50/tokenizers.21.4 target existing isolated installation, hub.36.2→.29.3; not bitwise hardware equivalence',estimate_seconds=654.255356,space_minimum_bytes=8_000_000_000)
  save(S/'runtime/formal-plan.json',cfg);save(D/'formal-plan.json',cfg)
  with zipfile.ZipFile(D/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
   for n in cfg['source_files']:z.write(S/'runtime'/n,n)
   z.write(S/'runtime/formal-plan.json','formal-plan.json')
  save(D/'package.json',dict(source=ref(D/'source.zip'),members={**cfg['source_files'],'formal-plan.json':ref(S/'runtime/formal-plan.json')},data=cfg['data'],transport=cfg['transport'],remote=REMOTE,mount=MOUNT,relative=REL));print(json.dumps(ref(S/'runtime/formal-plan.json')));return
 cfg=json.load(open(D/'formal-plan.json'));pkg=json.load(open(D/'package.json'))
 if a.action=='deploy':
  blob=(D/'data.transport.xz').read_bytes();parts=[blob[i:i+262144] for i in range(0,len(blob),262144)];wanted={str(i):dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for i,b in enumerate(parts)}
  existing=run("import json,hashlib;from pathlib import Path;r=Path("+repr(REMOTE)+");print(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.glob('part-*')}))",'deploy-inventory')
  for i,b in enumerate(parts):
   n='part-'+str(i)
   if n in existing:assert existing[n]==wanted[str(i)];continue
   call(D,'write_file',dict(mount_id=MOUNT,path=REL+'/'+n,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True),'data-part-'+str(i))
  call(D,'write_file',dict(mount_id=MOUNT,path=REL+'/source.zip',content=base64.b64encode((D/'source.zip').read_bytes()).decode(),encoding='base64',create_parents=True),'source-zip')
  code="""import json,lzma,hashlib,os,zipfile
from pathlib import Path
r=Path(%r);parts=%r;blob=b''.join((r/('part-'+str(i))).read_bytes() for i in range(len(parts)));assert hashlib.sha256(blob).hexdigest()==%r
raw=lzma.decompress(blob);assert len(raw)==%r and hashlib.sha256(raw).hexdigest()==%r
with (r/'data.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
assert hashlib.sha256((r/'source.zip').read_bytes()).hexdigest()==%r
z=zipfile.ZipFile(r/'source.zip');members=%r;assert set(z.namelist())==set(members)
for n,v in members.items():
 b=z.read(n);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 with (r/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for p in [r,r.parent]:fd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ALL_SOURCE_AND_RAW_INPUT_EXACT',members=len(members),data_sha256=hashlib.sha256(raw).hexdigest())))
"""%(REMOTE,wanted,cfg['transport']['sha256'],cfg['data']['bytes'],cfg['data']['sha256'],pkg['source']['sha256'],pkg['members'])
  save(D/'deployment.json',run(code,'deployment-finalize',60))
 elif a.action=='cpu':
  assert (D/'deployment.json').exists();save(D/'cpu.intent.json',dict(at=time.time(),budget=120,no_retry=True));v=call(D,'run_command',dict(mount_id=MOUNT,cwd=REL,command=shlex.join(['timeout','--signal=TERM','--kill-after=5','120',cfg['python'],'-u',REMOTE+'/native_cpu.py']),env={**cfg['environment'],'CUDA_VISIBLE_DEVICES':''},timeout_seconds=130),'native-CPU');save(D/'cpu-tool.json',v);assert v['exit_code']==0,v;print(json.dumps(v))
 elif a.action=='start':
  assert (D/'cpu-tool.json').exists() and json.load(open(D/'cpu-tool.json'))['exit_code']==0
  ticket=dict(created_unix=time.time(),launch_not_after_unix=time.time()+300,plan=dict(path=REMOTE+'/formal-plan.json',bytes=pkg['members']['formal-plan.json']['bytes'],sha256=pkg['members']['formal-plan.json']['sha256']))
  save(D/'start.intent.json',ticket);call(D,'write_file',dict(mount_id=MOUNT,path=REL+'/launch-ticket.json',content=json.dumps(ticket),encoding='utf-8',create_parents=False),'ticket')
  v=call(D,'start_process',dict(mount_id=MOUNT,cwd=REL,command=shlex.join([cfg['python'],'-u',REMOTE+'/driver.py']),env=cfg['environment']),'formal-start');save(D/'started.json',v);print(json.dumps(v))
 elif a.action=='observe':
  names=['cpu-complete.json','job/started.json','job/ended.json','job/fresh-gate.json','job/expired-before-entry.json','output/progress.json','output/complete.json']
  code="import json;from pathlib import Path;r=Path("+repr(REMOTE)+");o={n:json.loads((r/n).read_text()) for n in "+repr(names)+" if (r/n).exists()};p=r/'job/stderr';o['stderr']=p.read_text()[-4000:] if p.exists() else None;cfg=json.load(open(r/'formal-plan.json'));q=Path(cfg['GP23_model_manifest']['path']);assert __import__('hashlib').sha256(q.read_bytes()).hexdigest()==cfg['GP23_model_manifest']['sha256'];o['GP23-model-manifest.json']=json.loads(q.read_text());print(json.dumps(o))"
  v=run(code,'observe');save(D/('observation-'+str(time.time_ns())+'.json'),v);print(json.dumps(v))
 elif a.action=='pack':
  code="""import json,hashlib,zipfile,os
from pathlib import Path
r=Path(%r);e=json.load(open(r/'job/ended.json'));assert e['status'] in ['COMPLETE','FAILED'];members={};zpath=r/'terminal.zip'
with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(r.rglob('*')):
  if p.is_file() and (p.suffix in ('.json','.jsonl','.py') or str(p.relative_to(r)) in ('job/stdout','job/stderr')) and p.name!='data.json':
   n=str(p.relative_to(r));b=p.read_bytes();members[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());z.writestr(n,b)
 index=json.dumps(members,indent=2).encode();z.writestr('members.json',index)
with zpath.open('rb') as f:os.fsync(f.fileno())
idx=r/'terminal-index.json';idx.write_bytes(index)
with idx.open('rb') as f:os.fsync(f.fileno())
fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(archive=dict(bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest()),index=dict(bytes=len(index),sha256=hashlib.sha256(index).hexdigest()),members=len(members))))
"""%REMOTE
  save(D/'terminal-package.json',run(code,'pack',60))
 else:
  pack=json.load(open(D/'terminal-package.json'));out=D/'collected';out.mkdir(exist_ok=True)
  for remote_name,info in [('terminal-index.json',pack['index']),('terminal.zip',pack['archive'])]:
   path=D/remote_name;offset=path.stat().st_size if path.exists() else 0
   while offset<info['bytes']:
    v=call(D,'read_file',dict(mount_id=MOUNT,path=REL+'/'+remote_name,offset=offset,limit=min(262144,info['bytes']-offset),encoding='base64'),'collect-'+remote_name+'-'+str(offset));b=base64.b64decode(v['content']);assert b
    with path.open('ab') as f:f.write(b);f.flush();os.fsync(f.fileno())
    offset+=len(b)
   assert ref(path)['sha256']==info['sha256']
  members=json.load(open(D/'terminal-index.json'))
  with zipfile.ZipFile(D/'terminal.zip') as z:
   assert set(z.namelist())==set(members)|{'members.json'} and z.read('members.json')==(D/'terminal-index.json').read_bytes()
   for n,r in members.items():
    b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert ref(p)['sha256']==r['sha256']
  for directory,dirs,files in os.walk(out,topdown=False):
   fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  save(D/'collection-complete.json',dict(status='WHOLE_ZIP_AND_ALL_MEMBERS_SHA_FSYNC_REOPEN',members=len(members),package=ref(D/'terminal-package.json'),collected=str(out)))
if __name__=='__main__':main()
