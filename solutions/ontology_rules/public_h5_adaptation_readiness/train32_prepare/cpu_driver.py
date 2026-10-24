"""Single CPU-only bounded job; numerical contract then native token preparation."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,hashlib,zipfile,datetime
D=Path.cwd();t=time.monotonic();deadline=t+240
with (D/'driver-started.json').open('x') as f:json.dump(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cap_seconds=240,CPU_only=True),f)
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')
entry=json.loads((D/'entry.json').read_text());state=dict(running=True,CPU_only=True,stages=[]);rc=1
try:
 for name,cmd,cap in [('loss',[sys.executable,'loss_contract.py','--out','loss-result.json'],45),('native',[sys.executable,'native_cpu.py','--entry','entry.json'],180)]:
  left=min(cap,deadline-time.monotonic());assert left>0;s=time.monotonic()
  with (D/(name+'.stdout')).open('x') as out,(D/(name+'.stderr')).open('x') as err:
   p=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
   try:r=p.wait(timeout=left)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:r=p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);r=p.wait()
  state['stages'].append(dict(name=name,returncode=r,seconds=time.monotonic()-s));(D/'state.json').write_text(json.dumps(state,indent=2)+'\n');assert r==0,name
 rc=0
finally:
 state.update(running=False,returncode=rc,seconds=time.monotonic()-t,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(D/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 files=[p for p in D.iterdir() if p.is_file() and p.name not in ['terminal.zip','terminal-manifest.json','request.json.gz']]
 if (D/'native-result').exists():files.extend(p for p in (D/'native-result').iterdir() if p.is_file())
 members={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
 with zipfile.ZipFile(D/'terminal.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(D)))
  z.writestr('members.json',json.dumps(members,indent=2)+'\n')
 b=(D/'terminal.zip').read_bytes();manifest=dict(state=state,members=members,archive=dict(path=str(D/'terminal.zip'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 with (D/'terminal.zip').open('rb') as f:os.fsync(f.fileno())
 (D/'terminal-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(state=state,archive=manifest['archive'])),flush=True)
raise SystemExit(rc)
