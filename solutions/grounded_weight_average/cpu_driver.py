"""Single bounded CPU synthesis; hidden CUDA and no automatic rerun."""
import json,os,signal,subprocess,time
from pathlib import Path
from synthesize import save,ref
p=json.loads(Path('cpu-plan.json').read_text());assert p['phase']=='CPU_SYNTHESIS_ONLY'
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and p['budget_seconds']==120 and p['grace_seconds']==15
for source in p['source_files'].values():assert ref(source['path'])==source
start=time.monotonic();now=time.time()
save('driver-started.json',dict(pid=os.getpid(),plan=ref('cpu-plan.json'),at_unix=now,TERM_unix=now+120,KILL_unix=now+135))
with open('stdout.log','xb') as out,open('stderr.log','xb') as err:
 child=subprocess.Popen(p['argv'],stdout=out,stderr=err,start_new_session=True)
 save('child-started.json',dict(pid=child.pid));timeout=False
 try:rc=child.wait(timeout=max(.01,start+120-time.monotonic()))
 except subprocess.TimeoutExpired:
  timeout=True;os.killpg(child.pid,signal.SIGTERM)
  try:rc=child.wait(timeout=max(.01,start+135-time.monotonic()))
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
 for handle in [out,err]:handle.flush();os.fsync(handle.fileno())
result=dict(returncode=rc,timed_out=timeout,child_pid=child.pid,seconds=time.monotonic()-start,CUDA_VISIBLE_DEVICES='')
complete=Path(p['output'])/'complete.json'
if rc==0 and not timeout:
 assert complete.exists();result['complete']=ref(complete)
save('wrapper-complete.json',result)
raise SystemExit(rc or int(timeout))
