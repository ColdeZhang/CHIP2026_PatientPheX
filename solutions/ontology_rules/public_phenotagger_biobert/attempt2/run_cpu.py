"""One bounded CPU native strict-load/interface check, never overwriting an attempt."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(Path('plan.json').read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not Path('CPU-state.json').exists() and not Path('native-h5-result.json').exists()
assert p['CPU_cap_seconds']==174 and p['TERM_grace_seconds']==5
for n,h in p['source_sha256'].items():assert sha(n)==h,n
argv=['timeout','--signal=TERM','--kill-after=5s','174s',sys.executable,'check_native_h5.py']
s={'status':'running','start_utc':datetime.now(timezone.utc).isoformat(),'argv':argv,'plan_sha256':sha('plan.json'),'source_sha256':sha('check_native_h5.py'),'wrapper_sha256':sha(__file__),'CPU_threads':1,'GPU_visible':False}
Path('CPU-state.json').write_text(json.dumps(s,indent=2)+'\n');t=time.monotonic()
try:
 with Path('CPU-stdout.log').open('x') as out,Path('CPU-stderr.log').open('x') as err:r=subprocess.run(argv,stdout=out,stderr=err,timeout=190)
 s['returncode']=r.returncode;assert r.returncode==0
 s['status']='complete';s['report_sha256']=sha('native-h5-result.json')
except BaseException as e:s.update(status='failed_preserved',error_type=type(e).__name__);raise
finally:
 s['seconds']=time.monotonic()-t;s['end_utc']=datetime.now(timezone.utc).isoformat();Path('CPU-state.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
