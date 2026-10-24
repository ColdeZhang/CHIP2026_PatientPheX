from pathlib import Path
import sys,json,hashlib,shlex,datetime
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
T=Path(__file__).resolve().parent;assert not (T/'deployment-complete.json').exists()
root='research/medgemma-relation-control/formal-preparation';receipts=[]
for mount,name in [('69b991ee-183f-4add-af1a-579af92479c9','export_checkpoint_server.py'),('5a9955da-5240-41cc-9e52-05d3d8da8a3c','fetch_checkpoint.py')]:
 io.M=mount;path=root+'/'+name;digest=hashlib.sha256((T/name).read_bytes()).hexdigest()
 io.upload(str(T/name),path)
 r=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;p=Path('+repr(path)+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(digest)+';print(json.dumps(dict(path=str(p),sha256='+repr(digest)+',bytes=p.stat().st_size)))'),'cwd':'.','timeout_seconds':15});assert r['exit_code']==0;receipts.append({'mount':mount,'actual':r})
(T/'deployment-complete.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_jobs_started':True,'receipts':receipts},indent=2)+'\n');print('MED_FORMAL_TOOLS_ONLY_DEPLOYED')
