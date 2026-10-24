from pathlib import Path
import sys,json,hashlib,base64,shlex
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup']
import remote_io as io
E=Path(__file__).resolve().parent; D=E/'native-source'
assert not (D/'source-receipt.json').exists()
root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a30-fullcontext/runtime/transformers'
paths={n:root+'/models/gemma3/'+n for n in ['modeling_gemma3.py','configuration_gemma3.py']}
code='from pathlib import Path;import json,hashlib,base64;paths='+repr(paths)+';print(json.dumps({n:dict(path=p,bytes=Path(p).stat().st_size,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),base64=base64.b64encode(Path(p).read_bytes()).decode()) for n,p in paths.items()}))'
r=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':15})
assert r['exit_code']==0
records=json.loads(r['stdout'])
for n,v in records.items():
 b=base64.b64decode(v.pop('base64'));assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];(D/n).write_bytes(b)
(D/'source-receipt.json').write_text(json.dumps({'mount_id':io.M,'remote_read_only':True,'no_GPU_no_import_no_weights':True,'files':records},indent=2)+'\n')
print(json.dumps(records))
