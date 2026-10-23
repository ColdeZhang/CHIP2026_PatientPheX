"""Recover small actual CPU evidence; no weight transfer or model execution."""
from pathlib import Path
import sys,json,hashlib,shlex,base64
R=Path(__file__).parent;A=R/'attempt2';sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';remote='public-phenotagger-biobert-attempt2';c=Client();c.initialize()
def call(name,**a):
 q=c.request('tools/call',{'name':name,'arguments':{'mount_id':M,**a}});assert not q.get('isError');return q.get('structuredContent') or json.loads(q['content'][0]['text'])
names=['CPU-state.json','native-h5-result.json','variable-coverage.json','synthetic-representation.json','CPU-stdout.log','CPU-stderr.log']
code="from pathlib import Path;import json,hashlib;assert json.loads(Path('CPU-state.json').read_text())['status']=='complete';names="+repr(names)+";print(json.dumps({n:{'bytes':Path(n).stat().st_size,'sha256':hashlib.sha256(Path(n).read_bytes()).hexdigest()} for n in names}))"
v=call('run_command',command='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),cwd=remote,env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},timeout_seconds=30);assert v['exit_code']==0,v['stderr'];index=json.loads(v['stdout'])
for n in names:
 q=call('read_file',path=remote+'/'+n,offset=0,limit=index[n]['bytes']+1);b=base64.b64decode(q['content']) if q['encoding']=='base64' else q['content'].encode();assert len(b)==index[n]['bytes'] and hashlib.sha256(b).hexdigest()==index[n]['sha256'];assert not(A/n).exists();(A/n).write_bytes(b)
(A/'transfer.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(index))
