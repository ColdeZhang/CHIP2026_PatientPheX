from pathlib import Path
import sys,json,shlex,base64,hashlib,io,zipfile
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
E=Path(__file__).parent;S=Path('/workspace/patientphex-worktrees/explore-patient-competition-calibration/solutions/patient_competition_calibration');p=json.loads((S/'plan.json').read_text());P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';rel='research/patient-competition-calibration-v1';R=P+'/'+rel;M=p['mount_id'];c=Client();c.initialize()
def call(n,a,label):
 v=c.request('tools/call',{'name':n,'arguments':{'mount_id':M,**a}});(E/(label+'.json')).write_text(json.dumps(v,indent=2));assert not v.get('isError'),v;return v.get('structuredContent') or json.loads(v['content'][0]['text'])
files={'source/'+str(q.relative_to(S)):q.read_bytes() for q in S.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.suffix in ['.py','.json']};files['input/held.jsonl']=Path(p['f0_input']['path']).read_bytes();inv={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()};buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for n,b in files.items():z.writestr(n,b)
 z.writestr('inventory.json',json.dumps(inv,indent=2))
b=buf.getvalue();(E/'deployment.zip').write_bytes(b);x=call('write_file',{'path':rel+'/deployment.zip','content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True},'upload-response');assert x['sha256']==hashlib.sha256(b).hexdigest()
code="""from pathlib import Path
import json,hashlib,zipfile,os
D=Path(%r);p=D/'deployment.zip';assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:
 inv=json.loads(z.read('inventory.json'))
 for n,r in inv.items():
  b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];q=D/n;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
 report={'status':'PASS_SOURCE_AND_PURE_INPUT_IDENTITY','members':len(inv),'deployment_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};(D/'deployment-complete.json').write_text(json.dumps(report,indent=2)+'\\n');print(json.dumps(report))
"""%(R,hashlib.sha256(b).hexdigest());x=call('run_command',{'command':'python3 -c '+shlex.quote(code),'timeout_seconds':30},'deploy-response');assert x['exit_code']==0;(E/'deployment-complete.json').write_text(x['stdout']);print(x['stdout'],flush=True)
cmd=' '.join(['env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false','PYTHONPATH='+shlex.quote(p['pythonpath']),shlex.quote(p['python']),shlex.quote(R+'/source/prepare_held.py'),'--plan',shlex.quote(R+'/source/plan.json'),'--input',shlex.quote(R+'/input/held.jsonl'),'--output',shlex.quote(R+'/cpu')]);x=call('run_command',{'command':cmd,'timeout_seconds':60},'cpu-response');assert x['exit_code']==0,x;(E/'cpu-complete.json').write_text(x['stdout']);print(x['stdout'],flush=True)
