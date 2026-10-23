from pathlib import Path
import sys,json,io,zipfile,base64,hashlib,shlex
R=Path(__file__).parent;files=['run_train64.py','make_provenance.py','launch_cache.py','plan.json','input-manifest.json','train64.input.jsonl'];buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.writestr('train64-cache/'+n,(R/n).read_bytes())
b=buf.getvalue();sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();m='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
for i,offset in enumerate(range(0,len(b),196608)):
 r=c.request('tools/call',{'name':'write_file','arguments':{'mount_id':m,'path':'public-phenotagger-biobert/train64-upload/part-%03d'%i,'content':base64.b64encode(b[offset:offset+196608]).decode(),'encoding':'base64','create_parents':True}});assert not r.get('isError'),r
code="from pathlib import Path;import json,zipfile,io,hashlib;assert not Path('train64-cache/plan.json').exists();parts=sorted(Path('train64-upload').glob('part-*'));b=b''.join(p.read_bytes() for p in parts);assert hashlib.sha256(b).hexdigest()=="+repr(hashlib.sha256(b).hexdigest())+";z=zipfile.ZipFile(io.BytesIO(b));z.extractall('.');print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in z.namelist()}));[p.unlink() for p in parts]"
r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':m,'command':'../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'public-phenotagger-biobert','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30}});assert r['structuredContent']['exit_code']==0,r;actual=json.loads(r['structuredContent']['stdout'])
for n in files:assert actual['train64-cache/'+n]==hashlib.sha256((R/n).read_bytes()).hexdigest()
(R/'deployment.json').write_text(json.dumps({'archive_bytes':len(b),'remote_files_sha256':actual,'gold_uploaded':False},indent=2)+'\n')
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','FORCE_TF_AVAILABLE':'1','USE_TF':'1','USE_TORCH':'0'};command='timeout --signal=TERM --kill-after=5s 120 ../public-phenotagger-reuse/.venv/bin/python train64-cache/run_train64.py --prepare-only'
r=c.request('tools/call',{'name':'start_process','arguments':{'mount_id':m,'command':command,'cwd':'public-phenotagger-biobert','env':env}});(R/'CPU-preparation-job.json').write_text(json.dumps({'command':command,'env':env,'result':r},indent=2)+'\n');print(json.dumps(r))
