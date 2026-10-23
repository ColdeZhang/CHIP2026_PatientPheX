from pathlib import Path
import sys,json,io,zipfile,base64,hashlib,shlex
R=Path(__file__).parent;plan=json.loads((R/'plan.json').read_text());files=['extract_lookup.py','plan.json','span-whitelist.jsonl'];buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.writestr(n,(R/n).read_bytes())
b=buf.getvalue();sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();m=plan['runtime']['mount_id']
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':m,**a}});assert not r.get('isError'),r;return r['structuredContent']
assert not (R/'job.json').exists()
for i,off in enumerate(range(0,len(b),196608)):call('write_file',{'path':'public-phenotagger-top3-lookup/upload/part-%03d'%i,'content':base64.b64encode(b[off:off+196608]).decode(),'encoding':'base64','create_parents':True})
code="from pathlib import Path;import json,zipfile,io,hashlib;parts=sorted(Path('upload').glob('part-*'));b=b''.join(p.read_bytes() for p in parts);assert hashlib.sha256(b).hexdigest()=="+repr(hashlib.sha256(b).hexdigest())+";z=zipfile.ZipFile(io.BytesIO(b));z.extractall('.');print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in z.namelist()}));[p.unlink() for p in parts]"
r=call('run_command',{'command':plan['runtime']['python']+' -c '+shlex.quote(code),'cwd':plan['runtime']['cwd'],'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});assert r['exit_code']==0,r;actual=json.loads(r['stdout']);assert all(actual[n]==hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files);(R/'deployment.json').write_text(json.dumps({'zip_bytes':len(b),'actual_sha256':actual},indent=2)+'\n')
command='timeout --signal=TERM --kill-after=10s 120s '+plan['runtime']['python']+' extract_lookup.py';job=call('start_process',{'command':command,'cwd':plan['runtime']['cwd'],'env':plan['runtime']['environment']});(R/'job.json').write_text(json.dumps({'command':command,'environment':plan['runtime']['environment'],'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),'result':job},indent=2)+'\n');print(json.dumps(job))
