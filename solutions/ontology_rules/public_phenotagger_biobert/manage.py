"""Small public-source transport; signed CDN details stay private and unprinted."""
from pathlib import Path
import sys,json,hashlib,shlex,urllib.request,base64,argparse,concurrent.futures
ROOT=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f';REMOTE='public-phenotagger-biobert'
PYTHON='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/.venv/bin/python'
ENV={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','USE_TF':'1','USE_TORCH':'0','USE_FLAX':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def client():
 c=Client();c.initialize();return c
def call(c,name,**a):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':MOUNT,**a}});assert not r.get('isError'),name
 return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def command(c,code):
 r=call(c,'run_command',command=shlex.quote(PYTHON)+' -c '+shlex.quote(code),cwd=REMOTE,env=ENV,timeout_seconds=30);assert r['exit_code']==0,{'exit_code':r['exit_code'],'stderr':r['stderr'][-1000:]};return json.loads(r['stdout'])
def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['download','status','cpu','collect']);args=a.parse_args();p=json.loads((ROOT/'plan.json').read_text());c=client()
 if args.action=='download':
  assert not (ROOT/'download-job.json').exists()
  resolve='https://huggingface.co/'+p['model_repo']+'/resolve/'+p['revision']+'/'+p['weight_file']
  with urllib.request.urlopen(urllib.request.Request(resolve,method='HEAD'),timeout=45) as r:
   assert r.status==200;cdn=r.geturl();assert cdn.startswith('https://')
  private={k:p[k] for k in ['revision','weight_bytes','weight_sha256']};private.update(repo_id=p['model_repo'],source_resolve_url=resolve,chunk_bytes=16*1024**2,cdn_url=cdn)
  names=[*p['source_sha256'],'plan.json']
  def put(n):
   cc=client();call(cc,'write_file',path=REMOTE+'/'+n,encoding='base64',content=base64.b64encode((ROOT/n).read_bytes()).decode(),create_parents=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(put,names))
  call(c,'write_file',path=REMOTE+'/download-state.private.json',content=json.dumps(private),create_parents=True);del private,cdn
  code="from pathlib import Path;import os,json,hashlib,shutil;os.chmod('download-state.private.json',0o600);names="+repr(names)+";print(json.dumps({'files':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names},'disk_free_bytes':shutil.disk_usage('.').free}))"
  actual=command(c,code);assert actual['disk_free_bytes']>2*1024**3
  assert all(v==sha(ROOT/n) for n,v in actual['files'].items())
  (ROOT/'deployment.json').write_text(json.dumps(actual,indent=2)+'\n')
  argv='timeout --signal=TERM --kill-after=5s 300s '+shlex.quote(PYTHON)+' download_public_h5.py'
  job=call(c,'start_process',command=argv,cwd=REMOTE,env=ENV);(ROOT/'download-job.json').write_text(json.dumps({'command':argv,'plan_sha256':sha(ROOT/'plan.json'),'result':job},indent=2)+'\n');print(json.dumps(job))
 elif args.action=='status':
  print(json.dumps(command(c,"from pathlib import Path;import json;names=['download-status.json','weight-source-manifest.json','CPU-state.json'];print(json.dumps({n:json.loads(Path(n).read_text()) if Path(n).exists() else None for n in names}))")))
 elif args.action=='cpu':
  assert not (ROOT/'CPU-job.json').exists()
  actual=command(c,"from pathlib import Path;import json,hashlib;assert not Path('CPU-state.json').exists();m=json.loads(Path('weight-source-manifest.json').read_text());assert m['download_complete'];print(json.dumps(m))")
  assert actual['verified_full_weight_sha256']==p['weight_sha256']
  job=call(c,'start_process',command=shlex.quote(PYTHON)+' run_cpu.py',cwd=REMOTE,env=ENV);(ROOT/'CPU-job.json').write_text(json.dumps({'command':PYTHON+' run_cpu.py','env':ENV,'plan_sha256':sha(ROOT/'plan.json'),'result':job},indent=2)+'\n');print(json.dumps(job))
 else:
  names=['download-status.json','weight-source-manifest.json','CPU-state.json','native-h5-result.json','variable-coverage.json','synthetic-representation.json','CPU-stdout.log','CPU-stderr.log']
  actual=command(c,"from pathlib import Path;import json,hashlib;assert json.loads(Path('CPU-state.json').read_text())['status']=='complete';names="+repr(names)+";print(json.dumps({n:{'sha256':hashlib.sha256(Path(n).read_bytes()).hexdigest(),'bytes':Path(n).stat().st_size} for n in names}))")
  for n in names:
   r=call(c,'read_file',path=REMOTE+'/'+n,offset=0,limit=actual[n]['bytes']+1);b=base64.b64decode(r['content']) if r['encoding']=='base64' else r['content'].encode();assert len(b)==actual[n]['bytes'] and hashlib.sha256(b).hexdigest()==actual[n]['sha256'];assert not (ROOT/n).exists();(ROOT/n).write_bytes(b)
  (ROOT/'transfer.json').write_text(json.dumps(actual,indent=2)+'\n');print(json.dumps(actual))
if __name__=='__main__':
 try:main()
 except Exception as e:print('management_failed:'+type(e).__name__,file=sys.stderr);sys.exit(1)
