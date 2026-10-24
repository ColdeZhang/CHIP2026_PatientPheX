from pathlib import Path
import sys,shlex,json
D=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
code='''from pathlib import Path
import sys,json,os,importlib.metadata as md,importlib.util,hashlib,datetime
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.executable,python_version=sys.version,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),no_model_or_tensorflow_import=True,packages={})
for name in ['tensorflow','keras','transformers','numpy','nltk','h5py']:
 spec=importlib.util.find_spec(name);p=Path(spec.origin);entry=dict(module_origin=str(p),module_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 for dist in [name,'tensorflow-cpu'] if name=='tensorflow' else [name]:
  try:
   d=md.distribution(dist);entry.setdefault('distributions',[]).append(dict(name=dist,version=d.version,metadata_path=str(d._path)))
  except md.PackageNotFoundError:pass
 r['packages'][name]=entry
print(json.dumps(r))
'''
c=Client();c.initialize();r=c.request('tools/call',dict(name='run_command',arguments=dict(mount_id='476f81cf-09e7-445b-a7f0-d0e12b12e50f',command='./.venv/bin/python -c '+shlex.quote(code),cwd='public-phenotagger-reuse',env=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/tf-gpu-overlay',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),timeout_seconds=30)));assert not r.get('isError'),r;v=r['structuredContent'];assert v['exit_code']==0,v;d=json.loads(v['stdout']);assert d['CUDA_VISIBLE_DEVICES']=='';p=D/'runtime-metadata-after-terminal.json';assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
