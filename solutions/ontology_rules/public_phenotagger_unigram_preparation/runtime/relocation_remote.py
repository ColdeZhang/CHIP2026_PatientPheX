import json,sys,subprocess,platform,time
from pathlib import Path
P=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse')
site=P/'.venv/lib/python3.10/site-packages';overlay=P/'tf-gpu-overlay'
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=10);return dict(argv=a,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
so=overlay/'tensorflow/python/_pywrap_tensorflow_internal.so'
print(json.dumps(dict(scope='metadata/ldd only; no import/model/CUDA',venv_config=(P/'.venv/pyvenv.cfg').read_text(),source_python=dict(path=str(P/'.venv/bin/python'),realpath=str((P/'.venv/bin/python').resolve()),version=cmd([str(P/'.venv/bin/python'),'--version'])),wheel_tags=[dict(path=str(p),text=p.read_text()) for base in [site,overlay] for pattern in ['tensorflow*.dist-info/WHEEL','numpy*.dist-info/WHEEL','h5py*.dist-info/WHEEL'] for p in sorted(base.glob(pattern))],pth=[dict(path=str(p),text=p.read_text()) for p in sorted(site.glob('*.pth'))],tensorflow_so_dynamic_dependencies=cmd(['ldd',str(so)]))))
