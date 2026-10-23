"""Read-only metadata inventory. No TensorFlow/model import, no CUDA context."""
import os,sys,json,glob,time,subprocess,platform,shutil
from pathlib import Path
from datetime import datetime,timezone
start=time.monotonic();root=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
def command(argv):
 try:
  x=subprocess.run(argv,capture_output=True,text=True,timeout=8);return dict(argv=argv,rc=x.returncode,stdout=x.stdout.strip(),stderr=x.stderr.strip())
 except Exception as e:return dict(argv=argv,error=type(e).__name__)
def measure(p):
 p=Path(p)
 if not p.exists():return dict(path=str(p),exists=False)
 total=0;files=0;links=0
 if p.is_file():return dict(path=str(p),exists=True,bytes=p.stat().st_size,realpath=str(p.resolve()))
 for base,dirs,names in os.walk(p,followlinks=False):
  if time.monotonic()-start>35:return dict(path=str(p),exists=True,partial=True,bytes=total,files=files)
  for n in names:
   x=Path(base)/n
   if x.is_symlink():links+=1;continue
   try:total+=x.stat().st_size;files+=1
   except OSError:pass
 return dict(path=str(p),exists=True,bytes=total,files=files,symlinks_excluded=links)
paths=set();roots=[root,Path('/home/zhangyuheng/projects/ScratchVL')]
# Project-owned runtime/site directories only; no scanning arbitrary users or data.
for scan in roots:
 if not scan.exists():continue
 for base,dirs,names in os.walk(scan,followlinks=False):
  depth=len(Path(base).relative_to(scan).parts)
  if depth>5 or time.monotonic()-start>15:dirs[:]=[];continue
  dirs[:]=[d for d in dirs if d not in ['.git','node_modules','__pycache__','data','datasets','models','model']]
  if 'site-packages' in dirs:paths.add(str(Path(base)/'site-packages'));dirs.remove('site-packages')
  if any(n.startswith(('tensorflow','keras')) and n.endswith('.dist-info') for n in dirs):paths.add(base);dirs[:]=[]
for pat in ['/home/zhangyuheng/miniconda3/envs/*/lib/python*/site-packages','/dataset/miniforge3/envs/*/lib/python*/site-packages','/usr/local/lib/python*/dist-packages','/usr/local/lib/python*/site-packages']:
 paths.update(glob.glob(pat))
metadata=[];interpreters=[]
for s in sorted(paths):
 p=Path(s);tf=[]
 for pat in ['tensorflow*.dist-info/METADATA','keras*.dist-info/METADATA','nvidia_cudnn*.dist-info/METADATA','transformers*.dist-info/METADATA','numpy*.dist-info/METADATA','h5py*.dist-info/METADATA']:
  for q in p.glob(pat):
   fields={}
   for line in q.read_text(errors='replace').splitlines():
    if line.startswith(('Name: ','Version: ')):
     k,v=line.split(': ',1);fields[k]=v
   tf.append(fields)
 if tf:metadata.append(dict(path=s,distributions=tf,has_tensorflow=(p/'tensorflow').is_dir()))
 py=p.parent.parent.parent/'bin/python'
 if py.exists():interpreters.append(dict(path=str(py),realpath=str(py.resolve()),version=command([str(py),'--version'])))
libs=[]
for pattern in ['/usr/local/cuda*/lib64/*.so*','/usr/local/cuda*/targets/x86_64-linux/lib/*.so*']:
 for s in glob.glob(pattern):
  p=Path(s)
  if p.is_file() and any(p.name.startswith(n) for n in ['libcudnn','libcudart','libcublas','libcurand','libcusolver','libcusparse','libcufft','libnvrtc']):libs.append(dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size))
# Explicit source package sizes; skip absent roots and never read/hash weight bytes.
source=[measure(root/'public-phenotagger-reuse/.venv'),measure(root/'public-phenotagger-reuse/tf-gpu-overlay'),measure(root/'public-phenotagger-reuse/nltk-data')]
weight_stats=[]
for base in [root/'public-phenotagger-reuse',root/'public-phenotagger-bioformer']:
 if base.exists():
  for p in base.glob('*.h5'):weight_stats.append(dict(path=str(p),bytes=p.stat().st_size))
report=dict(at_utc=datetime.now(timezone.utc).isoformat(),hostname=platform.node(),platform=platform.platform(),python=sys.version,glibc=platform.libc_ver(),free_disk=shutil.disk_usage(root).free,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free','--format=csv,noheader,nounits']),gpu_processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']),metadata=metadata,interpreters=interpreters,CUDA_libraries=libs,source_directory_sizes=source,top_level_weight_stat_only=weight_stats,scope='metadata/stat only; no tensorflow/torch imports, model load or CUDA context',seconds=time.monotonic()-start)
print(json.dumps(report))
