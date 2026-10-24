from pathlib import Path
import os,sys,json,shutil,time
root=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
roots=[root/'research',root/'case-retrieval',Path('/home/zhangyuheng/.cache/modelscope'),Path('/home/zhangyuheng/.cache/huggingface/hub')]
found=[];scanned=0
for base in roots:
 if not base.exists():continue
 for p,ds,fs in os.walk(str(base),followlinks=False):
  depth=len(Path(p).relative_to(base).parts);scanned+=1
  ds[:]=[d for d in ds if d not in {'.venv','venv','site-packages','.git','node_modules'}]
  for name in ds+fs:
   if ('deepseek' in name.lower() or 'r1-0528' in name.lower()):
    x=Path(p)/name;st=x.lstat();found.append({'path':str(x),'uid':st.st_uid,'is_dir':x.is_dir(),'symlink':x.is_symlink()})
  if depth>=4:ds[:]=[]
usage=shutil.disk_usage(str(root));own=root/'research/deepseek-r1-0528-qwen3-8b-readiness'
print(json.dumps({'utc_epoch':time.time(),'uid':os.getuid(),'gid':os.getgid(),'python':sys.version,'root':str(root),'root_uid':root.stat().st_uid,'root_mode':oct(root.stat().st_mode & 0o777),'disk':dict(total=usage.total,used=usage.used,free=usage.free),'target_exists':own.exists(),'target_uid':own.stat().st_uid if own.exists() else None,'bounded_cache_roots':[str(x) for x in roots],'directories_examined':scanned,'matches':found},sort_keys=True))
