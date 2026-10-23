from pathlib import Path
import sys,json
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
u=Path(__file__).parent;c=Client();c.initialize()
p='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse'
env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','FORCE_TF_AVAILABLE':'1','USE_TF':'1','USE_TORCH':'0','TF_CPP_MIN_LOG_LEVEL':'2','PYTHONPATH':p+'/tf-gpu-overlay','LD_LIBRARY_PATH':p+'/tf-gpu-overlay/nvidia/cudnn/lib:/usr/local/cuda-11.8/lib64'}
args={'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','command':p+"/.venv/bin/python - <<'PYREMOTE'\n"+(u/'cpu_import_remote.py').read_text()+'\nPYREMOTE','cwd':'','env':env,'timeout_seconds':40}
(u/'cpu-import-command.json').write_text(json.dumps(args,indent=2)+'\n')
r=c.request('tools/call',{'name':'run_command','arguments':args});(u/'cpu-import-raw.json').write_text(json.dumps(r,indent=2)+'\n')
j=r.get('structuredContent')
if not j:j=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
if j.get('exit_code')==0:
 z=json.loads(j['stdout']);(u/'cpu-import-result.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z))
else: print(json.dumps(j))
