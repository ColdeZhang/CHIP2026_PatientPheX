from pathlib import Path
import json,sys,concurrent.futures
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
u=Path(__file__).parent
jobs=[('A100_3_12','476f81cf-09e7-445b-a7f0-d0e12b12e50f','/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/.venv/bin/python'),('A30_3','1593f159-19bc-41ca-8b4c-2e7455f5db99','/usr/bin/python3.10'),('A30_100','5a9955da-5240-41cc-9e52-05d3d8da8a3c','/usr/bin/python3.10')]
def job(t):
 label,mount,py=t;c=Client();c.initialize();r=c.request('tools/call',dict(name='run_command',arguments=dict(mount_id=mount,command=py+" - <<'PYREMOTE'\n"+(u/'abi_remote.py').read_text()+'\nPYREMOTE',cwd='',env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},timeout_seconds=15)))
 (u/(label+'-abi-raw.json')).write_text(json.dumps(r,indent=2)+'\n');j=r.get('structuredContent')
 if not j:j=json.loads(next(v['text'] for v in r['content'] if v['type']=='text'))
 assert j.get('exit_code')==0,j
 q=json.loads(j['stdout']);(u/(label+'-abi.json')).write_text(json.dumps(q,indent=2)+'\n');return label,q
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
 for x in e.map(job,jobs):print(json.dumps(x),flush=True)
