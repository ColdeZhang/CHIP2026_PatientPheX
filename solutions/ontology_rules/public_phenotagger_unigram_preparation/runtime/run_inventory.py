from pathlib import Path
import sys,json,hashlib,concurrent.futures
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
out=Path(__file__).parent;source=(out/'inventory_remote.py').read_text()
mounts={'A100_3_12':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','A30_3':'1593f159-19bc-41ca-8b4c-2e7455f5db99','A30_100':'5a9955da-5240-41cc-9e52-05d3d8da8a3c'}
def job(item):
 label,mount=item;c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':mount,'cwd':'','command':"python3 - <<'PYREMOTE'\n"+source+'\nPYREMOTE','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1'},'timeout_seconds':50}})
 with (out/(label+'.json')).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 j=r.get('structuredContent',{})
 if not j:
  j=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 if j.get('exit_code')==0:
  report=json.loads(j['stdout']);(out/(label+'-inventory.json')).write_text(json.dumps(report,indent=2)+'\n')
  return dict(label=label,rc=0,seconds=report['seconds'],metadata=len(report['metadata']),gpu=report['gpu'])
 return dict(label=label,result=j)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(job,mounts.items()):print(json.dumps(r),flush=True)
