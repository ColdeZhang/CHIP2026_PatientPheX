from pathlib import Path
import sys,json
sys.path.insert(0,'/workspace/CHIP2026_PheX/scripts')
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
u=Path(__file__).parent;c=Client();c.initialize();p='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse'
a=dict(mount_id='476f81cf-09e7-445b-a7f0-d0e12b12e50f',command="python3 - <<'PYREMOTE'\n"+(u/'relocation_remote.py').read_text()+'\nPYREMOTE',cwd='',env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},timeout_seconds=30)
(u/'relocation-command.json').write_text(json.dumps(a,indent=2)+'\n');r=c.request('tools/call',dict(name='run_command',arguments=a));(u/'relocation-raw.json').write_text(json.dumps(r,indent=2)+'\n');j=r.get('structuredContent')
if not j:j=json.loads(next(v['text'] for v in r['content'] if v['type']=='text'))
if j.get('exit_code')==0:
 q=json.loads(j['stdout']);(u/'relocation-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
else: print(json.dumps(j))
