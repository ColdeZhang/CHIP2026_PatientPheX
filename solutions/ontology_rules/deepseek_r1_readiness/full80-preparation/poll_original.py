"""Read-only original full80 process monitor; no start/kill/write remote actions."""
from pathlib import Path
import sys,json,os,time,shlex,hashlib,base64
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REMOTE='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/full80-v1'
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 assert (F/'job.json').exists();c=Client();c.initialize()
 code='''from pathlib import Path
import json,datetime,os,hashlib,base64
p=Path(REMOTE)
def read(n):
 q=p/n
 return json.loads(q.read_text()) if q.exists() else None
s=read('wrapper-running.json') or read('wrapper-started.json');w=read('wrapper-result.json');v=read('terminal/result.json') or read('terminal/progress.json')
out={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminal':w is not None,'wrapper':w or s,'progress':None}
if v:
 out['progress']={k:v.get(k) for k in ['status','phase','started_at','finished_at','new_generations','reused_generations','new_generation_target','reuse_target','coverage','program_seconds','error']};out['progress']['rows_completed']=len(v.get('rows',[]));out['progress']['last_completed']=(v.get('rows') or [None])[-1]
files=['wrapper-started.json','wrapper-running.json','fresh-GPU1.json','actual-input-contract.json']+(['wrapper-result.json','terminal/result.json'] if w else [])
out['actual_files']={n:{'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest(),'bytes':(p/n).stat().st_size,'base64':base64.b64encode((p/n).read_bytes()).decode()} for n in files if (p/n).exists()}
if s and s.get('timeout_pid'):
 q=Path('/proc')/str(s['timeout_pid'])/'stat';out['timeout_process_present']=q.exists()
 if q.exists():out['timeout_process_state']=q.read_text().split(') ',1)[1].split()[0]
print(json.dumps(out))
'''.replace('REMOTE',repr(REMOTE))
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)})
 if r.get('isError'):
  (F/'monitor-error-latest.json').write_text(json.dumps(r,indent=2)+'\n');raise RuntimeError('Read-only original monitor error; preserve job, do not restart')
 v=un(r);assert v['exit_code']==0;d=json.loads(v['stdout']);(F/'monitor-state.json').write_text(json.dumps({k:v for k,v in d.items() if k!='actual_files'},indent=2)+'\n')
 dest=F/('recovered' if d['terminal'] else 'monitor-latest');dest.mkdir(exist_ok=True)
 refs={}
 for n,e in d['actual_files'].items():
  raw=base64.b64decode(e['base64']);assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];q=dest/n;q.parent.mkdir(exist_ok=True,parents=True);q.write_bytes(raw);refs[n]={k:e[k] for k in ['bytes','sha256']}
 (dest/'collection-manifest.json').write_text(json.dumps({'observed_at':d['observed_at'],'terminal':d['terminal'],'files':refs},indent=2)+'\n')
 print(json.dumps({k:v for k,v in d.items() if k!='actual_files'}),flush=True)
 return d['terminal']
if __name__=='__main__':main()
