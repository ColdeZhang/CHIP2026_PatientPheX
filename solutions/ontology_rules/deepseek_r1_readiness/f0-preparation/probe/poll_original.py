"""One read-only original-job poll and terminal collection; never stop/restart."""
from pathlib import Path
import json,sys,hashlib,shlex,time,base64
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
G=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/f0-longest-v1'
def un(r):
 if r.get('isError'):raise RuntimeError('Read-only monitoring failed; original job remains unchanged')
 return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 c=Client();c.initialize();job=un(json.loads((G/'job.json').read_text()));statepath=G/'monitor-state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'cursor':0}
 r=c.request('tools/call',{'name':'read_process_output','arguments':dict(mount_id=M,process_id=job['id'],stream='stdout',cursor=state['cursor'],limit=10000)});o=un(r);state.update(cursor=o.get('next_cursor',state['cursor']),running=o.get('running'),observed_epoch=time.time(),process_id=job['id']);statepath.write_text(json.dumps(state,indent=2)+'\n')
 if o.get('content'):
  with (G/'wrapper.stdout.log').open('a') as f:f.write(o['content'])
 code="from pathlib import Path;import json,hashlib,base64,time;p=Path("+repr(remote)+");done=(p/'wrapper-result.json').exists();names=['wrapper-started.json','wrapper-running.json','fresh-GPU2.json'] if not done else ['wrapper-started.json','wrapper-running.json','fresh-GPU2.json','wrapper-result.json','probe.stdout.log','probe.stderr.log']+['terminal/'+q.name for q in sorted((p/'terminal').glob('*')) if q.is_file()];r={'observed_epoch':time.time(),'terminal':done,'files':{}};\nfor n in names:\n q=p/n\n if q.exists():\n  b=q.read_bytes();assert len(b)<1000000;r['files'][n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}\nprint(json.dumps(r))"
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),timeout_seconds=30)});v=un(r);assert v['exit_code']==0;snap=json.loads(v['stdout']);dest=G/('recovered' if snap['terminal'] else 'monitor-latest');dest.mkdir(exist_ok=True)
 manifest={k:snap[k] for k in ['observed_epoch','terminal']};manifest['files']={}
 for n,x in snap['files'].items():
  b=base64.b64decode(x['base64'],validate=True);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'];p=dest/n;p.parent.mkdir(exist_ok=True,parents=True)
  if snap['terminal'] and p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
  manifest['files'][n]={k:x[k] for k in ['bytes','sha256']}
 (dest/'collection-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 result=json.loads((dest/'wrapper-result.json').read_text()) if snap['terminal'] else {}
 print(json.dumps(dict(running=state['running'],terminal=snap['terminal'],status=result.get('result_status'),returncode=result.get('returncode'),seconds=result.get('GPU_main_seconds'),released=result.get('target_UUID_has_no_compute_process'),files=len(snap['files']))))
if __name__=='__main__':main()
