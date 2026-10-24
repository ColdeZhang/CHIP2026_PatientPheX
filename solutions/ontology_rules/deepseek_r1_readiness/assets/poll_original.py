from pathlib import Path
import sys,json,shlex,time,hashlib
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
A=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/assets'
def un(r):
 if isinstance(r.get('structuredContent'),dict):return r['structuredContent']
 for x in r.get('content',[]):
  if x.get('type')=='text':return json.loads(x['text'])
 raise ValueError('unknown MCP result')
job=un(json.loads((A/'download-job.json').read_text()));c=Client();c.initialize();statepath=A/'monitor-state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'cursor':0}
r=c.request('tools/call',{'name':'read_process_output','arguments':{'mount_id':M,'process_id':job['id'],'stream':'stdout','cursor':state['cursor'],'limit':12000}});p=un(r)
state.update(cursor=p.get('next_cursor',state['cursor']),running=p.get('running'),observed_epoch=time.time(),process_id=job['id']);statepath.write_text(json.dumps(state,indent=2)+'\n')
if p.get('content'):
 with (A/'download.stdout.log').open('a') as f:f.write(p['content'])
code="from pathlib import Path; import json,os,time; b=Path("+repr(remote)+"); d={n:json.loads((b/n).read_text()) for n in ['started.json','progress.json','complete.json','actual-header-inventory.json'] if (b/n).exists()}; d['observed_epoch']=time.time(); print(json.dumps(d))"
r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':M,'command':'python3 -c '+shlex.quote(code),'timeout_seconds':30}});x=un(r)
if x.get('exit_code')!=0:raise ValueError('metadata read did not complete')
snapshot=json.loads(x['stdout']);(A/'remote-progress-latest.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
for n in ('complete.json','actual-header-inventory.json','started.json'):
 if n in snapshot and not (A/n).exists():
  rr=un(c.request('tools/call',{'name':'read_file','arguments':{'mount_id':M,'path':'research/deepseek-r1-0528-qwen3-8b-readiness/assets/'+n,'offset':0,'limit':1000000}}));b=rr['content'].encode();assert hashlib.sha256(b).hexdigest()==rr['sha256'];(A/n).write_bytes(b)
f=snapshot.get('progress.json',{}).get('files',{});summary={n:({'status':v.get('status'),'bytes':v.get('bytes',v.get('bytes_received',v.get('partial_bytes'))),'attempts':len(v.get('attempts',[]))}) for n,v in f.items()}
print(json.dumps({'process_id':job['id'],'running':state['running'],'observed_epoch':snapshot['observed_epoch'],'started':snapshot.get('started.json'),'progress':summary,'complete_status':snapshot.get('complete.json',{}).get('status'),'actual_headers':snapshot.get('actual-header-inventory.json')},ensure_ascii=False))
