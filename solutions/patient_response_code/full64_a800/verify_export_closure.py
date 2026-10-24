"""One read-only check of both original checkpoint HTTP services after ACKs.

Does not stop processes, publish files remotely, load models, or query a GPU.
An UNKNOWN is preserved as an observation failure, never a model failure.
"""
from pathlib import Path
import hashlib, json, os, shlex, sys, time
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

REMOTE = r'''from pathlib import Path
import json,os,time
p=Path(PARAM['root']); observations={};script=str(p/'export_checkpoint_server.py')
for epoch in (1,2):
 control=p/'control'/('export-server-epoch-%03d'%epoch)
 child=json.loads((control/'child-started.json').read_text())
 cp=PARAM['output']+'/epoch-%03d'%epoch
 managed=PARAM['managed'][str(epoch)]
 matching=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   argv=(proc/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
   if script in argv and cp in argv and '--checkpoint' in argv and '--plan' in argv and str(p/'controller-plan.json') in argv:
    matching.append(dict(pid=int(proc.name),argv=argv,uid=proc.stat().st_uid))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 observations[str(epoch)]=dict(managed_pid=managed,managed_proc_exists=Path('/proc/%d'%managed).exists(),timeout_child_pid=child['pid'],timeout_child_proc_exists=Path('/proc/%d'%child['pid']).exists(),matching_exact_servers=matching)
print(json.dumps(dict(at_unix=time.time(),observations=observations,all_exact_servers_absent=all(not x['matching_exact_servers'] for x in observations.values()),signals_sent=0,new_services=0,new_model_or_quality_calls=0)))
'''

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--controller',type=Path,required=True);a=parser.parse_args();C=a.controller
 plan=json.loads((C/'controller-plan.json').read_text());assert (C/'campaign-complete.json').exists()
 for e in (1,2):assert (C/f'epoch-{e:03d}-export-ack.json').exists()
 def save(name,value):
  with (C/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 params=dict(root=plan['remote_absolute'],output=plan['output_dir'],managed={str(e):json.loads((C/f'epoch-{e:03d}-server-job.json').read_text())['pid'] for e in (1,2)})
 assert not (C/'export-service-closure-intent.json').exists()
 save('export-service-closure-intent.json',dict(at_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),params=params,remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),readonly=True))
 client=Client();client.initialize()
 result=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id='69b991ee-183f-4add-af1a-579af92479c9',command='python3 -c '+shlex.quote('PARAM='+repr(params)+'\n'+REMOTE),timeout_seconds=15)))
 save('export-service-closure-tool.json',result)
 if result.get('isError'):raise RuntimeError('Original read-only outcome unknown; retain receipt and recover only this read')
 actual=result.get('structuredContent') or json.loads(result['content'][0]['text']);assert actual['exit_code']==0
 data=json.loads(actual['stdout']);save('export-service-closure.json',data)
 assert data['all_exact_servers_absent'], 'Existing owned server requires explicit identity-checked cleanup; no signal sent here'
 print(json.dumps(data))
if __name__=='__main__':main()
