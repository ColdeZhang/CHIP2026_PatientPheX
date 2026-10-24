"""Close only the original, positively identified completed export server.

Keep the earlier failed read-only closure observation unchanged. No GPU,
training, HTTP restart, broad process match, or checkpoint modification.
"""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
CODE=r'''from pathlib import Path
import json,os,signal,time
observed=PARAM['observation']; actions=[]
for epoch,item in observed['observations'].items():
 for old in item['matching_exact_servers']:
  if old['argv'][0]=='/usr/bin/timeout':continue
  pid=old['pid'];proc=Path('/proc')/str(pid)
  if not proc.exists():actions.append(dict(pid=pid,already_absent=True));continue
  fd=os.pidfd_open(pid)
  try:
   argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   assert proc.stat().st_uid==os.getuid()==old['uid']==1045 and argv==old['argv']
   assert argv[argv.index('--checkpoint')+1]==PARAM['output']+'/epoch-%03d'%int(epoch)
   ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]
   signal.pidfd_send_signal(fd,signal.SIGTERM)
   actions.append(dict(pid=pid,uid=1045,start_ticks=ticks,exact_argv_before_signal=argv,signal='TERM',pidfd_identity=True))
  finally:os.close(fd)
cut=time.monotonic()+5
while time.monotonic()<cut:
 remaining=[]
 for item in observed['observations'].values():
  for old in item['matching_exact_servers']:
   proc=Path('/proc')/str(old['pid'])
   try:
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if argv==old['argv']:remaining.append(old['pid'])
   except FileNotFoundError:pass
 if not remaining:break
 time.sleep(.1)
print(json.dumps(dict(at_unix=time.time(),actions=actions,remaining_original_exact_processes=remaining,all_exact_servers_absent=not remaining,checkpoint_modified=False,new_services=0,new_model_or_quality_calls=0)))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);a=p.parse_args();C=a.controller
 obs=json.loads((C/'export-service-closure.json').read_text());assert not obs['all_exact_servers_absent'];assert (C/'campaign-complete.json').exists()
 plan=json.loads((C/'controller-plan.json').read_text())
 def save(n,x):
  with (C/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 save('export-service-cleanup-intent.json',dict(at_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),observation_sha256=hashlib.sha256((C/'export-service-closure.json').read_bytes()).hexdigest(),only_original_completed_export=True))
 c=Client();c.initialize();r=c.request('tools/call',dict(name='run_command',arguments=dict(mount_id=plan['mount_id'],command='python3 -c '+shlex.quote('PARAM='+repr(dict(observation=obs,output=plan['output_dir']))+'\n'+CODE),timeout_seconds=15)))
 save('export-service-cleanup-tool.json',r)
 if r.get('isError'):raise RuntimeError('Cleanup outcome unknown; inspect original PIDs, never blindly repeat signal')
 actual=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert actual['exit_code']==0
 d=json.loads(actual['stdout']);save('export-service-cleanup.json',d);assert d['all_exact_servers_absent'];print(json.dumps(d))
if __name__=='__main__':main()
