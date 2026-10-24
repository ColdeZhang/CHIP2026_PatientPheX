#!/usr/bin/env python3
"""Recover a completed run when managed output receipts fail; never launch a job."""
import argparse,base64,datetime,hashlib,json,shlex,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;r=json.loads((b/'resource.json').read_text());job=json.loads((b/'job.json').read_text())
    assert not (b/'collection.json').exists() and not (b/'terminal').exists()
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    c=Client();c.initialize()
    def call(name,**kwargs):
        value=c.request('tools/call',{'name':name,'arguments':{'mount_id':r['mount'],**kwargs}});assert not value.get('isError'),value;return value['structuredContent']
    code='''from pathlib import Path
import hashlib,json,subprocess,datetime
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
p=json.loads(Path('formal-plan.json').read_text());complete=json.loads(Path('training-complete.json').read_text())
assert not Path('/proc/PID/cmdline').exists()
assert complete['plan_sha256']==sha('formal-plan.json') and len(complete['arms'])==2
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);assert p['gpu_uuid'] not in apps
artifacts=[]
for arm in complete['arms']:
 assert arm['steps']==256 and len(arm['epochs'])==8
 for field in ['final','probabilities']:
  x=arm[field];actual=dict(path=x['path'],bytes=Path(x['path']).stat().st_size,sha256=sha(x['path']));assert actual==x;artifacts.append(actual)
Path('artifact-actual-hashes.json').write_text(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,all_actual_files_exact=True),indent=2)+'\\n')
Path('gpu-release.json').write_text(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=p['gpu_uuid'],target_empty=True,original_pid_absent=True,remote_training_exit_code_observed=False),indent=2)+'\\n')
names=['training-complete.json','gpu-release.json','artifact-actual-hashes.json']+[m+'/'+n for m in ['uniform','pair'] for n in ['progress.json','complete.json','probabilities.json']]
print(json.dumps([dict(name=n,bytes=Path(n).stat().st_size,sha256=sha(n)) for n in names]))
'''.replace('PID',str(job['result']['pid']))
    t=time.monotonic();receipt=call('run_command',command=r['python']+' -c '+shlex.quote(code),cwd=r['remote'],env=r['CPU_env'],timeout_seconds=60)
    evidence={'elapsed_seconds':time.monotonic()-t,'arguments_scope':'Read original PID/complete files/GPU; hash actual final states and probabilities; write only own metadata','result':receipt};(b/'actual-artifact-recovery.json').write_text(json.dumps(evidence,indent=2)+'\n');assert receipt['exit_code']==0
    items=json.loads(receipt['stdout']);dest=b/'terminal';dest.mkdir()
    for item in items:
        parts=[]
        for offset in range(0,item['bytes'],512*1024):
            value=call('read_file',path=r['remote']+'/'+item['name'],offset=offset,limit=min(512*1024,item['bytes']-offset));parts.append(base64.b64decode(value['content']) if value['encoding']=='base64' else value['content'].encode())
        payload=b''.join(parts);assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256'];p=dest/item['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'items':items,'remote':r,'recovery_method':'Original PID absent, target GPU empty, both final8ep states complete and actual file SHAs; managed read_process_output returned UNKNOWN_OUTCOME, no remote training exitcode observed.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_and_optimizer_states_retained_remote':True}
    (b/'collection.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'complete':True,'small_files':len(items),'GPU_empty':True,'model_actual_hashes_verified':True}))

if __name__=='__main__':main()
