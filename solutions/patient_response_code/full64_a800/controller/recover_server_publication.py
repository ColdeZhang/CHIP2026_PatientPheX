"""Resume a failed entry publication before any original server start call.

One managed CPU process validates/publishes the identical original entry, then
execs the unchanged io_driver under the original phase/campaign deadline.
"""
from pathlib import Path
import argparse, hashlib, json, os, shlex, sys, time
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

REMOTE = '''from pathlib import Path
import hashlib,json,os,sys,time
p=Path(PARAM['root']);e=PARAM['entry'];n=PARAM['entry_name']
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
assert time.time()<e['absolute_deadline_unix']
assert hashlib.sha256((p/'controller-plan.json').read_bytes()).hexdigest()==e['plan_sha256']
assert hashlib.sha256((p/'io_driver.py').read_bytes()).hexdigest()==PARAM['io_sha256']
assert not (p/'control'/('export-server-epoch-%03d'%e['epoch'])).exists()
c=p/('export-publication-recovery-epoch-%03d'%e['epoch']);c.mkdir(exist_ok=False)
raw=(json.dumps(e,indent=2)+chr(10)).encode();assert hashlib.sha256(raw).hexdigest()==PARAM['entry_sha256']
f=p/n
if f.exists():assert f.read_bytes()==raw
else:
 with f.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
for d in [p,c]:
 fd=os.open(str(d),os.O_RDONLY);os.fsync(fd);os.close(fd)
proof=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),pid=os.getpid(),at_unix=time.time(),original_phase_deadline=e['absolute_deadline_unix'],no_new_phase_clock=True)
with (c/'published.json').open('x') as h:json.dump(proof,h,indent=2);h.write(chr(10));h.flush();os.fsync(h.fileno())
os.execv(PARAM['python'],[PARAM['python'],'-u',str(p/'io_driver.py'),'--plan',str(p/'controller-plan.json'),'--campaign',str(p/'campaign.json'),'--entry',str(f)])
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controller',type=Path,required=True)
    ap.add_argument('--epoch',type=int,choices=[1,2],required=True)
    ap.add_argument('--not-before',type=float,required=True);a=ap.parse_args()
    C=a.controller;tag=f'epoch-{a.epoch:03d}';plan=json.loads((C/'controller-plan.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert time.time()>=a.not_before
    assert not (C/(tag+'-server-start-tool.json')).exists() and not (C/(tag+'-server-job.json')).exists()
    old=json.loads((C/(tag+'-server-entry-copy.json')).read_text());assert old.get('isError')
    entry=json.loads((C/(tag+'-export-intent.json')).read_text())
    assert entry['phase']=='server' and entry['epoch']==a.epoch and entry['plan_sha256']==sha(C/'controller-plan.json')
    campaign=json.loads((C/'campaign.json').read_text())
    assert time.time()<entry['absolute_deadline_unix']<=campaign['absolute_deadline_unix']
    assert entry['absolute_deadline_unix']<=entry['started_unix']+600
    def save(name,x):
        with (C/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    params=dict(root=plan['remote_absolute'],entry=entry,entry_name=tag+'-export-intent.json',
                entry_sha256=sha(C/(tag+'-export-intent.json')),python=plan['python'],
                io_sha256=plan['payload_files']['io_driver.py']['sha256'])
    code='PARAM='+repr(params)+'\n'+REMOTE;compile(code,'recover_remote_server.py','exec')
    save(tag+'-server-recovery-intent.json',dict(at_unix=time.time(),source_sha256=sha(__file__),
         original_entry_sha256=params['entry_sha256'],original_failure_sha256=sha(C/(tag+'-server-entry-copy.json')),
         original_start_process_not_called=True,not_before=a.not_before,command_sha256=hashlib.sha256(code.encode()).hexdigest(),no_deadline_reset=True))
    client=Client();client.initialize()
    result=client.request('tools/call',dict(name='start_process',arguments=dict(mount_id=plan['mount_id'],
         command=shlex.join([plan['python'],'-u','-c',code]),cwd=plan['remote_relative'],
         env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})))
    save(tag+'-server-recovery-start-tool.json',result)
    if result.get('isError'):raise RuntimeError('Original recovery start unknown; inspect exact remote claim, do not repeat')
    job=result.get('structuredContent') or json.loads(result['content'][0]['text'])
    save(tag+'-server-job.json',job);print(json.dumps(job))


if __name__=='__main__':main()
