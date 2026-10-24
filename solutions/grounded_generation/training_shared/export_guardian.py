"""Bound only the two owned HTTP export children, never a training process.

Supplement to the frozen controller: no HTTP start, checkpoint copy or model
call. Stop an exact export child only after its original parent exits or its
original export deadline expires. Preserve the original campaign clock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    data=(json.dumps(value,indent=2)+'\n').encode()
    if p.exists():assert p.read_bytes()==data;return
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    d=p.parent
    while True:
        fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if d.parent==d:break
        d=d.parent

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        rest=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,state=rest[0],ppid=int(rest[1]),pgid=int(rest[2]),sid=int(rest[3]),
            start_ticks=int(rest[19]),uid=p.stat().st_uid,
            cmdline=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None

def alive(v):return v is not None and v['state'] not in ('Z','X')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--campaign',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();plan=read(a.plan);campaign=read(a.campaign)
    assert os.getuid()==1045 and not os.environ.get('CUDA_VISIBLE_DEVICES')
    assert sha(a.plan)==campaign['plan_sha256'] and campaign['cap_seconds']==6000 and campaign['grace_seconds']==30
    deadline=campaign['absolute_deadline_unix'];assert deadline==campaign['started_unix']+6000 and time.time()<deadline
    root=Path(plan['remote_absolute']);assert root.name=='formal-v1' and root.parent.name=='grounded-generation-full64-v1'
    output=a.output.resolve();assert output.parent.parent==root.parent and output.name=='actual' and output.parent.name=='export-guardian-v1'
    output.mkdir(parents=True,exist_ok=False);os.chmod(output,0o700)
    save(output/'started.json',dict(pid=os.getpid(),source_sha256=sha(__file__),plan_sha256=sha(a.plan),campaign_sha256=sha(a.campaign),
        original_campaign_deadline=deadline,started_unix=time.time(),GPU_hidden=True,models_or_HTTP_start_or_export_calls=0))
    seen={};done={}
    while time.time()<deadline+30 and len(done)<2:
        for epoch in (1,2):
            if epoch in done:continue
            control=root/'control'/('export-server-epoch-%03d'%epoch)
            if not (control/'child-started.json').exists() or not (control/'driver-started.json').exists():continue
            driver=read(control/'driver-started.json');child=read(control/'child-started.json');entry=read(root/('epoch-%03d-export-intent.json'%epoch))
            assert entry['phase']=='server' and entry['epoch']==epoch and entry['plan_sha256']==campaign['plan_sha256']
            limit=entry['absolute_deadline_unix'];assert limit<=deadline and limit<=entry['started_unix']+600
            expected=[plan['python'],'-u',str(root/'export_checkpoint_server.py'),'--plan',str(a.plan),
                '--checkpoint',str(Path(plan['output_dir'])/('epoch-%03d'%epoch)),'--deadline',str(limit)]
            assert driver['command']==expected and driver['absolute_deadline_unix']==limit
            assert sha(root/'export_checkpoint_server.py')==plan['payload_files']['export_checkpoint_server.py']['sha256']
            target=process(child['pid'])
            if not alive(target):
                done[epoch]=dict(status='already_exited_no_signal',pid=child['pid'],observed_unix=time.time())
                save(output/('epoch-%03d-closed.json'%epoch),done[epoch]);continue
            if target['cmdline']!=expected or target['uid']!=1045 or target['pgid']!=target['pid'] or target['sid']!=target['pid']:
                save(output/('epoch-%03d-identity-rejected.json'%epoch),dict(actual=target,expected=expected,no_signal=True))
                raise RuntimeError('Exact owned HTTP identity unavailable; never signal another process')
            if epoch not in seen:
                seen[epoch]=dict(child=target,parent=process(driver['pid']),original_export_deadline=limit)
                save(output/('epoch-%03d-identity.json'%epoch),seen[epoch])
            assert target['start_ticks']==seen[epoch]['child']['start_ticks']
            parent=process(driver['pid']);original_parent=seen[epoch]['parent']
            expected_parent=[plan['python'],'-u',str(root/'io_driver.py'),'--plan',str(a.plan),'--campaign',str(a.campaign),
                '--entry',str(root/('epoch-%03d-export-intent.json'%epoch))]
            parent_alive=alive(parent) and original_parent is not None and parent['start_ticks']==original_parent['start_ticks'] and parent['uid']==1045 and parent['cmdline']==expected_parent
            if parent_alive and time.time()<limit:continue
            reason='original_parent_exited' if not parent_alive else 'original_export_deadline_reached'
            fresh=process(child['pid'])
            if not alive(fresh):continue
            assert fresh['start_ticks']==target['start_ticks'] and fresh['cmdline']==expected and fresh['pgid']==fresh['pid']
            report=dict(reason=reason,child=fresh,parent=parent,original_export_deadline=limit,TERM_unix=time.time(),signals=[])
            try:os.killpg(fresh['pid'],signal.SIGTERM);report['signals'].append('TERM')
            except ProcessLookupError:pass
            kill_at=min(time.time()+30,limit+30,deadline+30)
            while time.time()<kill_at and alive(process(fresh['pid'])):time.sleep(.1)
            remaining=process(fresh['pid'])
            if alive(remaining):
                assert remaining['start_ticks']==fresh['start_ticks'] and remaining['cmdline']==expected and remaining['pgid']==fresh['pid']
                try:os.killpg(fresh['pid'],signal.SIGKILL);report['signals'].append('KILL')
                except ProcessLookupError:pass
            report['after']=process(fresh['pid']);report['exit_observed']=not alive(report['after']);report['closed_unix']=time.time();done[epoch]=report
            save(output/('epoch-%03d-closed.json'%epoch),report)
        if len(done)<2:time.sleep(min(1,max(0,deadline+30-time.time())))
    save(output/'complete.json',dict(at_unix=time.time(),closed_epochs=sorted(done),original_campaign_deadline=deadline,
        no_training_process_touched=True,no_model_or_export_restarted=True))

if __name__=='__main__':main()
