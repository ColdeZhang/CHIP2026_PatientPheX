"""Recover only final metadata acknowledgement after both durable checkpoints.

Read the original source ACK first. Existing bytes must match; only an absent
file is published, with the same original bytes. No trainer or transfer starts.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

CODE='''from pathlib import Path
import base64,hashlib,json,os,time
p=Path(P['path']);b=base64.b64decode(P['raw']);assert os.getuid()==1045
assert time.time()<P['deadline'] and hashlib.sha256(b).hexdigest()==P['sha256']
assert hashlib.sha256((p.parent/'epoch-002/checkpoint.json').read_bytes()).hexdigest()==P['checkpoint_sha256']
existed=p.exists()
if existed:assert not p.is_symlink() and p.read_bytes()==b
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with p.open('rb') as f:os.fsync(f.fileno())
fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
assert p.read_bytes()==b
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),preexisting_exact=existed,created_only_if_absent=not existed,no_model_or_transfer=True,at_unix=time.time())))
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['publish','stop','finish'])
    ap.add_argument('--controller',type=Path,required=True);a=ap.parse_args();C=a.controller
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=json.loads((C/'controller-plan.json').read_text());campaign=json.loads((C/'campaign.json').read_text())
    assert campaign['plan_sha256']==sha(C/'controller-plan.json') and time.time()<campaign['absolute_deadline_unix']
    for epoch in [1,2]:
        ack=json.loads((C/f'epoch-{epoch:03d}-export-ack.json').read_text())
        verified=json.loads((C/f'epoch-{epoch:03d}-export-verification.json').read_text())
        assert verified['all_files_reopened_SHA'] and verified['ack']==ack and ack['global_step']==34*epoch
    def save(n,j):
        with (C/n).open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    def call(name,args,label):
        client=Client();client.initialize();r=client.request('tools/call',dict(name=name,arguments=dict(mount_id=plan['mount_id'],**args)));save(label,r)
        if r.get('isError'):raise RuntimeError('Preserved original recovery outcome; inspect before any further mutation')
        return r.get('structuredContent') or json.loads(r['content'][0]['text'])
    if a.action=='publish':
        assert not (C/'epoch-002-source-ack-copy.json').exists()
        original=json.loads((C/'epoch-002-source-ack-tool.json').read_text());assert original.get('isError')
        raw=(C/'epoch-002-export-ack.json').read_bytes();ack=json.loads(raw)
        params=dict(path=plan['output_dir']+'/export-ack-epoch-002.json',raw=base64.b64encode(raw).decode(),
                    sha256=sha(C/'epoch-002-export-ack.json'),checkpoint_sha256=ack['checkpoint_sha256'],deadline=campaign['absolute_deadline_unix'])
        code='P='+repr(params)+'\n'+CODE;compile(code,'recover_ack_remote.py','exec')
        stamp=str(time.time_ns());save('final-ack-recovery-'+stamp+'-intent.json',dict(at_unix=time.time(),source_sha256=sha(__file__),ACK_sha256=params['sha256'],original_failed_publication=sha(C/'epoch-002-source-ack-tool.json'),read_compare_before_any_write=True,no_new_model_or_transfer=True))
        r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=20),'final-ack-recovery-'+stamp+'-tool.json');assert r['exit_code']==0
        proof=json.loads(r['stdout']);assert proof['sha256']==params['sha256'] and proof['bytes']==len(raw)
        save('epoch-002-source-ack-copy.json',proof);print(json.dumps(proof))
    elif a.action=='stop':
        assert (C/'epoch-002-source-ack-copy.json').exists() and not (C/'epoch-002-server-stop-tool.json').exists()
        job=json.loads((C/'epoch-002-server-job.json').read_text())
        save('final-server-stop-intent.json',dict(at_unix=time.time(),original_job_id=job['id'],no_second_stop=True))
        r=call('stop_process',dict(process_id=job['id']),'epoch-002-server-stop-tool.json');save('epoch-002-server-stop.json',r);print(json.dumps(r))
    else:
        assert (C/'epoch-002-server-stop.json').exists()
        for epoch in [1,2]:assert json.loads((C/f'epoch-{epoch:03d}-source-ack-copy.json').read_text())['sha256']==sha(C/f'epoch-{epoch:03d}-export-ack.json')
        done=dict(status='TWO_COMPLETE_EPOCHS_68_STEPS_DURABLE_ACKS_VERIFIED',at_epoch=time.time(),campaign=campaign,epoch1_ack_sha256=sha(C/'epoch-001-export-ack.json'),epoch2_ack_sha256=sha(C/'epoch-002-export-ack.json'),no_quality=True,final_metadata_recovery_source_sha256=sha(__file__))
        save('campaign-complete.json',done);print(json.dumps(done))


if __name__=='__main__':main()
