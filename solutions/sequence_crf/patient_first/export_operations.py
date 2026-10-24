"""Bounded, resumable-by-inspection checkpoint transfer; never restarts training."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','start-server','start-fetch','collect','stop-server'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mcp-client-dir', default='/workspace/patientphex-tools')
    args = parser.parse_args(); b = args.directory
    resource = json.loads((b/'resource.json').read_text())
    plan = json.loads((b/'export-plan.json').read_text())
    assert plan['cap_seconds'] == 180 and plan['TERM_grace_seconds'] == 10
    assert plan['checkpoint'].endswith('/formal2ep/epoch-002')
    sys.path.insert(0, args.mcp_client_dir)
    from mcp_client import Client
    client = Client(); client.initialize()
    def save(name, value):
        with (b/name).open('x') as handle:
            json.dump(value,handle,indent=2); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        descriptor = os.open(str(b), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    def call(mount, name, **kwargs):
        result = client.request('tools/call',dict(name=name,arguments=dict(mount_id=mount,**kwargs)))
        if result.get('isError'): raise RuntimeError(json.dumps(result))
        return result['structuredContent']
    def run(mount, code, timeout=30):
        result = call(mount,'run_command',command='python3 -c '+shlex.quote(code),cwd='.',timeout_seconds=timeout)
        assert result['exit_code'] == 0,result
        return result
    def stop_server():
        assert (b/'export-server-job.json').exists()
        if (b/'export-server-stop.json').exists(): return
        job=json.loads((b/'export-server-job.json').read_text())
        result=call(resource['mount'],'stop_process',process_id=job['id'])
        save('export-server-stop.json',result)
    source_control=resource['control_root']+'/export'
    target_tools=resource['persistent_tools']
    project='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
    target_control=project+'/sequence-round2/patient-first-formal/control/export'
    target_fetch=project+'/'+target_tools+'/fetch_checkpoint.py'
    if args.mode=='prepare':
        assert not (b/'export-fetcher-deployment-intent.json').exists()
        item=plan['fetch_source'];raw=Path(item['path']).read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        save('export-fetcher-deployment-intent.json',dict(source=item,target=target_fetch,no_GPU=True))
        code="from pathlib import Path;import shutil,json;p=Path(ROOT);assert not p.exists();assert shutil.disk_usage('.').free>=FREE;p.mkdir(parents=True);print(json.dumps(dict(free=shutil.disk_usage('.').free,no_GPU=True)))".replace('ROOT',repr(target_tools)).replace('FREE',str(resource['minimum_free_persistent_bytes']))
        run(resource['persistent_mount'],code)
        call(resource['persistent_mount'],'write_file',path=target_tools+'/fetch_checkpoint.py',encoding='base64',content=base64.b64encode(raw).decode())
        code="from pathlib import Path;import hashlib,json;p=Path(FILE);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA;compile(raw,str(p),'exec');print(json.dumps(dict(sha256=SHA,no_GPU=True)))".replace('FILE',repr(target_fetch)).replace('SHA',repr(item['sha256']))
        result=run(resource['persistent_mount'],code);save('export-fetcher-deployment.json',result);print(json.dumps(result));return
    if args.mode=='stop-server': stop_server();return
    if args.mode=='start-server':
        assert (b/'export-fetcher-deployment.json').exists() and not (b/'export-intent.json').exists()
        training=json.loads((b/'training-collection.json').read_text())
        assert training['wrapper']['returncode']==0 and not training['wrapper'].get('endpoint_error')
        deadline=time.time()+180
        save('export-intent.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),absolute_deadline_unix=deadline,
             plan_sha256=hashlib.sha256((b/'export-plan.json').read_bytes()).hexdigest(),once_only=True,
             unknown_outcome='Inspect original job/logs, never issue a second server start.'))
        source=resource['remote_preparation_absolute']+'/export_checkpoint_server.py'
        command=[resource['python'],'-u',source,'--checkpoint',plan['checkpoint']]
        code='''from pathlib import Path
import hashlib,time,subprocess,json
assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()==SOURCE_SHA
p=Path(CONTROL);assert not p.exists();p.mkdir(parents=True)
remaining=int(DEADLINE-time.time());assert remaining>0
with (p/'server.log').open('xb') as out,(p/'server.err').open('xb') as err:
 done=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(remaining)+'s']+COMMAND,stdout=out,stderr=err)
(p/'server-exit.json').write_text(json.dumps(dict(returncode=done.returncode,at_unix=time.time()))+chr(10))
'''.replace('SOURCE_SHA',repr(plan['server_source']['sha256'])).replace('SOURCE',repr(source)).replace('CONTROL',repr(source_control)).replace('DEADLINE',repr(deadline)).replace('COMMAND',repr(command))
        job=call(resource['mount'],'start_process',command='python3 -u -c '+shlex.quote(code),cwd='.',env={'CUDA_VISIBLE_DEVICES':''})
        save('export-server-job.json',job);print(json.dumps(job));return
    intent=json.loads((b/'export-intent.json').read_text());deadline=intent['absolute_deadline_unix']
    if args.mode=='start-fetch':
        assert not (b/'export-fetch-intent.json').exists() and time.time()<deadline
        code="from pathlib import Path;import json;p=Path(LOG);lines=p.read_text().splitlines() if p.exists() else [];print(json.dumps(dict(ready=next((json.loads(s) for s in lines if s.startswith('{')),None))))".replace('LOG',repr(source_control+'/server.log'))
        result=run(resource['mount'],code);ready=json.loads(result['stdout'])['ready']
        if ready is None:print(json.dumps(dict(server_not_ready_yet=True)));return
        assert ready['ready'] and ready['manifest']['completed_epochs']==2 and ready['manifest']['global_step']==32
        training=json.loads((b/'training-collection.json').read_text())
        assert ready['manifest']['checkpoint_sha256']==training['wrapper']['final_checkpoint_metadata_sha256']
        assert ready['manifest']['files']['_run/training_complete.json']['sha256']==training['wrapper']['training_complete_sha256']
        save('export-server-ready.json',ready)
        argv=[resource['persistent_python'],'-u',target_fetch,'--url','http://'+resource['source_http_private_address']+':'+str(ready['port']),
              '--manifest-sha256',ready['manifest_sha256'],'--destination',plan['destination'],'--deadline',str(deadline)]
        code='''from pathlib import Path
import hashlib,time,subprocess,json
assert hashlib.sha256(Path(FETCH).read_bytes()).hexdigest()==FETCH_SHA
p=Path(CONTROL);assert not p.exists();p.mkdir(parents=True)
remaining=int(DEADLINE-time.time());assert remaining>0
with (p/'fetch.log').open('xb') as out,(p/'fetch.err').open('xb') as err:
 done=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(remaining)+'s']+ARGV,stdout=out,stderr=err)
(p/'fetch-exit.json').write_text(json.dumps(dict(returncode=done.returncode,at_unix=time.time()))+chr(10))
'''.replace('FETCH_SHA',repr(plan['fetch_source']['sha256'])).replace('FETCH',repr(target_fetch)).replace('CONTROL',repr(target_control)).replace('DEADLINE',repr(deadline)).replace('ARGV',repr(argv))
        save('export-fetch-intent.json',dict(argv=argv,absolute_deadline_unix=deadline,once_only=True))
        job=call(resource['persistent_mount'],'start_process',command='python3 -u -c '+shlex.quote(code),cwd='.',env={'CUDA_VISIBLE_DEVICES':''})
        save('export-fetch-job.json',job);print(json.dumps(job));return
    assert (b/'export-fetch-intent.json').exists() and not (b/'final-export-collection.json').exists()
    code='''from pathlib import Path
import json,hashlib,time
p=Path(CONTROL);exitfile=p/'fetch-exit.json'
if not exitfile.exists():print(json.dumps(dict(complete=False)))
else:
 result=json.loads(exitfile.read_text());assert result['returncode']==0,(result,(p/'fetch.err').read_text())
 root=Path(DESTINATION);raw=(root/'export-ack-epoch-002.json').read_bytes();ack=json.loads(raw)
 assert ack['status']=='persistent_checkpoint_verified' and ack['completed_epochs']==2 and ack['global_step']==32
 assert ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files']
 for name,item in ack['files'].items():
  path=root/name;assert path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes']
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  assert h.hexdigest()==item['sha256']
 print(json.dumps(dict(complete=True,ack=ack,ack_sha256=hashlib.sha256(raw).hexdigest(),all_files_SHA_verified=True,completed_epoch=2,fetch_exit=result,verified_at_unix=time.time())))
'''.replace('CONTROL',repr(target_control)).replace('DESTINATION',repr(plan['destination']))
    result=run(resource['persistent_mount'],code,60);actual=json.loads(result['stdout'])
    if not actual['complete']:print(json.dumps(actual));return
    ready=json.loads((b/'export-server-ready.json').read_text())
    assert actual['ack']['files']==ready['manifest']['files'] and actual['ack']['transport_manifest_sha256']==ready['manifest_sha256']
    save('final-export-collection.json',actual)
    try:stop_server()
    finally:print(json.dumps(actual))


if __name__=='__main__':main()
