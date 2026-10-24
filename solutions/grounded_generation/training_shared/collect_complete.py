"""Freeze terminal formal metadata and actual checkpoint locators; never replay training."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time
import zipfile

MOUNT = '69b991ee-183f-4add-af1a-579af92479c9'
RELATIVE = 'grounded-generation-full64-v1/formal-v1'
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REMOTE = PROJECT + '/' + RELATIVE
CHUNK = 32768


def ref(path):
    p = Path(path).resolve()
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def sync(path):
    while True:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if path.parent == path:
            break
        path = path.parent


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.is_symlink()
    if path.exists():
        assert path.read_bytes() == data
    else:
        pending = path.with_name(path.name + '.pending')
        if pending.exists():
            assert pending.read_bytes() == data, 'Keep incomplete bytes for explicit recovery'
        else:
            with pending.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        os.link(pending, path)
        pending.unlink()
    with path.open('rb') as f:
        os.fsync(f.fileno())
    sync(path.parent)


def save(path, value):
    publish(path, (json.dumps(value, indent=2) + '\n').encode())


def safe(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and p.as_posix() == name
    assert '..' not in p.parts and '\\' not in name and '\x00' not in name
    return p


REMOTE_CODE = r'''
import hashlib,json,os,stat,zipfile
from pathlib import Path,PurePosixPath
cfg=json.loads(PAYLOAD);root=Path(cfg['root']);output=Path(cfg['output']);bundle=root/'.collection';archive=bundle/'training-metadata.zip'
def sha(data):return hashlib.sha256(data).hexdigest()
def ref(p):
 data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=sha(data))
def sync(p):
 while True:
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if p.parent==p:break
  p=p.parent
def safe(name):
 p=PurePosixPath(name);assert name and not p.is_absolute() and p.as_posix()==name and '..' not in p.parts and chr(92) not in name and chr(0) not in name
 return p
def put(p,data):
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 sync(p.parent)
def verify_archive():
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();assert len(names)==len(set(names))
  index_bytes=z.read('member-index.json');index=json.loads(index_bytes)
  assert index['package']==cfg['package_reference'] and index['remote_root']==str(root)
  assert set(names)==set(index['files'])|{'member-index.json'}
  for name,item in index['files'].items():
   safe(name);info=z.getinfo(name);assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
   value=z.read(name);assert len(value)==item['bytes'] and sha(value)==item['sha256']
  return dict(status='frozen',archive=ref(archive),member_index_sha256=sha(index_bytes),members=len(index['files']),wrapper=index['wrapper'],child_exit=index['child_exit'])
def endpoint():
 plan=json.loads((root/'controller-plan.json').read_text());assert sha((root/'controller-plan.json').read_bytes())==cfg['controller_plan_sha256']
 assert cfg['campaign_complete']['status']=='TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
 wrappers={};children=[]
 for epoch in (1,2):
  control=root/'control'/('epoch-%03d'%epoch);wrapper_path=control/'wrapper-complete.json'
  if not wrapper_path.exists():return dict(status='not_terminal',reason='original epoch wrapper missing',epoch=epoch)
  wrapper=json.loads(wrapper_path.read_text());driver=json.loads((control/'driver-started.json').read_text());child=json.loads((control/'child-started.json').read_text())
  assert wrapper['returncode']==0 and not wrapper.get('timed_out') and 'endpoint_error' not in wrapper
  assert driver['plan_sha256']==cfg['controller_plan_sha256'] and driver['absolute_deadline_unix']==cfg['campaign_complete']['campaign']['absolute_deadline_unix']
  pid=child['pid'];proc=Path('/proc')/str(pid)/'stat';state=proc.read_text().split(') ',1)[1].split()[0] if proc.exists() else None
  if state not in (None,'Z','X'):return dict(status='not_terminal',reason='original child still running',child_pid=pid,process_state=state)
  children.append(dict(epoch=epoch,pid=pid,state=state));wrappers[str(epoch)]=wrapper
  ack_path=output/('export-ack-epoch-%03d.json'%epoch);assert sha(ack_path.read_bytes())==cfg['ack_sha256'][str(epoch)]
  ack=json.loads(ack_path.read_text());assert ack['status']=='persistent_checkpoint_verified' and ack['global_step']==22*epoch
  assert ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files'] and ack['destination_directories_fsynced']
  checkpoint=output/('epoch-%03d'%epoch);meta=json.loads((checkpoint/'checkpoint.json').read_text())
  assert ack['checkpoint_sha256']==sha((checkpoint/'checkpoint.json').read_bytes())==wrapper['checkpoint_sha256']
  assert meta['completed_epochs']==epoch and meta['global_step']==22*epoch
  assert meta['fingerprint']['source_sha256']==plan['trainer_sha256']
 complete=json.loads((output/'training_complete.json').read_text());assert complete['epochs']==2 and complete['rows']==173 and complete['global_step']==44
 assert complete['source_sha256']==plan['trainer_sha256'] and not complete['probe_state_used']
 assert wrappers['2']['training_complete_sha256']==sha((output/'training_complete.json').read_bytes())
 summary=dict(returncode=0,timed_out=False,all_epochs_persistent_and_readback_verified=True,epochs=wrappers,campaign_complete=cfg['campaign_complete'],derived_from_original_two_wrappers_and_actual_ACKs=True)
 return dict(status='terminal',wrapper=summary,child_exit=dict(children=children,no_running_process=True))
if archive.exists():
 assert not archive.is_symlink();print(json.dumps(verify_archive()));raise SystemExit(0)
state=endpoint()
if state['status']!='terminal':print(json.dumps(state));raise SystemExit(0)
if bundle.exists():
 print(json.dumps(dict(status='incomplete_freeze_preserved',files=[p.name for p in bundle.iterdir()],no_repackage=True)));raise SystemExit(0)
if cfg['mode']=='inventory':
 print(json.dumps(dict(state,archive_exists=False)));raise SystemExit(0)
assert cfg['mode']=='freeze'
for name,item in cfg['members'].items():
 safe(name)
 p=root/name;assert not p.is_symlink();actual=ref(p)
 assert actual['bytes']==item['bytes'] and actual['sha256']==item['sha256']
files={};external_checkpoints={}
for parent,prefix in [(root,''),(output,'output/')]:
 for p in sorted(parent.rglob('*')):
  relative=prefix+p.relative_to(parent).as_posix()
  if parent==root and (relative=='.collection' or relative.startswith('.collection/')):continue
  assert not p.is_symlink()
  if not p.is_file():continue
  if parent==root and (relative in cfg['members'] or p.name.startswith('part-') or p.name=='deployment.tar.gz'):continue
  safe(relative)
  with p.open('rb') as f:os.fsync(f.fileno())
  sync(p.parent);item=ref(p)
  if p.name in ('training_state.pt','adapter_model.safetensors'):
   assert parent==output
   epoch=int(p.relative_to(output).parts[0].split('-')[1]);ack=json.loads((output/('export-ack-epoch-%03d.json'%epoch)).read_text())
   member='/'.join(p.relative_to(output).parts[1:]);assert ack['files'][member]=={k:item[k] for k in ('bytes','sha256')}
   external_checkpoints[relative]=dict(bytes=item['bytes'],sha256=item['sha256'],original_SHM_path=item['path'],persistent_path=ack['persistent_directory']+'/'+member,persistent_mount_id=ack['persistent_mount_id'],actual_ACK_sha256=cfg['ack_sha256'][str(epoch)],not_downloaded=True,source_actual_fsynced_and_reopened_SHA=True)
  else:files[relative]=dict(bytes=item['bytes'],sha256=item['sha256'],original_path=item['path'])
assert {'control/epoch-001/wrapper-complete.json','control/epoch-002/wrapper-complete.json','output/training_complete.json','output/export-ack-epoch-001.json','output/export-ack-epoch-002.json'}<=set(files)
assert 'member-index.json' not in files
assert set(external_checkpoints)=={'output/epoch-%03d/'%epoch+name for epoch in (1,2) for name in ('training_state.pt','adapter/adapter_model.safetensors')}
index=dict(schema='grounded-full64-twoepoch-terminal-members-v1',package=cfg['package_reference'],remote_root=str(root),files=files,wrapper=state['wrapper'],child_exit=state['child_exit'],exclusion_policy='Frozen deployment inputs/parts excluded; large checkpoint weights and fullstate kept at exact persistent original locators with newly reopened SHA, not downloaded; all other outputs included',external_checkpoints=external_checkpoints,original_files_fsynced=True)
index_bytes=(json.dumps(index,indent=2)+chr(10)).encode()
bundle.mkdir(exist_ok=False);sync(root)
put(bundle/'member-index.json',index_bytes)
pending=bundle/'training-metadata.pending.zip'
with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED) as z:
 z.writestr('member-index.json',index_bytes)
 for name,item in files.items():
  value=Path(item['original_path']).read_bytes();assert len(value)==item['bytes'] and sha(value)==item['sha256'];z.writestr(name,value)
with pending.open('rb') as f:os.fsync(f.fileno())
os.link(pending,archive);pending.unlink();sync(bundle)
receipt=verify_archive();put(bundle/'archive.json',(json.dumps(receipt,indent=2)+chr(10)).encode())
print(json.dumps(receipt))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--client-dir', type=Path, default=Path('/workspace/patientphex-tools'))
    ap.add_argument('--inventory-only', action='store_true')
    a = ap.parse_args()
    assert __debug__
    os.umask(0o077)
    root = a.directory.resolve()
    package_path = root / 'collection-package.json'
    campaign_complete=json.loads((root/'campaign-complete.json').read_text())
    deployment=json.loads((root/'deployment-manifest.json').read_text());controller=json.loads((root/'controller-plan.json').read_text())
    assert campaign_complete['status']=='TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
    guardian_path=root.parent/'export-guardian-v1/terminal-collection.json'
    guardian=json.loads(guardian_path.read_text())
    assert guardian['both_original_HTTP_children_exited'] and guardian['collection_only_no_remote_or_signals']
    assert guardian['campaign_complete']==ref(root/'campaign-complete.json')
    assert guardian['actual']['started.json']['plan_sha256']==ref(root/'controller-plan.json')['sha256']
    # Freeze metadata only after the original two HTTP stdout writers have also exited.
    package=dict(mount=MOUNT,remote=REMOTE,members=deployment['files'],controller_plan=ref(root/'controller-plan.json'),
        campaign_complete=ref(root/'campaign-complete.json'),HTTP_guardian_terminal=ref(guardian_path),ACKs={str(e):ref(root/('epoch-%03d-export-ack.json'%e)) for e in (1,2)},source=ref(__file__))
    for e in (1,2):assert campaign_complete['epoch%d_ack_sha256'%e]==package['ACKs'][str(e)]['sha256']
    save(package_path,package)
    assert package['mount'] == MOUNT and package['remote'] == REMOTE
    assert isinstance(package['members'], dict) and 'formal-plan.json' in package['members']
    for name in package['members']: safe(name)
    transfer = root / 'collection-transfer'
    transfer.mkdir(exist_ok=True)
    lock = (transfer / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in transfer.glob('*.backoff.json'):
        assert time.time() >= json.loads(path.read_text())['retry_not_before_unix']
    receipts = transfer / 'receipts'
    receipts.mkdir(exist_ok=True)
    sys.path.insert(0, str(a.client_dir.resolve()))
    from mcp_client import Client
    client = Client()
    try:
        client.initialize()
    except Exception as error:
        save(receipts / ('initialize-' + str(time.time_ns()) + '.error.json'), dict(error_type=type(error).__name__, UNKNOWN=True))
        raise RuntimeError('Initialization failed; original artifacts and partials preserved') from None

    def call(tool, arguments, tag):
        prefix = receipts / (tag + '-' + str(time.time_ns()))
        save(Path(str(prefix) + '.intent.json'), dict(tool=tool, arguments=arguments))
        try:
            response = client.request('tools/call', dict(name=tool, arguments=arguments))
        except Exception as error:
            save(Path(str(prefix) + '.error.json'), dict(error_type=type(error).__name__, UNKNOWN=True, no_automatic_retry=True))
            raise RuntimeError('UNKNOWN; next invocation inventories before continuation') from None
        text = json.dumps(response)
        for key in ['MCP_TOKEN', 'MCP_URL']:
            if os.environ.get(key):
                text = text.replace(os.environ[key], '[REDACTED]')
        response = json.loads(text)
        if response.get('isError'):
            save(Path(str(prefix) + '.error.json'), response)
            if 'LUNA_OFFLINE' in text:
                save(transfer / ('offline-' + str(time.time_ns()) + '.backoff.json'), dict(retry_not_before_unix=time.time() + 185, seconds=185))
            raise RuntimeError('Tool error preserved; no automatic retry or probe action')
        value = response.get('structuredContent')
        if value is None:
            value = json.loads(next(v['text'] for v in response['content'] if v['type'] == 'text'))
        summary = {k: v for k, v in value.items() if tool != 'read_file' or k != 'content'}
        if tool == 'read_file':
            summary['encoded_content_length'] = len(value['content'])
        save(Path(str(prefix) + '.response.json'), summary)
        return value

    def remote(mode):
        cfg = dict(root=REMOTE, output=controller['output_dir'], mode=mode, members=package['members'], package_reference=ref(package_path),controller_plan_sha256=package['controller_plan']['sha256'],campaign_complete=campaign_complete,ack_sha256={e:v['sha256'] for e,v in package['ACKs'].items()})
        code = REMOTE_CODE.replace('PAYLOAD', repr(json.dumps(cfg)))
        result = call('run_command', dict(mount_id=MOUNT, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=40), mode)
        assert result['exit_code'] == 0
        return json.loads(result['stdout'])

    # Always inventory first: this recovers UNKNOWN freeze responses without repackaging.
    ready = remote('inventory')
    if a.inventory_only:
        print(json.dumps(ready))
        return
    if ready['status'] == 'terminal' and ready.get('archive_exists') is False:
        ready = remote('freeze')
    if ready['status'] != 'frozen':
        print(json.dumps(ready))
        raise SystemExit('No completed immutable archive; evidence retained, no retries')
    binding = dict(remote_archive=ready['archive'], member_index_sha256=ready['member_index_sha256'],
                   package=ref(package_path), mount=MOUNT)
    save(transfer / 'archive-binding.json', binding)
    tool_path = Path(ready['archive']['path']).relative_to(Path(PROJECT)).as_posix()
    assert tool_path == RELATIVE + '/.collection/training-metadata.zip' and '..' not in PurePosixPath(tool_path).parts
    chunks = transfer / 'chunks'
    chunks.mkdir(exist_ok=True)
    pieces = []
    for offset in range(0, ready['archive']['bytes'], CHUNK):
        wanted = min(CHUNK, ready['archive']['bytes'] - offset)
        path = chunks / (str(offset) + '.bin')
        meta = chunks / (str(offset) + '.json')
        if path.exists():
            known = json.loads(meta.read_text())
            assert known['archive_sha256'] == ready['archive']['sha256'] and known['offset'] == offset
            assert known['local'] == ref(path) and path.stat().st_size == wanted
            data = path.read_bytes()
        else:
            value = call('read_file', dict(mount_id=MOUNT, path=tool_path, offset=offset, limit=wanted), 'read-' + str(offset))
            data = base64.b64decode(value['content']) if value.get('encoding') == 'base64' else value['content'].encode()
            assert len(data) == wanted
            publish(path, data)
            save(meta, dict(offset=offset, archive_sha256=ready['archive']['sha256'], local=ref(path)))
        pieces.append(data)
    data = b''.join(pieces)
    assert len(data) == ready['archive']['bytes'] and hashlib.sha256(data).hexdigest() == ready['archive']['sha256']
    archive = transfer / 'training-metadata.zip'
    publish(archive, data)
    terminal = root / 'terminal'
    terminal.mkdir(exist_ok=True)
    files = {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        index_bytes = z.read('member-index.json')
        assert hashlib.sha256(index_bytes).hexdigest() == ready['member_index_sha256']
        index = json.loads(index_bytes)
        assert index['package'] == ref(package_path) and index['remote_root'] == REMOTE
        assert set(names) == set(index['files']) | {'member-index.json'}
        for name, item in index['files'].items():
            relative = safe(name)
            info = z.getinfo(name)
            assert not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
            value = z.read(name)
            assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256']
            target = terminal / relative
            assert not target.is_symlink() and target.resolve().is_relative_to(terminal)
            publish(target, value)
            files[name] = ref(target)
        publish(terminal / 'member-index.json', index_bytes)
    wrapper = index['wrapper']
    assert wrapper == ready['wrapper'] == index['wrapper'] and index['child_exit']['no_running_process']
    report = dict(schema='grounded-full64-twoepoch-metadata-collection-v1', local=str(terminal), files=files,
        archive=ref(archive), remote_archive=ready['archive'], member_index=ref(terminal / 'member-index.json'),
        original_remote=REMOTE, mount=MOUNT, package=ref(package_path), source=ref(__file__),
        wrapper=wrapper, child_exit_before_freeze=index['child_exit'],
        formal_reported_success=(wrapper['returncode'] == 0 and not wrapper['timed_out'] and wrapper.get('all_epochs_persistent_and_readback_verified') is True and 'endpoint_error' not in wrapper),
        complete_only_collector_failure_outputs_remain_in_original_namespace=True, all_original_nonweight_output_files_included=True, actual_external_checkpoints=index['external_checkpoints'],
        all_actual_SHA_verified=True, files_and_ancestor_directories_fsynced=True, zip_safety_checked=True,
        read_file_chunk_limit=CHUNK, no_training_reexecution_or_GPU_query_or_kill=True)
    save(root / 'collection.json', report)
    print(json.dumps(dict(collection=ref(root / 'collection.json'), files=len(files),
                         formal_reported_success=report['formal_reported_success'])))


if __name__ == '__main__':
    main()
