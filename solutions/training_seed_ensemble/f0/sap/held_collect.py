"""Collect all frozen native CPU outcomes and actual terminal; never rerun the contract."""
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

MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
RELATIVE = 'training-seed-ensemble-v1/F0-Sap-v1'
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
cfg=json.loads(PAYLOAD);root=Path(cfg['root']);bundle=root/'.collection';archive=bundle/'held-query-output.zip'
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
 wrapper_path=root/'wrapper-complete.json'
 if not wrapper_path.exists():return dict(status='not_terminal',reason='wrapper-complete missing')
 wrapper=json.loads(wrapper_path.read_text());child=json.loads((root/'child-started.json').read_text());driver=json.loads((root/'driver-started.json').read_text())
 assert wrapper['child_pid']==child['child_pid'] and child['driver_pid']==driver['driver_pid']
 assert driver['plan_sha256']==cfg['members']['held-query-plan.json']['sha256']
 pid=wrapper['child_pid'];p=Path('/proc')/str(pid)/'stat';state=p.read_text().split(') ',1)[1].split()[0] if p.exists() else None
 if state not in (None,'Z','X'):return dict(status='not_terminal',reason='original child still running',child_pid=pid,process_state=state)
 if wrapper['returncode']==0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper:
  assert wrapper['complete_necessary_query_members_verified']
  assert (root/'output/complete.json').is_file()
 return dict(status='terminal',wrapper=wrapper,child_exit=dict(pid=pid,process_state=state,no_running_process=True))
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
files={};external_large_artifacts={}
for p in sorted(root.rglob('*')):
 relative=p.relative_to(root).as_posix()
 if relative=='.collection' or relative.startswith('.collection/'):continue
 assert not p.is_symlink()
 if not p.is_file():continue
 if relative in cfg['members'] or (p.parent==root and p.name.startswith('deployment') and p.name.endswith('.part')):continue
 safe(relative)
 with p.open('rb') as f:os.fsync(f.fileno())
 sync(p.parent);item=ref(p)
 if relative=='output/query-vectors.npy':
  external_large_artifacts[relative]=dict(bytes=item['bytes'],sha256=item['sha256'],original_path=item['path'],not_downloaded=True,actual_fsynced_and_reopened_SHA=True)
 else:files[relative]=dict(bytes=item['bytes'],sha256=item['sha256'],original_path=item['path'])
assert {'driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'}<=set(files)
assert 'member-index.json' not in files
assert set(external_large_artifacts)=={'output/query-vectors.npy'} if state['wrapper']['returncode']==0 else True
index=dict(schema='grounded-held-query-terminal-members-v1',package=cfg['package_reference'],remote_root=str(root),files=files,wrapper=state['wrapper'],child_exit=state['child_exit'],exclusion_policy='Only frozen deployment input/source members and deployment parts excluded; all actual numeric query scores/runtime/terminal included; original query vectors retained at exact external SHA/fsync locator',external_large_artifacts=external_large_artifacts,original_files_fsynced=True)
index_bytes=(json.dumps(index,indent=2)+chr(10)).encode()
bundle.mkdir(exist_ok=False);sync(root)
put(bundle/'member-index.json',index_bytes)
pending=bundle/'held-query-output.pending.zip'
with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED) as z:
 z.writestr('member-index.json',index_bytes)
 for name,item in files.items():
  value=(root/name).read_bytes();assert len(value)==item['bytes'] and sha(value)==item['sha256'];z.writestr(name,value)
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
    package_path = root / 'package.json'
    package = json.loads(package_path.read_text())
    assert package['mount'] == MOUNT and package['remote'] == REMOTE
    assert isinstance(package['members'], dict) and 'held-query-plan.json' in package['members']
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
        cfg = dict(root=REMOTE, mode=mode, members=package['members'], package_reference=ref(package_path))
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
    assert tool_path == RELATIVE + '/.collection/held-query-output.zip' and '..' not in PurePosixPath(tool_path).parts
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
    archive = transfer / 'held-query-output.zip'
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
    wrapper = json.loads((terminal / 'wrapper-complete.json').read_text())
    assert wrapper == ready['wrapper'] == index['wrapper'] and index['child_exit']['no_running_process']
    report = dict(schema='grounded-held-query-complete-collection-v1', local=str(terminal), files=files,
        archive=ref(archive), remote_archive=ready['archive'], member_index=ref(terminal / 'member-index.json'),
        original_remote=REMOTE, mount=MOUNT, package=ref(package_path), source=ref(__file__),
        wrapper=wrapper, child_exit_before_freeze=index['child_exit'],
        query_reported_success=(wrapper['returncode'] == 0 and not wrapper['timed_out'] and wrapper.get('complete_necessary_query_members_verified') is True and 'endpoint_error' not in wrapper),
        failed_or_timed_out_outputs_also_preserved=True, all_original_output_files_included=True, no_model_checkpoints_written=True,
        all_actual_SHA_verified=True, files_and_ancestor_directories_fsynced=True, zip_safety_checked=True,
        read_file_chunk_limit=CHUNK, no_inference_reexecution_or_GPU_query_or_kill=True)
    save(root / 'collection.json', report)
    print(json.dumps(dict(collection=ref(root / 'collection.json'), files=len(files),
                         query_reported_success=report['query_reported_success'])))


if __name__ == '__main__':
    main()
