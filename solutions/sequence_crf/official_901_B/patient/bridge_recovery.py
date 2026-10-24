"""Explicit one-stage transport recovery; never invokes inference or repackages."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import sys
import time

SOURCE_MOUNT = '476f81cf-09e7-445b-a7f0-d0e12b12e50f'
RELAY_MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE = 'sequence-round2/official-901-B/patient/bridge-recovery'
RELAY = 'sequence-round2/official-901-B/patient-bridge-recovery'
SHA = '1df686ceaaf69e55913198482494533ede264dad2effa4561dd4ed4da9593e36'
SIZE = 897210


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path.read_bytes()))


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
    if path.exists():
        assert path.read_bytes() == data
        return
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    sync(path.parent)


def save(path, data):
    publish(path, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['deploy', 'launch', 'fetch', 'download', 'stop', 'finish'])
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--client-dir', type=Path, default=Path('/workspace/patientphex-tools'))
    a = ap.parse_args()
    os.umask(0o077)
    root = a.run_dir.resolve()
    out = root / 'bridge-recovery'
    out.mkdir(exist_ok=True)
    collection = root / 'inference-collection'
    lock = (collection / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = json.loads((collection / 'archive-binding.json').read_text())
    assert binding['mount'] == SOURCE_MOUNT
    assert binding['archive']['sha256'] == SHA and binding['archive']['bytes'] == SIZE
    token_path = out / 'ephemeral-token'
    token = token_path.read_text() if token_path.exists() else None

    def clean(value):
        text = json.dumps(value, ensure_ascii=False)
        for secret in [os.environ.get('MCP_TOKEN'), os.environ.get('MCP_URL'), token]:
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return json.loads(text)

    def call(name, args, tag):
        stamp = str(time.time_ns())
        save(out / (tag + '-' + stamp + '.intent.json'), dict(tool=name, arguments=clean(args), at_unix=time.time()))
        try:
            response = client.request('tools/call', dict(name=name, arguments=args))
        except Exception as exc:
            save(out / (tag + '-' + stamp + '.error.json'), dict(type=type(exc).__name__, unknown=True))
            raise RuntimeError('UNKNOWN; preserve state and inventory before retry') from None
        if response.get('isError'):
            save(out / (tag + '-' + stamp + '.error.json'), clean(response))
            if 'LUNA_OFFLINE' in json.dumps(response):
                save(out / (tag + '-' + stamp + '.backoff.json'), dict(retry_not_before_unix=time.time() + 180))
            raise RuntimeError('Tool error saved; no automatic retry')
        value = response.get('structuredContent')
        if value is None:
            value = json.loads(next(b['text'] for b in response['content'] if b['type'] == 'text'))
        summary = {k: v for k, v in value.items() if name != 'read_file' or k != 'content'}
        save(out / (tag + '-' + stamp + '.response.json'), clean(summary))
        return value

    def run(mount, code, tag, env=None):
        args = dict(mount_id=mount, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=40)
        if env:
            args['env'] = env
        value = call('run_command', args, tag)
        assert value['exit_code'] == 0, 'Remote failure saved; no automatic retry'
        return json.loads(value['stdout'])

    if a.stage != 'finish':
        for p in out.glob('*.backoff.json'):
            assert time.time() >= json.loads(p.read_text())['retry_not_before_unix']
        sys.path.insert(0, str(a.client_dir.resolve()))
        from mcp_client import Client
        client = Client()
        client.initialize()

    if a.stage == 'deploy':
        assert not (out / 'deployment.json').exists()
        source = Path(__file__).with_name('bridge_recovery_server.py')
        result = call('write_file', dict(mount_id=SOURCE_MOUNT, path=REMOTE + '/bridge_recovery_server.py',
                      content=source.read_text(), encoding='utf8', create_parents=True), 'deploy')
        assert result['sha256'] == ref(source)['sha256']
        save(out / 'deployment.json', dict(source=ref(source), remote=result))
        print(json.dumps(dict(deployed=True, source=ref(source))))
    elif a.stage == 'launch':
        assert (out / 'deployment.json').exists() and not (out / 'launch-intent.json').exists()
        assert token is None
        token = secrets.token_urlsafe(32)
        publish(token_path, token.encode())
        save(out / 'launch-intent.json', dict(at_unix=time.time(), ttl_seconds=180, archive_sha256=SHA,
             source_mount=SOURCE_MOUNT, exactly_one_start=True, authorization='parent transport-only private bridge'))
        result = call('start_process', dict(mount_id=SOURCE_MOUNT, cwd='.',
             command='python3 -u ' + REMOTE + '/bridge_recovery_server.py',
             env={'PATIENT_TRANSFER_TOKEN': token}), 'launch')
        save(out / 'launch.json', result)
        print(json.dumps(dict(http_launched=True, process_id=result['id'], pid=result['pid'], ttl_seconds=180)))
    elif a.stage == 'fetch':
        launch = json.loads((out / 'launch.json').read_text())
        assert token
        code = """import os,json,hashlib,time,urllib.request,subprocess;from pathlib import Path
ips=subprocess.check_output(['hostname','-I'],text=True).split();assert '10.5.2.100' in ips
root=Path(REMOTE);root.mkdir(parents=True,exist_ok=True);target=root/'patient-complete.zip';tmp=root/'patient-complete.pending';start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not target.exists():
 assert not tmp.exists(), 'Partial relay exists; inventory before any new fetch'
 req=urllib.request.Request('http://10.5.3.12:'+os.environ['PATIENT_TRANSFER_PORT']+'/frozen.zip',headers={'Authorization':'Bearer '+os.environ['PATIENT_TRANSFER_TOKEN']})
 with urllib.request.urlopen(req,timeout=30) as response:
  value=response.read(897211)
 assert len(value)==897210 and hashlib.sha256(value).hexdigest()==EXPECTED
 with tmp.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
 assert tmp.stat().st_size==897210 and sha(tmp)==EXPECTED
 os.link(tmp,target);tmp.unlink()
assert target.stat().st_size==897210 and sha(target)==EXPECTED
with target.open('rb') as f:os.fsync(f.fileno())
p=root.resolve()
while True:
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if p.parent==p:break
 p=p.parent
print(json.dumps(dict(path=str(target.resolve()),mount_relative_path=str(target),bytes=target.stat().st_size,sha256=sha(target),seconds=time.monotonic()-start,fsynced_and_reopened=True,private_ip_verified='10.5.2.100')))
""".replace('REMOTE', repr(RELAY)).replace('EXPECTED', repr(SHA))
        result = run(RELAY_MOUNT, code, 'fetch', {'PATIENT_TRANSFER_TOKEN': token, 'PATIENT_TRANSFER_PORT': str(launch['port'])})
        save(out / 'fetch.json', result)
        print(json.dumps(dict(relay_fetch_complete=True, **result)))
    elif a.stage == 'download':
        fetched = json.loads((out / 'fetch.json').read_text())
        assert fetched['sha256'] == SHA and fetched['bytes'] == SIZE
        chunks = collection / 'chunks'
        entries = []
        for offset in range(0, SIZE, 131072):
            size = min(131072, SIZE - offset)
            path = chunks / (str(offset) + '.bin')
            meta = chunks / (str(offset) + '.json')
            if path.exists():
                record = json.loads(meta.read_text())
                assert record['archive_sha256'] == SHA and record['offset'] == offset
                assert record['local'] == ref(path) and path.stat().st_size == size
            else:
                result = call('read_file', dict(mount_id=RELAY_MOUNT, path=RELAY + '/patient-complete.zip',
                              offset=offset, limit=size), 'read-' + str(offset))
                data = base64.b64decode(result['content']) if result.get('encoding') == 'base64' else result['content'].encode()
                assert len(data) == size
                publish(path, data)
                save(meta, dict(archive_sha256=SHA, offset=offset, requested_bytes=size,
                     local=ref(path), transport_source_mount=RELAY_MOUNT, original_source_mount=SOURCE_MOUNT))
            entries.append(ref(path))
            print(json.dumps(dict(downloaded_bytes=offset + size, total=SIZE)), flush=True)
        data = b''.join(Path(item['path']).read_bytes() for item in entries)
        assert len(data) == SIZE and sha(data) == SHA
        target = collection / 'patient-complete.zip'
        publish(target, data)
        assert ref(target)['sha256'] == SHA
        save(out / 'download.json', dict(archive=ref(target), chunks=entries, preserved_original_prefix_bytes=262144,
             original_prefix_chunks=entries[:2], all_bytes_exact=True, no_repackage=True, no_NN=True))
        print(json.dumps(dict(local_archive_complete=True, archive=ref(target))))
    elif a.stage == 'stop':
        launch = json.loads((out / 'launch.json').read_text())
        call('stop_process', dict(mount_id=SOURCE_MOUNT, process_id=launch['id']), 'stop')
        code = """import json;from pathlib import Path
root=Path(REMOTE);pid=PID;p=Path('/proc')/str(pid)/'stat';state=p.read_text().split(') ',1)[1].split()[0] if p.exists() else None
assert state in [None,'Z','X'], 'Owned HTTP process still running'
print(json.dumps(dict(pid=pid,process_state=state,no_running_process=True,started=json.loads((root/'started.json').read_text()),ended=json.loads((root/'ended.json').read_text()) if (root/'ended.json').exists() else None)))
""".replace('REMOTE', repr(REMOTE)).replace('PID', str(launch['pid']))
        result = run(SOURCE_MOUNT, code, 'stop-verification')
        save(out / 'cleanup.json', result)
        if token_path.exists():
            token_path.unlink()
            sync(out)
        print(json.dumps(dict(http_stopped=True, no_running_process=True, ephemeral_token_removed=True)))
    elif a.stage == 'finish':
        downloaded = json.loads((out / 'download.json').read_text())
        assert downloaded['archive'] == ref(collection / 'patient-complete.zip')
        cleanup = json.loads((out / 'cleanup.json').read_text())
        assert cleanup['no_running_process'] and not token_path.exists()
        receipts = ['source-inventory.json', 'deployment.json', 'launch.json', 'fetch.json', 'download.json', 'cleanup.json']
        report = dict(schema='patient-immutable-zip-private-bridge-v1', original_binding=ref(collection / 'archive-binding.json'),
             original_archive=binding['archive'], original_mount=SOURCE_MOUNT, relay_mount=RELAY_MOUNT,
             local_archive=downloaded['archive'], preserved_original_prefix_bytes=262144,
             member_index_sha256=binding['member_index_sha256'], no_repackage=True, no_NN=True,
             original_model_and_inference_unchanged=True, http_stopped=True, ephemeral_token_removed=True,
             receipts={name: ref(out / name) for name in receipts},
             sources=[ref(__file__), ref(Path(__file__).with_name('bridge_recovery_server.py'))])
        save(out / 'transport-receipt.json', report)
        print(json.dumps(dict(transport_receipt=ref(out / 'transport-receipt.json'), archive=downloaded['archive'])))


if __name__ == '__main__':
    main()
