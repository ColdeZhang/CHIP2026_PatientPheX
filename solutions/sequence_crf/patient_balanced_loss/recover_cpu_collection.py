"""Read-only 32 KiB recovery of an already completed CPU contract directory."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

MOUNT = '476f81cf-09e7-445b-a7f0-d0e12b12e50f'
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
RELATIVE = 'sequence-round2/patient-balanced-alpha05/cpu-contract-v1/actual'
ROOT = PROJECT + '/' + RELATIVE
RESULT_SHA = 'cd96f25bca0005756b212a06b1277b9950e4f7f8ab07d0d988cc41d5a08c26f2'
CHUNK = 32768


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    p = Path(path).resolve()
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=sha(data))


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
        temporary = path.with_name(path.name + '.pending')
        if temporary.exists():
            assert temporary.read_bytes() == data, 'Partial pending file retained; inspect before continuation'
        else:
            with temporary.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        os.link(temporary, path)
        temporary.unlink()
    with path.open('rb') as f:
        os.fsync(f.fileno())
    sync(path.parent)


def save(path, value):
    publish(path, (json.dumps(value, indent=2) + '\n').encode())


def safe(name):
    p = PurePosixPath(name)
    assert name and p.as_posix() == name and not p.is_absolute()
    assert '..' not in p.parts and '\\' not in name and '\x00' not in name
    return p


def validate_inventory(value):
    assert value['root'] == ROOT
    assert value['files']['result.json'] == dict(bytes=4037, sha256=RESULT_SHA)
    required = {'launch.json', 'child-started.json', 'terminal.json', 'result.json',
                'observations.jsonl', 'selected-layouts.json', 'stdout.log', 'stderr.log'}
    assert required <= set(value['files']) and 'failure.json' not in value['files']
    for name, item in value['files'].items():
        safe(name)
        assert type(item['bytes']) is int and 0 <= item['bytes'] < 512 * 1024
        assert len(item['sha256']) == 64
    terminal = value['terminal']
    assert terminal['returncode'] == 0 and not terminal['timed_out'] and terminal['child_pid'] == 267289


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['inventory', 'collect'])
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--client-dir', type=Path, default=Path('/workspace/patientphex-tools'))
    a = ap.parse_args()
    os.umask(0o077)
    out = a.directory.resolve()
    recovery = out / 'read-recovery'
    recovery.mkdir(exist_ok=True)
    lock = (recovery / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for p in recovery.glob('*.backoff.json'):
        assert time.time() >= json.loads(p.read_text())['retry_not_before_unix']
    sys.path.insert(0, str(a.client_dir.resolve()))
    from mcp_client import Client
    client = Client()
    client.initialize()

    def call(name, args, tag):
        stamp = str(time.time_ns())
        prefix = recovery / (tag + '-' + stamp)
        save(Path(str(prefix) + '.intent.json'), dict(at_unix=time.time(), tool=name, arguments=args))
        try:
            response = client.request('tools/call', dict(name=name, arguments=args))
        except Exception as error:
            save(Path(str(prefix) + '.error.json'), dict(error_type=type(error).__name__, UNKNOWN=True, automatic_retry=False))
            raise RuntimeError('UNKNOWN saved; no automatic retry') from None
        rendered = json.dumps(response)
        for key in ['MCP_TOKEN', 'MCP_URL']:
            if os.environ.get(key):
                rendered = rendered.replace(os.environ[key], '[REDACTED]')
        response = json.loads(rendered)
        if response.get('isError'):
            save(Path(str(prefix) + '.error.json'), response)
            if 'LUNA_OFFLINE' in rendered:
                save(Path(str(prefix) + '.backoff.json'), dict(retry_not_before_unix=time.time() + 185, seconds=185))
            raise RuntimeError('Remote error preserved; no automatic retry')
        value = response.get('structuredContent')
        if value is None:
            value = json.loads(next(v['text'] for v in response['content'] if v['type'] == 'text'))
        summary = {k: v for k, v in value.items() if name != 'read_file' or k != 'content'}
        if name == 'read_file':
            summary['encoded_content_length'] = len(value['content'])
        save(Path(str(prefix) + '.response.json'), summary)
        return value

    if a.stage == 'inventory':
        code = """import json,hashlib;from pathlib import Path
root=Path(ROOT);terminal=json.loads((root/'terminal.json').read_text());assert terminal['returncode']==0 and not terminal['timed_out']
files={}
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():
  data=p.read_bytes();files[p.relative_to(root).as_posix()]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(root=str(root),files=files,terminal=terminal)))
""".replace('ROOT', repr(ROOT))
        value = call('run_command', dict(mount_id=MOUNT, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=30), 'inventory')
        assert value['exit_code'] == 0
        inventory = json.loads(value['stdout'])
        validate_inventory(inventory)
        path = recovery / ('inventory-' + str(time.time_ns()) + '.json')
        save(path, dict(at_unix=time.time(), mount=MOUNT, actual=inventory, source=ref(__file__)))
        print(json.dumps(dict(inventory=ref(path), files=len(inventory['files']))))
        return
    assert a.inventory is not None
    receipt = json.loads(a.inventory.read_text())
    assert receipt['mount'] == MOUNT
    inventory = receipt['actual']
    validate_inventory(inventory)
    binding = recovery / 'inventory-binding.json'
    if binding.exists():
        previous = json.loads(binding.read_text())
        assert previous['actual'] == inventory, 'Immutable endpoint inventory changed; preserve evidence and stop'
    else:
        save(binding, dict(actual=inventory, first_inventory=ref(a.inventory), mount=MOUNT))
    chunks = recovery / 'chunks'
    collected = out / 'collected'
    collected.mkdir(exist_ok=True)
    files = {}
    for name, item in sorted(inventory['files'].items()):
        relative = safe(name)
        target = collected / relative
        assert target.resolve().is_relative_to(collected)
        if target.exists():
            assert ref(target)['bytes'] == item['bytes'] and ref(target)['sha256'] == item['sha256']
        else:
            pieces = []
            for offset in range(0, item['bytes'], CHUNK):
                wanted = min(CHUNK, item['bytes'] - offset)
                piece = chunks / relative / (str(offset) + '.bin')
                meta = piece.with_suffix('.json')
                if piece.exists():
                    previous = json.loads(meta.read_text())
                    assert previous['remote_file'] == item and previous['offset'] == offset
                    assert previous['local'] == ref(piece) and previous['local']['bytes'] == wanted
                    data = piece.read_bytes()
                else:
                    value = call('read_file', dict(mount_id=MOUNT, path=RELATIVE + '/' + relative.as_posix(),
                        offset=offset, limit=wanted), 'read-' + name.replace('/', '_') + '-' + str(offset))
                    data = base64.b64decode(value['content']) if value.get('encoding') == 'base64' else value['content'].encode()
                    assert len(data) == wanted
                    publish(piece, data)
                    save(meta, dict(offset=offset, remote_file=item, local=ref(piece), remote_path=RELATIVE + '/' + name))
                pieces.append(data)
            data = b''.join(pieces)
            assert len(data) == item['bytes'] and sha(data) == item['sha256']
            publish(target, data)
        files[name] = ref(target)
        assert files[name]['bytes'] == item['bytes'] and files[name]['sha256'] == item['sha256']
        print(json.dumps(dict(collected=name, bytes=item['bytes'], complete_files=len(files), total_files=len(inventory['files']))), flush=True)
    package = json.loads((out / 'package.json').read_text())
    terminal = json.loads((collected / 'terminal.json').read_text())
    result = json.loads((collected / 'result.json').read_text())
    assert terminal == inventory['terminal']
    assert terminal['result']['sha256'] == RESULT_SHA and terminal['result']['bytes'] == 4037
    assert result['status'] == 'PASS_CPU_SYNTHETIC_LOSS_AND_LOGITS_GRADIENT_CONTRACT'
    for key, name in [('source', 'cpu_contract.py'), ('plan', 'cpu-contract-plan.json')]:
        source = next(item for item in package['members'] if item['name'] == name)
        assert result[key]['bytes'] == source['bytes'] and result[key]['sha256'] == source['sha256']
    for key, name in [('observations', 'observations.jsonl'), ('selected_layouts', 'selected-layouts.json')]:
        assert result[key]['bytes'] == files[name]['bytes'] and result[key]['sha256'] == files[name]['sha256']
    observations = [json.loads(line) for line in (collected / 'observations.jsonl').read_text().splitlines()]
    assert len(observations) == result['comparisons'] and all(o['passed'] for o in observations)
    save(out / 'collection.json', dict(schema='patient-balanced-CPU-contract-readonly-collection-v1',
        local=str(collected), files=files, all_actual_SHA_verified=True, original_remote=ROOT, mount=MOUNT,
        inventory=ref(a.inventory), inventory_binding=ref(binding), source=ref(__file__), package=ref(out / 'package.json'),
        files_and_ancestor_directories_fsynced=True, complete_recursive_directory=True,
        original_failure_receipts_preserved=True, observations_all_passed=True, observations=len(observations),
        original_CPU_returncode=terminal['returncode'], original_CPU_seconds=terminal['seconds'],
        original_child_pid=terminal['child_pid'], no_new_CPU_contract_execution=True, no_NN_or_model_or_repackage_or_HTTP=True))
    print(json.dumps(dict(collection=ref(out / 'collection.json'), result=files['result.json'], files=len(files))))


if __name__ == '__main__':
    main()
