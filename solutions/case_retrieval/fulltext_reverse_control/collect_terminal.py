"""Freeze once and transfer one ended fulltext-reverse job; never starts inference or scoring."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE_PREFIX = 'research/qwen35-fulltext-reverse-20261024'
FREEZE = r'''
import pathlib,json,hashlib,zipfile,os
p=pathlib.Path(REMOTE_PLACEHOLDER)
def ref(f):return dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
t=json.loads((p/'terminal.json').read_bytes());assert 'finished_utc' in t
members=dict(t['files']);members['terminal.json']=ref(p/'terminal.json')
for name,expected in members.items():
 q=pathlib.Path(name);assert not q.is_absolute() and '..' not in q.parts
 assert not (p/name).is_symlink() and ref(p/name)==expected
z=p/'terminal-artifacts.zip';m=p/'terminal-artifacts.manifest.json'
assert not z.exists() and not m.exists()
with zipfile.ZipFile(str(z),'x',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
 for name in sorted(members):out.write(str(p/name),name)
with z.open('rb') as f:os.fsync(f.fileno())
manifest=dict(archive=dict(path=str(z),**ref(z)),members=members,terminal_status=t['status'],terminal_exit=t.get('exit_code'),scope='Original terminal inventory plus terminal; no NN or scoring.')
with m.open('x') as f:json.dump(manifest,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(archive=manifest['archive'],manifest=dict(path=str(m),**ref(m)))))
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def unpack(result):
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--kind', choices=['probe', 'full80'], required=True)
    parser.add_argument('--existing', action='store_true', help='Resume only the already frozen archive; no server mutation.')
    args = parser.parse_args(); directory = args.directory.resolve()
    remote = REMOTE_PREFIX + '/' + args.kind + '-v1'
    freeze_code = FREEZE.replace('REMOTE_PLACEHOLDER', repr(remote))
    sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
    from mcp_client import Client
    client = Client(); client.initialize()
    receipt = directory / 'archive-freeze-receipt.json'
    if not args.existing:
        assert not receipt.exists()
        save(directory / 'collection-intent.json', dict(source_sha256=digest(Path(__file__).read_bytes()),
            remote_directory=remote, one_archive_only=True, no_inference_or_quality=True))
        try:
            result = client.request('tools/call', dict(name='run_command', arguments=dict(
                mount_id=MOUNT, command='python3 -c ' + shlex.quote(freeze_code), timeout_seconds=30)))
        except Exception as error:
            save(directory / 'archive-freeze-exception.json', dict(exception_type=type(error).__name__,
                remote_state_unknown=True, action='Read-only inventory required; do not freeze again.'))
            raise
        save(receipt, result)
    else:
        result = json.loads(receipt.read_bytes())
    assert not result.get('isError'), 'Original archive freeze error retained; no automatic retry.'
    unwrapped = unpack(result); assert unwrapped['exit_code'] == 0
    inventory = json.loads(unwrapped['stdout'])
    chunks = directory / 'transfer-chunks'; chunks.mkdir(exist_ok=True)
    for kind in ('archive', 'manifest'):
        expected = inventory[kind]; pieces = []
        for offset in range(0, expected['bytes'], 200000):
            count = min(200000, expected['bytes'] - offset)
            part = chunks / ('%s-%09d.bin' % (kind, offset))
            ack = part.with_suffix('.json')
            if part.exists():
                previous = json.loads(ack.read_bytes()); data = part.read_bytes()
                assert previous['bytes'] == len(data) == count and previous['sha256'] == digest(data)
            else:
                try:
                    result = client.request('tools/call', dict(name='read_file', arguments=dict(
                        mount_id=MOUNT, path=expected['path'], offset=offset, limit=count)))
                except Exception as error:
                    save(chunks / ('exception-%s-%09d.json' % (kind, offset)),
                        dict(exception_type=type(error).__name__, offset=offset, requested_bytes=count))
                    raise
                if result.get('isError'):
                    save(chunks / ('error-%s-%09d.json' % (kind, offset)), result)
                    raise RuntimeError('Original transfer error retained; no retry.')
                value = unpack(result)
                data = base64.b64decode(value['content']) if value.get('encoding') == 'base64' else value['content'].encode()
                assert len(data) == count
                with part.open('xb') as handle:
                    handle.write(data); handle.flush(); os.fsync(handle.fileno())
                save(ack, dict(offset=offset, bytes=len(data), sha256=digest(data)))
            pieces.append(data)
        data = b''.join(pieces)
        assert len(data) == expected['bytes'] and digest(data) == expected['sha256']
        target = directory / Path(expected['path']).name
        if target.exists():
            assert target.read_bytes() == data
        else:
            with target.open('xb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
    manifest = json.loads((directory / 'terminal-artifacts.manifest.json').read_bytes())
    output = directory / 'collected'; output.mkdir(exist_ok=True)
    with zipfile.ZipFile(directory / 'terminal-artifacts.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(manifest['members'])
        assert set(archive.namelist()) == set(manifest['members'])
        for name, expected in manifest['members'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data = archive.read(name); assert len(data) == expected['bytes'] and digest(data) == expected['sha256']
            path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert not path.is_symlink() and path.read_bytes() == data
            else:
                with path.open('xb') as handle:
                    handle.write(data); handle.flush(); os.fsync(handle.fileno())
    for path in sorted([p for p in directory.rglob('*') if p.is_dir()] + [directory], key=lambda p:len(p.parts), reverse=True):
        descriptor = os.open(str(path), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
    final = dict(status='ALL_ORIGINAL_TERMINAL_MEMBERS_RESTORED_SHA_SIZE', source_sha256=digest(Path(__file__).read_bytes()),
        archive=inventory['archive'], manifest=inventory['manifest'], members=len(manifest['members']),
        terminal_status=manifest['terminal_status'], no_new_NN_or_decode_parser_quality=True)
    save(directory / 'collection.json', final); print(json.dumps(final))


if __name__ == '__main__':
    main()
