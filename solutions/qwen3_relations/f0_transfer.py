"""Restricted, authenticated, short-lived transfer of the frozen F0 assets."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def pack(plan):
    output = Path(plan['export_dir'])
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for prefix, directory in plan['directories'].items():
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[prefix + '/' + str(path.relative_to(directory))] = path
    for name, info in plan['files'].items():
        path = Path(info['path'])
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
        files[name] = path
    # Transformers legitimately ships an optional integration named
    # integrations/bitsandbytes.py. Exclude the unused dependency package,
    # not that unchanged source file inside the complete runtime.
    assert all(not name.startswith(('runtime/bitsandbytes', 'training-runtime/bitsandbytes'))
               and not name.endswith('training_state.pt') for name in files)
    inventory = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in files.items()}
    save(output / 'inventory.json', inventory)
    archive = output / 'f0-assets.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        tar.add(output / 'inventory.json', arcname='inventory.json', recursive=False)
    with archive.open('rb') as stream:
        os.fsync(stream.fileno())
    result = {'status': 'FROZEN_SINGLE_ASSET_BUNDLE', 'archive': str(archive),
              'bytes': archive.stat().st_size, 'sha256': sha(archive),
              'members': len(files) + 1, 'payload_bytes': sum(x['bytes'] for x in inventory.values()),
              'exclusions': ['public8Bbase already cached', 'unused bitsandbytes', 'pycache', 'optimizer state']}
    save(output / 'pack-result.json', result)
    print(json.dumps(result), flush=True)


def serve(plan):
    archive = Path(plan['archive'])
    assert archive.stat().st_size == plan['bytes'] and sha(archive) == plan['sha256']
    secret = os.environ['QWEN_TRANSFER_AUTH']
    assert len(secret) >= 32
    ttl = plan.get('ttl_seconds', 300)
    assert 0 < ttl <= 300
    deadline = time.monotonic() + ttl

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            authorized = hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + secret)
            if self.path != '/bundle' or not authorized or time.monotonic() >= deadline:
                self.send_error(403)
                return
            requested = self.headers.get('Range')
            if requested and requested != 'bytes=0-4095':
                self.send_error(416)
                return
            length = min(4096, archive.stat().st_size) if requested else archive.stat().st_size
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', 'application/gzip')
            self.send_header('Content-Length', str(length))
            if requested:
                self.send_header('Content-Range', f'bytes 0-{length-1}/{archive.stat().st_size}')
            self.end_headers()
            self.connection.settimeout(30)
            sent, tick = 0, time.monotonic()
            with archive.open('rb') as stream:
                while sent < length:
                    block = stream.read(min(1024**2, length - sent))
                    if time.monotonic() >= deadline:
                        break
                    self.wfile.write(block)
                    sent += len(block)
                    time.sleep(max(0, sent / (20 * 1024**2) - (time.monotonic() - tick)))

    server = HTTPServer((plan.get('bind_host', '127.0.0.1'), int(os.environ['PORT'])), Handler)
    server.timeout = .5
    print(json.dumps({'status': 'AUTHENTICATED_SINGLE_FILE_SERVER', 'TTL_seconds': ttl,
                      'only_path': '/bundle', 'secret_logged': False}), flush=True)
    while time.monotonic() < deadline:
        server.handle_request()
    server.server_close()


def receive(plan):
    started = time.monotonic()
    out = Path(plan['receive_dir'])
    out.mkdir(parents=True, exist_ok=False)
    url, token = os.environ['QWEN_TRANSFER_URL'], os.environ['QWEN_TRANSFER_AUTH']
    parts = urlsplit(url)
    endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip('/') + '/bundle', parts.query, ''))
    request = urllib.request.Request(endpoint, headers={'Authorization': 'Bearer ' + token})
    archive = out / 'assets.tar.gz'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=90) as response, archive.open('xb') as stream:
        assert int(response.headers['Content-Length']) == plan['bytes']
        count = 0
        for block in iter(lambda: response.read(1024**2), b''):
            count += len(block)
            assert count <= plan['bytes']
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    assert archive.stat().st_size == plan['bytes'] and sha(archive) == plan['sha256']
    target = out / 'assets'
    target.mkdir()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == plan['members']
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        inventory = json.load(tar.extractfile('inventory.json'))
        assert {m.name for m in members} == set(inventory) | {'inventory.json'}
        for member in members:
            dest = target / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('xb') as output:
                for block in iter(lambda: source.read(1024**2), b''):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
        for name, expected in inventory.items():
            file = target / name
            assert file.stat().st_size == expected['bytes'] and sha(file) == expected['sha256'], name
    for directory in sorted({target, out, *(p.parent for p in target.rglob('*'))}, key=lambda x: -len(x.parts)):
        fd = os.open(directory, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    result = {'status': 'PASS_AUTHENTICATED_TRANSFER_ALL_MEMBER_SHA', 'bytes': plan['bytes'],
              'sha256': plan['sha256'], 'members': plan['members'], 'seconds': time.monotonic() - started,
              'assets': str(target), 'URL_or_token_recorded': False, 'GPU': False}
    save(out / 'transfer-result.json', result)
    fd = os.open(out, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pack', 'serve', 'receive'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    globals()[args.mode](json.loads(Path(args.plan).read_text()))
