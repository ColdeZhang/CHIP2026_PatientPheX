"""CPU-only bounded four-file private transfer; never imports a model or optimizer."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def verify(root, files):
    for name, item in files.items():
        path = root/name
        assert not path.is_symlink() and path.is_file()
        with path.open('rb') as handle: os.fsync(handle.fileno())
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    for directory in [root/'adapter', root, root.parent]: fsync_dir(directory)


def serve(plan):
    root = Path(plan['source_directory']); files = plan['files']
    assert set(files) == {'checkpoint.json', 'adapter/README.md', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    metadata = json.loads((root/'checkpoint.json').read_text())
    assert metadata['completed_epochs'] == 2 and metadata['global_step'] == 22
    assert metadata['fingerprint']['source_sha256'] == plan['trainer_sha256']
    for name in files:
        if name != 'checkpoint.json': assert metadata['files'][name] == files[name]['sha256']
    verify(root, files)
    payload = (json.dumps(dict(files=files, source_directory=str(root), source_fsync_and_reopen_SHA=True), sort_keys=True)+'\n').encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == '/manifest.json': raw = payload
            else:
                name = self.path[len('/files/'):] if self.path.startswith('/files/') else ''
                if name not in files or self.path != '/files/'+name: self.send_error(404); return
                path = root/name
                self.send_response(200); self.send_header('Content-Length', str(files[name]['bytes'])); self.end_headers()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(1048576), b''): self.wfile.write(block)
                return
            self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    server = ThreadingHTTPServer((plan['source_private_address'], 0), Handler)
    print(json.dumps(dict(ready=True, port=server.server_address[1], manifest_sha256=hashlib.sha256(payload).hexdigest(), manifest=json.loads(payload))), flush=True)
    server.timeout=.5
    deadline=time.monotonic()+plan['cap_seconds']
    try:
        while time.monotonic()<deadline:server.handle_request()
    finally:server.server_close()


def fetch(plan, url, manifest_sha, deadline):
    started = time.monotonic(); root = Path(plan['destination']); files = plan['files']
    assert time.time() < deadline and not root.exists()
    assert shutil.disk_usage(plan['persistent_project']).free >= plan['minimum_free_bytes']
    with urllib.request.urlopen(url+'/manifest.json', timeout=min(20, deadline-time.time())) as response: raw = response.read(65536)
    assert hashlib.sha256(raw).hexdigest() == manifest_sha
    manifest = json.loads(raw); assert manifest['files'] == files and manifest['source_fsync_and_reopen_SHA']
    staging = root.with_name(root.name+'.copy-incomplete'); assert not staging.exists()
    staging.mkdir(parents=True); (staging/'adapter').mkdir()
    for name, item in files.items():
        assert time.time() < deadline
        downloaded = 0; transfer_started = time.monotonic()
        with urllib.request.urlopen(url+'/files/'+name, timeout=min(30, deadline-time.time())) as response, (staging/name).open('xb') as handle:
            assert int(response.headers['Content-Length']) == item['bytes']
            while True:
                block = response.read(1048576)
                if not block: break
                handle.write(block); downloaded += len(block)
                assert downloaded <= item['bytes'] and time.time() < deadline
                delay = downloaded/plan['maximum_bytes_per_second']-(time.monotonic()-transfer_started)
                if delay > 0: time.sleep(delay)
            handle.flush(); os.fsync(handle.fileno())
        assert downloaded == item['bytes'] and sha(staging/name) == item['sha256']
    verify(staging, files); os.rename(staging, root)
    ancestor = root
    while True:
        fsync_dir(ancestor)
        if str(ancestor) == plan['persistent_project']: break
        ancestor = ancestor.parent
    verify(root, files)
    receipt = dict(status='PASS_adapter_only_second_persistent_copy', at_utc=datetime.now(timezone.utc).isoformat(),
        destination=str(root), source_directory=plan['source_directory'], files=files,
        total_bytes=sum(x['bytes'] for x in files.values()), source_manifest_sha256=manifest_sha,
        source_fsync_and_reopen_SHA=True, destination_all_files_fsync_and_reopen_SHA=True,
        destination_ancestor_directories_fsynced=True, original_source_retained=True,
        optimizer_state_copied=False, GPU_or_model_calls=0, seconds=time.monotonic()-started)
    with (root/'copy-ack.json').open('x') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    fsync_dir(root); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'fetch']); parser.add_argument('--plan', required=True)
    parser.add_argument('--url'); parser.add_argument('--manifest-sha'); parser.add_argument('--deadline', type=float)
    args = parser.parse_args(); plan = json.loads(Path(args.plan).read_text())
    assert plan['cap_seconds'] == 180 and plan['grace_seconds'] == 10
    if args.mode == 'serve': serve(plan)
    else: fetch(plan, args.url, args.manifest_sha, args.deadline)


if __name__ == '__main__': main()
