#!/usr/bin/env python3
"""Download a pinned public HF file from a temporary official CDN URL.
The URL is read from a private runtime file and never emitted to output.
Only writes within this script's directory; no GPU is used.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
import urllib.request

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / 'cdn-runtime.private.json'
STATUS = HERE / 'download-recovery-status.json'
START = time.monotonic()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def progress(stage, **kwargs):
    value = {'stage': stage, 'elapsed_seconds': round(time.monotonic() - START, 3), **kwargs}
    write_json(STATUS, value)
    print(json.dumps(value), flush=True)


def main():
    cfg = json.loads(PRIVATE.read_text())
    os.chmod(PRIVATE, 0o600)
    deadline = datetime.fromisoformat(cfg['absolute_deadline_utc']).timestamp()
    if time.time() >= deadline: raise TimeoutError('original deadline exhausted')
    expected_bytes = cfg['weight_bytes']
    expected_sha = cfg['weight_sha256']
    final = HERE / 'model' / 'pytorch_model.bin'
    if final.exists():
        if final.stat().st_size != expected_bytes or digest(final) != expected_sha:
            raise ValueError('existing_weight_does_not_match_pinned_identity')
        progress('existing_weight_verified', bytes=expected_bytes, sha256=expected_sha)
    else:
        parts = HERE / '.coref-download-parts'
        parts.mkdir(exist_ok=True, mode=0o700)
        chunk = cfg.get('chunk_bytes', 16 * 1024 * 1024)
        ranges = [(i, start, min(start + chunk, expected_bytes) - 1)
                  for i, start in enumerate(range(0, expected_bytes, chunk))]
        def download(item):
            index, first, last = item
            dest = parts / f'{index:05d}.part'
            size = last - first + 1
            if dest.exists() and dest.stat().st_size == size:
                return {'index': index, 'bytes': size, 'resumed': True}
            for attempt in range(4):
                tmp = dest.with_suffix('.tmp')
                try:
                    req = urllib.request.Request(cfg['cdn_url'], headers={'Range': f'bytes={first}-{last}'})
                    if time.time() >= deadline: raise TimeoutError('original deadline exhausted')
                    with urllib.request.urlopen(req, timeout=min(30,max(1,deadline-time.time()))) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{expected_bytes}':
                            raise ValueError('unexpected_range_response')
                        received = 0
                        with tmp.open('wb') as stream:
                            while received < size:
                                block = response.read(min(1024 * 1024, size - received))
                                if not block:
                                    break
                                stream.write(block)
                                received += len(block)
                    if received != size:
                        raise ValueError('incomplete_range')
                    tmp.replace(dest)
                    return {'index': index, 'bytes': size, 'resumed': False}
                except Exception as exc:
                    if tmp.exists():
                        tmp.unlink()
                    if attempt == 3:
                        raise RuntimeError(f'range_{index}_failed_{type(exc).__name__}') from None
                    time.sleep(min(2 ** attempt, 4))
        complete, transferred = 0, 0
        progress('downloading', completed_parts=0, total_parts=len(ranges), expected_bytes=expected_bytes, workers=4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download, item) for item in ranges]
            for f in as_completed(futures):
                result = f.result()
                complete += 1
                transferred += result['bytes']
                if complete % 8 == 0 or complete == len(ranges):
                    progress('downloading', completed_parts=complete, total_parts=len(ranges), completed_bytes=transferred, expected_bytes=expected_bytes)
        staging = HERE / 'model' / 'pytorch_model.bin.tmp'
        progress('assembling', expected_bytes=expected_bytes)
        with staging.open('wb') as output:
            for index, first, last in ranges:
                with (parts / f'{index:05d}.part').open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        output.write(block)
        actual_sha = digest(staging)
        if staging.stat().st_size != expected_bytes or actual_sha != expected_sha:
            raise ValueError('full_weight_checksum_mismatch')
        staging.replace(final)
        progress('weight_verified', bytes=expected_bytes, sha256=actual_sha)
        for part in parts.iterdir():
            part.unlink()
        parts.rmdir()
    PRIVATE.unlink()
    manifest={k:cfg[k] for k in ['repo_id','revision','source_resolve_url','weight_bytes','weight_sha256']}
    manifest.update({'download_complete':True,'verified_full_weight_sha256':digest(final),'download_seconds':time.monotonic()-START,'temporary_url_config_removed':not PRIVATE.exists(),'download_script_sha256':digest(Path(__file__)),'GPU_used':False})
    write_json(HERE/'weight-source-manifest.json',manifest)
    plan=json.loads((HERE/'download-plan.json').read_text())
    verified={}
    for a in plan['files']:
        f=HERE/'model'/a['name']; actual=digest(f); assert f.stat().st_size==a['bytes']
        if a.get('sha256'): assert actual==a['sha256']
        else: assert hashlib.sha1(b'blob '+str(a['bytes']).encode()+b'\0'+f.read_bytes()).hexdigest()==a['git_blob_sha1']
        verified[a['name']]={'bytes':a['bytes'],'sha256':actual,'source_url':a['url']}
    write_json(HERE/'download-complete.json',{'status':'complete_verified_all_files','files':verified,'recovery':'official CDN ranges after HF connection timeout','original_absolute_deadline_utc':cfg['absolute_deadline_utc']})
    progress('complete',bytes=expected_bytes,sha256=expected_sha,temporary_url_config_removed=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Deliberately omit exception messages/tracebacks; urllib errors may contain URLs.
        progress('failed', error_type=type(exc).__name__)
        sys.exit(1)
