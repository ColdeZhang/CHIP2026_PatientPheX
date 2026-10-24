"""Bounded four-worker Range resume from immutable, SHA-bound old prefixes.

Preparation only until an owner freezes resume-plan.json after the old job exits.
No model imports, extraction, original-prefix mutation, or quality evaluation.
"""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import os
import threading
import time
import urllib.request

R = Path(__file__).resolve().parent
PLAN = R / 'resume-plan.json'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
plan = json.loads(PLAN.read_text())
assert plan['status'] == 'authorized_frozen_resume_after_attempt1_exit'
assert plan['workers'] == 4 and plan['chunk_bytes'] == 8 * 1024 * 1024
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['source_sha256']
assert plan['prior_job_terminal'] and plan['budget_seconds'] > 0
started = time.monotonic()
deadline = started + plan['budget_seconds']
lock = threading.Lock()
stop = threading.Event()
outdir = R / 'downloads'
outdir.mkdir(exist_ok=False)
chunks = R / 'chunks'
chunks.mkdir(exist_ok=False)
state = {'status': 'running', 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
         'budget_seconds': plan['budget_seconds'], 'source_sha256': plan['source_sha256'],
         'prefixes': [], 'chunks_complete': [], 'no_model_or_NN_loaded': True}


def save():
    with lock:
        tmp = R / 'resume-state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(R / 'resume-state.json')


def check_time():
    if stop.is_set() or time.monotonic() >= deadline:
        raise TimeoutError('fixed resume budget exhausted or peer failed')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            check_time()
            h.update(data)
    return h.hexdigest()


def fetch(task):
    asset, begin, end = task
    part = chunks / (asset['filename'] + '.' + str(begin) + '-' + str(end) + '.part')
    offset = begin
    failures = 0
    t = time.monotonic()
    with part.open('xb') as f:
        while offset <= end:
            check_time()
            request = urllib.request.Request(asset['url'], headers={
                'User-Agent': 'PatientPheX-NeuralCR-resume/1.0',
                'Range': f'bytes={offset}-{end}', 'If-Range': asset['etag']})
            try:
                with urllib.request.urlopen(request, timeout=min(20, max(1, deadline - time.monotonic()))) as response:
                    assert response.status == 206
                    assert response.headers.get('Content-Range') == f'bytes {offset}-{end}/{asset["bytes"]}'
                    assert response.headers.get('ETag') == asset['etag']
                    while offset <= end:
                        check_time()
                        data = response.read(min(1024 * 1024, end - offset + 1))
                        assert data, 'short range response'
                        f.write(data)
                        offset += len(data)
                f.flush()
            except Exception:
                f.flush()
                failures += 1
                if failures >= 3 or time.monotonic() >= deadline or stop.is_set():
                    raise
        assert f.tell() == end - begin + 1
    done = part.with_suffix('.done')
    part.rename(done)
    record = {'filename': asset['filename'], 'start': begin, 'end_inclusive': end,
              'bytes': done.stat().st_size, 'sha256': sha_file(done),
              'seconds': time.monotonic() - t, 'network_failures': failures}
    with lock:
        state['chunks_complete'].append(record)
    save()
    return record


try:
    save()
    tasks = []
    for asset in plan['assets']:
        prefix = Path(asset['prefix']['path'])
        assert prefix.stat().st_size == asset['prefix']['bytes'] <= asset['bytes']
        assert sha_file(prefix) == asset['prefix']['sha256']
        assert asset['asset_id'] and asset['etag'] and asset['url'].startswith('https://github.com/data-team-uhn/NeuralCR/releases/download/1.1/')
        state['prefixes'].append(asset['prefix'])
        for start in range(asset['prefix']['bytes'], asset['bytes'], plan['chunk_bytes']):
            tasks.append((asset, start, min(asset['bytes'] - 1, start + plan['chunk_bytes'] - 1)))
    save()
    # A single four-worker executor spans both assets: never four per asset.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, task) for task in tasks]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    outputs = []
    for asset in plan['assets']:
        destination = outdir / asset['filename']
        h = hashlib.sha256()
        count = 0
        parts = [Path(asset['prefix']['path'])] + [
            chunks / (asset['filename'] + '.' + str(start) + '-' + str(end) + '.done')
            for item, start, end in tasks if item['filename'] == asset['filename']]
        with destination.open('xb') as target:
            for part in parts:
                with part.open('rb') as source:
                    for data in iter(lambda: source.read(1048576), b''):
                        check_time()
                        target.write(data)
                        h.update(data)
                        count += len(data)
        assert count == destination.stat().st_size == asset['bytes']
        outputs.append({'filename': asset['filename'], 'bytes': count, 'sha256': h.hexdigest(),
                        'etag': asset['etag'], 'official_asset_id': asset['asset_id'],
                        'source_url': asset['url'], 'reused_prefix': asset['prefix']})
    state.update(status='complete_official_assets_not_loaded', outputs=outputs,
                 elapsed_seconds=time.monotonic() - started,
                 finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert state['elapsed_seconds'] <= plan['budget_seconds']
    save()
    (outdir / 'complete.json').write_text(json.dumps(state, indent=2) + '\n')
except BaseException as error:
    stop.set()
    state.update(status='failed_preserved', error=repr(error), elapsed_seconds=time.monotonic() - started)
    save()
    raise
