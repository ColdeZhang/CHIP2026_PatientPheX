"""Only two fixed 1MiB public-asset Range requests, each hard-limited to 15s."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

R = Path(__file__).resolve().parent
assets = [
    {'filename': 'HP.tar.gz', 'bytes': 583483718, 'asset_id': 33396567, 'etag': '"0x8D9B91C61B87AEC"'},
    {'filename': 'pmc_model_new.bin', 'bytes': 1345465351, 'asset_id': 33395993, 'etag': '"0x8D9B91C5F51715C"'},
]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''


def worker(index):
    asset = assets[index]
    started = time.monotonic()
    report = {'asset': asset, 'range': [0, 1048575], 'complete': False}
    sidecar = R / 'output' / (asset['filename'] + '.json')
    payload = R / 'output' / (asset['filename'] + '.part')
    url = 'https://github.com/data-team-uhn/NeuralCR/releases/download/1.1/' + asset['filename']
    try:
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575', 'If-Range': asset['etag'],
            'User-Agent': 'PatientPheX-NeuralCR-fixed-1MiB-network-census/1.0'})
        with urllib.request.urlopen(request, timeout=12) as response:
            report.update(status=response.status, content_range=response.headers.get('Content-Range'),
                          etag=response.headers.get('ETag'), header_seconds=time.monotonic()-started)
            sidecar.write_text(json.dumps(report, indent=2)+'\n')
            assert report['status'] == 206
            assert report['content_range'] == f'bytes 0-1048575/{asset["bytes"]}'
            assert report['etag'] == asset['etag']
            count = 0
            h = hashlib.sha256()
            with payload.open('xb') as target:
                while count < 1048576:
                    data = response.read(min(65536, 1048576-count))
                    assert data
                    target.write(data)
                    target.flush()
                    h.update(data)
                    count += len(data)
            assert count == payload.stat().st_size == 1048576
            report.update(complete=True, bytes=count, sha256=h.hexdigest())
    except Exception as error:
        report['error_type'] = type(error).__name__
        # Do not print exception URL/redirect location; only fixed public metadata is retained.
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        sidecar.write_text(json.dumps(report, indent=2)+'\n')


if len(sys.argv) > 1:
    worker(int(sys.argv[1]))
else:
    (R / 'output').mkdir(exist_ok=False)
    results = []
    start = time.monotonic()
    for index, asset in enumerate(assets):
        before = time.monotonic()
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(index)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            wrapper = {'returncode': result.returncode, 'timeout_15s': False}
        except subprocess.TimeoutExpired:
            wrapper = {'returncode': None, 'timeout_15s': True}
        wrapper['actual_wrapper_seconds'] = time.monotonic()-before
        path = R / 'output' / (asset['filename']+'.json')
        part = R / 'output' / (asset['filename']+'.part')
        report = json.loads(path.read_text()) if path.exists() else {'asset': asset, 'complete': False}
        report.update(wrapper, actual_payload_bytes=part.stat().st_size if part.exists() else 0)
        results.append(report)
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'assets': results,
              'elapsed_seconds': time.monotonic()-start, 'no_model_or_GPU': True,
              'no_bulk_download': True, 'redirect_or_signed_URLs_not_recorded': True}
    (R / 'network-census.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
