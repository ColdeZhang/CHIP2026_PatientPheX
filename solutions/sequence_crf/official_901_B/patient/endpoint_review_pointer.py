"""Append a verified review pointer without changing the frozen endpoint ZIP."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import sys
import time
from endpoint_supplement import MOUNT, TARGET, encoded, publish, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--client-dir', type=Path, default=Path('/workspace/patientphex-tools'))
    a = ap.parse_args()
    out = a.run_dir.resolve() / 'endpoint-supplement-v1'
    package = json.loads((out / 'package.json').read_text())
    actual = ref(a.review)
    assert actual['bytes'] == 8983 and actual['sha256'] == 'cd7a6040516af20916746f99af9cdec78a389b7d73dfe1045ee687ef0937310a'
    pointer = dict(schema='patient-endpoint-postfreeze-review-pointer-v1',
        review=actual, frozen_package=ref(out / 'package.json'), frozen_archive=package['archive'],
        pointer_only=True, review_contents_not_duplicated=True, archive_not_repacked=True,
        no_new_score=True, no_NN=True, source=ref(__file__))
    path = out / 'final-review-pointer.json'
    publish(path, encoded(pointer))
    intent = out / 'review-pointer-deployment-intent.json'
    assert not intent.exists(), 'Existing intent requires inventory; do not blindly reissue'
    publish(intent, encoded(dict(at_unix=time.time(), mount=MOUNT, target=TARGET + '/final-review-pointer.json', local=ref(path))))
    code = """import base64,hashlib,json,os;from pathlib import Path
p=Path(DEST);data=base64.b64decode(DATA);assert p.parent.is_dir()
if p.exists():assert p.read_bytes()==data
else:
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
with p.open('rb') as f:os.fsync(f.fileno())
q=p.parent.resolve()
while True:
 fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if q.parent==q:break
 q=q.parent
assert p.read_bytes()==data
print(json.dumps(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fsynced_and_reopened=True)))
""".replace('DEST', repr(TARGET + '/final-review-pointer.json')).replace('DATA', repr(base64.b64encode(path.read_bytes()).decode()))
    sys.path.insert(0, str(a.client_dir.resolve()))
    from mcp_client import Client
    client = Client()
    client.initialize()
    try:
        result = client.request('tools/call', dict(name='run_command', arguments=dict(mount_id=MOUNT, cwd='.',
            command='python3 -c ' + shlex.quote(code), timeout_seconds=30)))
    except Exception as e:
        publish(out / 'review-pointer-deployment-error.json', encoded(dict(type=type(e).__name__, unknown=True)))
        raise RuntimeError('UNKNOWN; inventory before retry') from None
    publish(out / 'review-pointer-deployment-response.json', encoded(result))
    assert not result.get('isError')
    value = result.get('structuredContent') or json.loads(result['content'][0]['text'])
    assert value['exit_code'] == 0
    remote = json.loads(value['stdout'])
    assert remote['sha256'] == ref(path)['sha256'] and remote['bytes'] == path.stat().st_size
    publish(out / 'review-pointer-persistent.json', encoded(dict(local=ref(path), remote=remote, mount=MOUNT)))
    print(json.dumps(dict(receipt=ref(out / 'review-pointer-persistent.json'), remote=remote)))


if __name__ == '__main__':
    main()
