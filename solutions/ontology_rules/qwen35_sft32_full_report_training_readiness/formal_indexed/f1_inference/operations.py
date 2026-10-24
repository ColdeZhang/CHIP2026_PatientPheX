"""Single full80 launch; small verified transport pieces and read-only recovery."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import sys
import time
import zipfile

sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

ap = argparse.ArgumentParser()
ap.add_argument('action', choices=['deploy', 'start', 'inspect', 'collect'])
ap.add_argument('--directory', type=Path, required=True)
a = ap.parse_args()
B = a.directory
plan = json.loads((B / 'plan.json').read_text())
root = '/home/zhangyuheng/projects/CHIP2026_PatientPheX/' + plan['remote']
digest = lambda b: hashlib.sha256(b).hexdigest()


def save(name, value):
    with (B / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


client = Client()
client.initialize()


def call(name, arguments, receipt):
    result = client.request('tools/call', {'name': name, 'arguments': {'mount_id': plan['mount_id'], **arguments}})
    save(receipt, result)
    if result.get('isError'):
        raise RuntimeError('MCP unknown/failure; inspect original state before any recovery')
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


def remote_python(code, parameters, receipt, timeout=30):
    code = 'PARAM=' + repr(parameters) + '\n' + code
    assert '\x00' not in code
    compile(code, 'remote-command.py', 'exec')
    result = call('run_command', {'command': 'python3 -c ' + shlex.quote(code),
                                 'timeout_seconds': timeout}, receipt)
    assert result['exit_code'] == 0
    return json.loads(result['stdout'])


if a.action == 'deploy':
    assert not (B / 'deployment-intent.json').exists()
    manifest = json.loads((B / 'deployment-manifest.json').read_text())
    payload = (B / 'deployment.tar.gz').read_bytes()
    assert digest(payload) == manifest['archive']['sha256']
    pieces = [payload[i:i + 65536] for i in range(0, len(payload), 65536)]
    refs = [{'name': f'part-{i:03d}', 'bytes': len(data), 'sha256': digest(data)} for i, data in enumerate(pieces)]
    save('deployment-intent.json', {'at_epoch': time.time(), 'archive': manifest['archive'],
                                  'remote': root, 'pieces': refs})
    for i, data in enumerate(pieces):
        call('write_file', {'path': plan['remote'] + '/' + refs[i]['name'],
            'content': base64.b64encode(data).decode(), 'encoding': 'base64',
            'create_parents': True}, f'upload-{i:03d}-tool.json')
    code = '''from pathlib import Path
import hashlib,json,os,tarfile
p=Path(PARAM['root']);m=PARAM['manifest'];parts=PARAM['pieces']
assert os.getuid()==p.stat().st_uid==1045
os.chmod(p,0o700)
assert not (p/'driver-started.json').exists()
for item in parts:
 b=(p/item['name']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
target=p/'deployment.tar.gz'
if not target.exists():
 with target.open('xb') as out:
  for item in parts:out.write((p/item['name']).read_bytes())
  out.flush();os.fsync(out.fileno())
b=target.read_bytes();assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256']
with tarfile.open(target) as t:
 members=t.getmembers();assert len(members)==len(m['files']) and set(x.name for x in members)==set(m['files'])
 for item in members:
  assert item.isfile() and item.name==Path(item.name).name
  data=t.extractfile(item).read();ref=m['files'][item.name]
  assert len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256']
  dest=p/item.name
  if dest.exists():assert dest.read_bytes()==data
  else:
   with dest.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
  os.chmod(dest,0o600)
  if dest.suffix=='.py':compile(dest.read_text(),item.name,'exec')
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'ACTUAL_ALL_SOURCE_INPUT_TRAINING_ACK_SHA_COMPILE_FSYNC','files':len(m['files']),'pieces':len(parts),'plan_sha256':hashlib.sha256((p/'plan.json').read_bytes()).hexdigest(),'uid':os.getuid(),'no_model_loaded':True}))
'''
    result = remote_python(code, {'root': root, 'manifest': manifest, 'pieces': refs}, 'deployment-tool.json')
    save('deployment-result.json', result)
    print(json.dumps(result))
elif a.action == 'start':
    deployment = json.loads((B / 'deployment-result.json').read_text())
    assert deployment['status'] == 'ACTUAL_ALL_SOURCE_INPUT_TRAINING_ACK_SHA_COMPILE_FSYNC'
    assert deployment['plan_sha256'] == digest((B / 'plan.json').read_bytes())
    assert not (B / 'start-intent.json').exists()
    save('start-intent.json', {'at_epoch': time.time(), 'cap_seconds': 1200,
        'plan_sha256': digest((B / 'plan.json').read_bytes()),
        'source_sha256': digest((B / 'infer_full80.py').read_bytes()), 'remote': root})
    result = call('start_process', {'command': shlex.quote(plan['python']) + ' -u launch.py',
        'cwd': plan['remote'], 'env': plan['env']}, 'start-tool.json')
    save('job.json', result)
    print(json.dumps(result))
elif a.action == 'inspect':
    stamp = str(time.time_ns())
    code = '''from pathlib import Path
import json
p=Path(PARAM['root']);out={}
for name in ['driver-started.json','child-started.json','wrapper-complete.json','summary.json','failure.json','gpu-release.json']:
 f=p/name
 if f.is_file():out[name]=json.loads(f.read_text())
out['completed_canonical_files']=len(list((p/'canonical-records').glob('*.json')))
print(json.dumps(out))
'''
    result = remote_python(code, {'root': root}, 'inspect-' + stamp + '-tool.json', 20)
    save('inspect-' + stamp + '.json', result)
    print(json.dumps(result))
else:
    assert not (B / 'collect-intent.json').exists(), 'Original collection may be partially downloaded; inspect and resume original bytes'
    save('collect-intent.json', {'at_epoch': time.time(), 'remote': root, 'no_new_model_or_quality': True})
    code = '''from pathlib import Path
import json,zipfile,hashlib,os
p=Path(PARAM['root'])
assert (p/'wrapper-complete.json').is_file() and (p/'gpu-release.json').is_file()
names=['actual-input-and-assets.json','base-loading.json','summary.json','complete.json','failure.json','fresh-gpu.json','driver-started.json','child-started.json','wrapper-complete.json','gpu-release.json','responses.jsonl','canonical.jsonl','stderr.log','stdout.log']
paths=[p/n for n in names if (p/n).is_file()]
for folder in ['generated-ids','raw-responses','canonical-records']:paths.extend(sorted((p/folder).glob('*.json')))
refs=[]
for f in paths:
 assert not f.is_symlink();b=f.read_bytes();assert len(b)<2000000
 refs.append({'name':f.relative_to(p).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
target=p/'collected-terminal.zip'
if not target.exists():
 with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
  for f,ref in zip(paths,refs):z.writestr(ref['name'],f.read_bytes())
 with target.open('rb') as f:os.fsync(f.fileno())
 fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
with zipfile.ZipFile(target) as z:
 assert len(z.namelist())==len(refs) and set(z.namelist())==set(r['name'] for r in refs)
 for ref in refs:
  b=z.read(ref['name']);assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
b=target.read_bytes();assert len(b)<10000000
print(json.dumps({'remote_archive':str(target),'zip_sha256':hashlib.sha256(b).hexdigest(),'zip_bytes':len(b),'files':refs}))
'''
    metadata = remote_python(code, {'root': root}, 'collect-manifest-tool.json')
    save('collect-remote-manifest.json', metadata)
    pieces = []
    for offset in range(0, metadata['zip_bytes'], 49152):
        size = min(49152, metadata['zip_bytes'] - offset)
        code = '''from pathlib import Path
import json,hashlib,base64
with Path(PARAM['path']).open('rb') as f:f.seek(PARAM['offset']);b=f.read(PARAM['size'])
assert len(b)==PARAM['size']
print(json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'b64':base64.b64encode(b).decode()}))
'''
        result = remote_python(code, {'path': metadata['remote_archive'], 'offset': offset, 'size': size}, f'collect-part-{offset:09d}-tool.json')
        data = base64.b64decode(result['b64'])
        assert len(data) == result['bytes'] == size and digest(data) == result['sha256']
        (B / f'collect-part-{offset:09d}.bin').write_bytes(data)
        pieces.append(data)
    payload = b''.join(pieces)
    assert len(payload) == metadata['zip_bytes'] and digest(payload) == metadata['zip_sha256']
    (B / 'terminal.zip').write_bytes(payload)
    dest = B / 'terminal'
    dest.mkdir(exist_ok=False)
    expected = {ref['name']: ref for ref in metadata['files']}
    assert len(expected) == len(metadata['files'])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
        for name, ref in expected.items():
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert len(path.parts) == 1 or (len(path.parts) == 2 and path.parts[0] in ['generated-ids', 'raw-responses', 'canonical-records'])
            data = archive.read(name)
            assert len(data) == ref['bytes'] and digest(data) == ref['sha256']
            (dest / path).parent.mkdir(exist_ok=True)
            (dest / path).write_bytes(data)
    manifest = {'status': 'ACTUAL_UNIQUE_FULL80_TERMINAL_COLLECTED', **metadata,
        'remote': root, 'plan_sha256': digest((B / 'plan.json').read_bytes()),
        'model_calls_repeated': False, 'quality_calls': 0}
    save('terminal-manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'files': len(expected),
                      'zip_bytes': metadata['zip_bytes'], 'zip_sha256': metadata['zip_sha256']}))
