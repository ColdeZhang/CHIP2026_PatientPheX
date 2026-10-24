"""Bounded deployment, single start and read-only collection for the probe."""
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

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['deploy', 'start', 'inspect', 'collect'])
parser.add_argument('--directory', type=Path, required=True)
args = parser.parse_args()
B = args.directory
plan = json.loads((B / 'plan.json').read_text())
root = '/home/zhangyuheng/projects/CHIP2026_PatientPheX/' + plan['remote']


def save(name, value):
    with (B / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def unwrap(value):
    return value.get('structuredContent') or json.loads(value['content'][0]['text'])


client = Client()
client.initialize()


def call(name, arguments, receipt):
    result = client.request('tools/call', {'name': name, 'arguments': {'mount_id': plan['mount_id'], **arguments}})
    save(receipt, result)
    if result.get('isError'):
        raise RuntimeError('MCP result unknown/failed; inspect original state, never repeat start blindly')
    return unwrap(result)


if args.action == 'deploy':
    assert not (B / 'deployment-intent.json').exists()
    manifest = json.loads((B / 'deployment-manifest.json').read_text())
    payload = (B / 'deployment.tar.gz').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['archive']['sha256']
    save('deployment-intent.json', {'at_epoch': time.time(), 'archive': manifest['archive'], 'remote': root})
    call('write_file', {'path': plan['remote'] + '/deployment.tar.gz',
         'content': base64.b64encode(payload).decode(), 'encoding': 'base64', 'create_parents': True}, 'upload-tool.json')
    code = '''from pathlib import Path
import hashlib,json,os,tarfile
p=Path(PARAM['root']);m=PARAM['manifest']
assert os.getuid()==p.stat().st_uid==1045
os.chmod(p,0o700)
assert not (p/'driver-started.json').exists()
b=(p/'deployment.tar.gz').read_bytes();assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256']
with tarfile.open(p/'deployment.tar.gz') as t:
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
print(json.dumps({'status':'ACTUAL_ALL_SOURCE_INPUT_SHA_COMPILE_FSYNC','files':len(m['files']),'plan_sha256':hashlib.sha256((p/'plan.json').read_bytes()).hexdigest(),'uid':os.getuid(),'no_model_loaded':True}))
'''
    code = 'PARAM=' + repr({'root': root, 'manifest': manifest}) + '\n' + code
    assert '\x00' not in code
    compile(code, 'deploy-remote.py', 'exec')
    result = call('run_command', {'command': 'python3 -c ' + shlex.quote(code), 'timeout_seconds': 30}, 'deployment-tool.json')
    assert result['exit_code'] == 0
    save('deployment-result.json', json.loads(result['stdout']))
    print(result['stdout'])
elif args.action == 'start':
    assert json.loads((B / 'deployment-result.json').read_text())['status'] == 'ACTUAL_ALL_SOURCE_INPUT_SHA_COMPILE_FSYNC'
    assert not (B / 'start-intent.json').exists()
    save('start-intent.json', {'at_epoch': time.time(), 'cap_seconds': 240,
        'plan_sha256': hashlib.sha256((B / 'plan.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256((B / 'runner.py').read_bytes()).hexdigest(), 'remote': root})
    result = call('start_process', {'command': shlex.quote(plan['python']) + ' -u launch.py',
        'cwd': plan['remote'], 'env': plan['env']}, 'start-tool.json')
    save('job.json', result)
    print(json.dumps(result))
elif args.action == 'inspect':
    stamp = str(time.time_ns())
    code = '''from pathlib import Path
import json
p=Path(ROOT)
out={}
for name in ['driver-started.json','child-started.json','wrapper-complete.json','summary.json','failure.json','gpu-release.json']:
 f=p/name
 if f.is_file():out[name]=json.loads(f.read_text())
print(json.dumps(out))
'''
    code = 'ROOT=' + repr(root) + '\n' + code
    result = call('run_command', {'command': 'python3 -c ' + shlex.quote(code), 'timeout_seconds': 20}, 'inspect-' + stamp + '-tool.json')
    assert result['exit_code'] == 0
    out = json.loads(result['stdout'])
    save('inspect-' + stamp + '.json', out)
    print(json.dumps(out))
else:
    assert not (B / 'terminal-manifest.json').exists()
    code = '''from pathlib import Path
import json,zipfile,io,base64,hashlib
p=Path(ROOT)
assert (p/'wrapper-complete.json').is_file() and (p/'gpu-release.json').is_file()
names=['actual-input-and-assets.json','base-loading.json','generated-ids.json','raw-response.json','format-result.json','summary.json','failure.json','fresh-gpu.json','driver-started.json','child-started.json','wrapper-complete.json','gpu-release.json','stderr.log','stdout.log']
buf=io.BytesIO();refs=[]
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  f=p/name
  if f.is_file():
   b=f.read_bytes();assert len(b)<1000000
   z.writestr(name,b);refs.append({'name':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
b=buf.getvalue();assert len(b)<200000
print(json.dumps({'zip_b64':base64.b64encode(b).decode(),'zip_sha256':hashlib.sha256(b).hexdigest(),'files':refs}))
'''
    code = 'ROOT=' + repr(root) + '\n' + code
    result = call('run_command', {'command': 'python3 -c ' + shlex.quote(code), 'timeout_seconds': 30}, 'collect-tool.json')
    assert result['exit_code'] == 0
    data = json.loads(result['stdout'])
    payload = base64.b64decode(data['zip_b64'])
    assert hashlib.sha256(payload).hexdigest() == data['zip_sha256']
    (B / 'terminal.zip').write_bytes(payload)
    dest = B / 'terminal'
    dest.mkdir(exist_ok=False)
    expected = {ref['name']: ref for ref in data['files']}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected)
        for name, ref in expected.items():
            assert name == Path(name).name
            b = archive.read(name)
            assert len(b) == ref['bytes'] and hashlib.sha256(b).hexdigest() == ref['sha256']
            (dest / name).write_bytes(b)
    manifest = {'status': 'ACTUAL_UNIQUE_PROBE_TERMINAL_COLLECTED', 'files': data['files'],
        'remote': root, 'zip_sha256': data['zip_sha256'], 'zip_bytes': len(payload),
        'plan_sha256': hashlib.sha256((B / 'plan.json').read_bytes()).hexdigest(),
        'model_calls_repeated': False, 'quality_calls': 0}
    save('terminal-manifest.json', manifest)
    print(json.dumps(manifest))
