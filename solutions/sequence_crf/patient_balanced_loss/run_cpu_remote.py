"""Bounded transport for one model-free CPU contract; stages never auto-retry."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
import zipfile

MOUNT = '476f81cf-09e7-445b-a7f0-d0e12b12e50f'
RELATIVE = 'sequence-round2/patient-balanced-alpha05/cpu-contract-v1'
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REMOTE = PROJECT + '/' + RELATIVE
PYTHON = '/dataset/miniforge3/envs/vlm-det/bin/python'
NATIVE = PROJECT + '/case-retrieval/qwen35-preparation/runtime/transformers/loss/loss_utils.py'


def ref(path):
    p = Path(path).resolve(); data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    parent = path.parent.resolve()
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['pack', 'deploy', 'start', 'collect'])
    ap.add_argument('--directory', type=Path, required=True)
    args = ap.parse_args(); out = args.directory.resolve(); here = Path(__file__).resolve().parent
    os.umask(0o077)
    if args.stage == 'pack':
        plan = json.loads((here / 'cpu-contract-plan.json').read_text())
        sources = json.loads((here / 'cpu-local-inputs.json').read_text())
        files = {'cpu-contract-plan.json': here / 'cpu-contract-plan.json', 'cpu_contract.py': here / 'cpu_contract.py'}
        for key, source in sources.items():
            assert ref(source['path']) == source
            if key != 'native_loss_utils': files[Path(plan['references'][key]['path']).name] = Path(source['path'])
        members = []
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, path in sorted(files.items()):
                item = ref(path); members.append(dict(name=name, **item)); z.writestr(name, path.read_bytes())
        raw = archive.getvalue(); assert len(raw) < 256 * 1024
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'deployment.zip').open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        save(out / 'package.json', dict(archive=ref(out / 'deployment.zip'), members=members,
            source=ref(__file__), plan=ref(here / 'cpu-contract-plan.json'), remote=REMOTE, mount=MOUNT))
        print(json.dumps(ref(out / 'package.json'))); return
    package = json.loads((out / 'package.json').read_text())
    assert package['source'] == ref(__file__) and package['archive'] == ref(out / 'deployment.zip')
    sys.path.insert(0, '/workspace/patientphex-tools')
    from mcp_client import Client
    client = Client(); client.initialize()

    def call(name, arguments, tag):
        dest = out / 'transport' / (tag + '-' + str(time.time_ns()))
        safe = dict(arguments)
        if 'content' in safe: safe['content'] = '[exact frozen package content omitted]'
        save(dest.with_suffix('.intent.json'), dict(tool=name, arguments=safe))
        try: response = client.request('tools/call', dict(name=name, arguments=arguments))
        except Exception as error:
            save(dest.with_suffix('.error.json'), dict(error_type=type(error).__name__, UNKNOWN=True, automatic_retry=False))
            raise RuntimeError('UNKNOWN; inventory actual state before any continuation') from None
        encoded = json.dumps(response)
        for key in ('MCP_TOKEN', 'MCP_URL'):
            if os.environ.get(key): encoded = encoded.replace(os.environ[key], '[REDACTED]')
        response = json.loads(encoded)
        if response.get('isError'):
            save(dest.with_suffix('.error.json'), response)
            raise RuntimeError('Tool error saved; stop without automatic retry')
        result = response.get('structuredContent')
        if result is None: result = json.loads(next(x['text'] for x in response['content'] if x['type'] == 'text'))
        save(dest.with_suffix('.response.json'), result)
        return result

    def command(code, tag, timeout=30):
        result = call('run_command', dict(mount_id=MOUNT, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=timeout), tag)
        assert result['exit_code'] == 0, tag
        return json.loads(result['stdout'])

    if args.stage == 'deploy':
        assert not (out / 'deployment.json').exists()
        inventory = command('from pathlib import Path;import json,hashlib; p=Path(' + repr(REMOTE) + '); n=Path(' + repr(NATIVE) + '); print(json.dumps(dict(exists=p.exists(), native=dict(bytes=n.stat().st_size,sha256=hashlib.sha256(n.read_bytes()).hexdigest()),python_exists=Path(' + repr(PYTHON) + ').is_file())))', 'inventory')
        assert not inventory['exists'] and inventory['python_exists']
        assert inventory['native'] == dict(bytes=8540, sha256='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe')
        raw = (out / 'deployment.zip').read_bytes()
        receipt = call('write_file', dict(mount_id=MOUNT, path=RELATIVE + '/deployment.zip', content=base64.b64encode(raw).decode(), encoding='base64', create_parents=True), 'upload')
        assert receipt['size'] == len(raw) and receipt['sha256'] == package['archive']['sha256']
        code = '''from pathlib import Path
import json,hashlib,zipfile,os
cfg=CFG; root=Path(ROOT)
z=root/'deployment.zip'; assert hashlib.sha256(z.read_bytes()).hexdigest()==cfg['archive']['sha256']
with zipfile.ZipFile(z) as archive:
 assert set(archive.namelist())=={x['name'] for x in cfg['members']}
 for item in cfg['members']:
  assert Path(item['name']).name==item['name']; data=archive.read(item['name'])
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  p=root/item['name']
  with p.open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
  if p.suffix=='.py': compile(data,str(p),'exec')
p=root
while True:
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if p.parent==p: break
 p=p.parent
print(json.dumps(dict(deployed_members=len(cfg['members']),root=str(root),all_SHA_fsync_reopen=True,no_torch_or_model_import=True)))
'''.replace('CFG', repr(package)).replace('ROOT', repr(REMOTE))
        deployed = command(code, 'verify')
        save(out / 'deployment.json', dict(inventory=inventory, actual=deployed, package=ref(out / 'package.json'), no_execution=True))
    elif args.stage == 'start':
        assert (out / 'deployment.json').exists() and not (out / 'start-intent.json').exists()
        source = REMOTE + '/cpu_contract.py'; plan = REMOTE + '/cpu-contract-plan.json'
        cmd = shlex.join([PYTHON, source, '--plan', plan])
        save(out / 'start-intent.json', dict(command=cmd, mount=MOUNT, no_automatic_retry=True, cap_seconds=60, term_grace_seconds=5))
        result = call('start_process', dict(mount_id=MOUNT, cwd='.', command=cmd, env=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')), 'start')
        save(out / 'started.json', result)
    else:
        code = '''from pathlib import Path
import json,hashlib,base64
root=Path(ROOT)/'actual'; assert (root/'terminal.json').exists()
files=[]
for p in sorted(root.iterdir()):
 assert p.is_file();raw=p.read_bytes(); assert len(raw)<512*1024
 files.append(dict(name=p.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
print(json.dumps(dict(files=files)))
'''.replace('ROOT', repr(REMOTE))
        actual = command(code, 'collect', timeout=45)
        collected = out / 'collected'; collected.mkdir(exist_ok=False)
        refs = {}
        for item in actual['files']:
            assert Path(item['name']).name == item['name']
            raw = base64.b64decode(item.pop('data')); assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
            dest = collected / item['name']
            with dest.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
            refs[item['name']] = ref(dest)
        save(out / 'collection.json', dict(files=refs, all_actual_SHA_verified=True, original_remote=REMOTE + '/actual', source=ref(__file__)))
    print(json.dumps(dict(stage=args.stage, directory=str(out), complete=True)))


if __name__ == '__main__': main()
