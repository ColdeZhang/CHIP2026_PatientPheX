"""Explicit recoverable stages; imports alone perform no remote/model/quality work."""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
import zipfile

ROUTE = Path(__file__).resolve().parents[1]
DOWN = ROUTE / 'downstream'
spec = importlib.util.spec_from_file_location('dpo_control', ROUTE / 'control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
ref, digest, now = control.localref, control.digest, control.now
CHUNK = 512 * 1024


def save(path, value):
    control.save(path, value)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_exact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'Existing artifact differs: ' + str(path)
        return
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def record(path, value):
    write_exact(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def content_matches(path, expected):
    got = ref(path)
    assert all(got[k] == expected[k] for k in ['bytes', 'sha256']), str(path)


def read_chunk(remote, path, offset, size):
    assert 0 < size <= CHUNK
    r = remote.call('read_file', path=control.relative(str(path)), offset=offset, limit=size)
    data = base64.b64decode(r['content']) if r.get('encoding') == 'base64' else r['content'].encode()
    assert len(data) <= size
    return data


def read_small(remote, path):
    data = read_chunk(remote, path, 0, CHUNK)
    assert len(data) < CHUNK, 'Metadata unexpectedly reaches transfer limit'
    return data


def core_plan(directory):
    path = directory / 'execution-plan.json'
    p = json.loads(path.read_text())
    assert p['execution_authorized'] is True and p['id'] == 'E-20261024-PATIENT-SWAP-DPO-001'
    assert p['mount_id'] == control.MOUNT
    return p


def state(directory):
    out = directory / 'downstream-v1'
    out.mkdir(exist_ok=True)
    return out


def inspect(directory, remote, inference=False):
    p = core_plan(directory); root = p['run_root'] + ('/infer80' if inference else '')
    names = (['child-start.json', 'driver-result.json', 'native/progress.json', 'native/complete.json',
              'native/failure.json', 'terminal-receipt.json'] if inference else
             ['driver-start.json', 'driver-progress.json', 'driver-result.json', 'closed-before-formal.json',
              'reference/complete.json', 'reference/progress.json', 'probe/complete.json',
              'probe/progress.json', 'formal/complete.json', 'formal/progress.json',
              'formal/epoch-001-persistent-ack.json'])
    code = 'import json\nfrom pathlib import Path\nr=Path(' + repr(root) + ')\n'
    code += 'print(json.dumps({n:json.loads((r/n).read_text()) for n in ' + repr(names) + ' if (r/n).is_file()}))'
    result = remote.call('run_command', command=shlex.join([p['python'], '-c', code]), cwd='',
                         env={'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1'}, timeout_seconds=20)
    save(state(directory) / ('inspect-' + ('infer-' if inference else 'core-') + str(time.time_ns()) + '.json'), result)
    assert result['exit_code'] == 0
    value = json.loads(result['stdout']); print(json.dumps(value)); return value


def collect_core(directory, remote):
    p = core_plan(directory); out = state(directory) / 'core'
    names = ['driver-start.json', 'driver-result.json', 'formal/start.json', 'formal/complete.json',
             'formal/verified-assets.json', 'formal/epoch-001-persistent-ack.json', 'formal/epoch-001/checkpoint.json']
    for name in names:
        write_exact(out / name, read_small(remote, p['run_root'] + '/' + name))
    driver = json.loads((out / 'driver-result.json').read_text())
    assert driver['status'] == 'COMPLETE_ALL_THREE_PHASES'
    assert [s['phase'] for s in driver['stages']] == ['reference', 'probe', 'formal']
    assert all(s['return_code'] == 0 and not s['timed_out'] and s['complete_file_exists']
               and not s['released_gpu']['target_apps'] for s in driver['stages'])
    for name in ['driver-start.json', 'formal/start.json']:
        start = json.loads((out / name).read_text())
        expected = dict(ref(directory / 'execution-plan.json'), path=p['application'] + '/execution-plan.json')
        assert start['plan'] == expected
    done = json.loads((out / 'formal/complete.json').read_text())
    ack = json.loads((out / 'formal/epoch-001-persistent-ack.json').read_text())
    assert done['acknowledgement'] == ack and done['parent_adapter_files'] == p['parent_adapter_files']
    assert done['fresh_optimizer_not_resume'] is True
    content_matches(out / 'formal/epoch-001/checkpoint.json', ack['checkpoint'])
    checkpoint = json.loads((out / 'formal/epoch-001/checkpoint.json').read_text())
    assert checkpoint['epoch'] == 1 and checkpoint['global_step'] == 10
    assert checkpoint['counts'] == done['counts'] and checkpoint['recipe'] == p['recipe']
    assert {k: {x: v[x] for x in ['bytes', 'sha256']} for k, v in ack['files'].items()} == checkpoint['files']
    record(state(directory) / 'core-collected.json', {'core_plan': ref(directory / 'execution-plan.json'),
           'files': {n: ref(out / n) for n in names}, 'large_states_remain_remote': True})


def check_receipt(directory, name):
    receipt = json.loads((state(directory) / name).read_text())
    for item in receipt['files'].values():
        assert ref(Path(item['path'])) == item
    return receipt


def local_once(directory, label, command):
    out = state(directory)
    intent = out / (label + '-intent.json')
    assert not intent.exists(), 'Recover original local stage; do not repeat an unknown operation'
    save(intent, {'at_utc': now(), 'command': command})
    r = subprocess.run(command, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
    save(out / (label + '-response.json'), {'return_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
    assert r.returncode == 0


def bind_infer(directory):
    check_receipt(directory, 'core-collected.json')
    p = core_plan(directory); out = state(directory); formal = out / 'core/formal'
    command = [sys.executable, str(DOWN / 'bind.py'), 'infer', '--template', str(DOWN / 'inference-template.json'),
               '--formal-complete', str(formal / 'complete.json'), '--persistent-ack', str(formal / 'epoch-001-persistent-ack.json'),
               '--verified-assets', str(formal / 'verified-assets.json'), '--remote-verified-assets', p['run_root'] + '/formal/verified-assets.json',
               '--held-encoded-remote', p['application'] + '/downstream-inputs/held80.jsonl.gz',
               '--prompts-gzip-remote', p['application'] + '/downstream-inputs/prompts80.jsonl.gz',
               '--run-root', p['run_root'] + '/infer80', '--output', str(out / 'inference-plan.json')]
    local_once(directory, 'bind-infer', command)


def deployment_items(directory):
    out = state(directory); p = json.loads((out / 'inference-plan.json').read_text()); core = core_plan(directory)
    assert p['enabled'] is True and p['run_root'] == core['run_root'] + '/infer80'
    assert p['budgets'] == {'native_seconds': 1200, 'TERM_grace': 30}
    items = []
    for name, expected in p['source_files'].items():
        item = ref(DOWN / name); assert item['sha256'] == expected
        items.append({'local': item, 'remote': core['application'] + '/downstream/' + name})
    for key in ['held_encoded', 'prompts_gzip']:
        item = p['local_inputs'][key]; assert ref(Path(item['path'])) == item
        items.append({'local': item, 'remote': p[key]['path']})
    items.append({'local': ref(out / 'inference-plan.json'), 'remote': core['application'] + '/downstream/inference-plan.json'})
    assert all(0 < x['local']['bytes'] <= CHUNK for x in items)
    return items


def deploy_infer(directory, remote):
    out = state(directory); items = deployment_items(directory)
    record(out / 'deployment.json', items)
    for index, item in enumerate(items):
        intent = out / f'upload-{index:02d}-intent.json'; receipt = out / f'upload-{index:02d}-receipt.json'
        data = Path(item['local']['path']).read_bytes()
        if not intent.exists():
            save(intent, {'at_utc': now(), **item})
            response = remote.put(item['remote'], data)
        else:
            previous = json.loads(intent.read_text())
            assert all(previous[k] == item[k] for k in ['local', 'remote'])
            response = {'recovered_by_exact_read': True}
        # Unknown upload is recovered only by reading its original path; never rewrite it.
        assert read_small(remote, item['remote']) == data
        if not receipt.exists(): save(receipt, {'item': item, 'response': response, 'remote_byteexact': True})
    durable=out/'deployment-durability.json'
    if not durable.exists():
        code='''import hashlib,json,os
from pathlib import Path
items=ITEMS
for item in items:
 p=Path(item['remote']);b=p.read_bytes()
 assert len(b)==item['local']['bytes'] and hashlib.sha256(b).hexdigest()==item['local']['sha256']
 with p.open('rb') as f:os.fsync(f.fileno())
 current=p.parent;stop=Path(STOP)
 assert current==stop or stop in current.parents
 while True:
  fd=os.open(current,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if current==stop:break
  current=current.parent
print(json.dumps({'status':'ALL_DEPLOYMENT_SHA_FSYNC_PARENT_CHAIN_PASS','files':len(items)}))
'''.replace('ITEMS',repr(items)).replace('STOP',repr(core_plan(directory)['application'].rsplit('/',1)[0]))
        result=remote.call('run_command',command=shlex.join([core_plan(directory)['python'],'-c',code]),cwd='',timeout_seconds=30)
        assert result['exit_code']==0;record(durable,result)
    record(out / 'deployed.json', {'files': {str(i): item['local'] for i, item in enumerate(items)},
                                  'deployment': ref(out / 'deployment.json'), 'all_remote_byteexact': True,
                                  'durability':ref(durable)})


def start_infer(directory, remote):
    out = state(directory); check_receipt(directory, 'deployed.json'); p = core_plan(directory)
    assert json.loads((out / 'deployment.json').read_text()) == deployment_items(directory)
    intent = out / 'infer-start-intent.json'
    assert not intent.exists(), 'Start already intended: inspect the ORIGINAL infer80 run, never start again'
    args = {'command': shlex.join([p['python'], p['application'] + '/downstream/driver.py', '--plan', p['application'] + '/downstream/inference-plan.json']),
            'cwd': control.relative(p['application'] + '/downstream'),
            'env': dict(p['environment'], CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')}
    save(intent, {'at_utc': now(), 'arguments': args, 'plan': ref(out / 'inference-plan.json')})
    response = remote.call('start_process', **args)
    save(out / 'infer-start-response.json', response); print(json.dumps(response))


def fetch_archive(remote, expected, destination):
    assert 0 < expected['bytes'] <= 64 * 1024**2
    if destination.exists(): content_matches(destination, expected); return
    part = destination.with_suffix('.zip.partial')
    offset = part.stat().st_size if part.exists() else 0
    assert offset <= expected['bytes']
    with part.open('ab') as f:
        while offset < expected['bytes']:
            size = min(CHUNK, expected['bytes'] - offset)
            data = read_chunk(remote, expected['path'], offset, size)
            assert len(data) == size, 'Short read; retain partial and original receipt'
            f.write(data); f.flush(); os.fsync(f.fileno()); offset += size
    content_matches(part, expected); part.rename(destination)


def unpack_verified(archive, destination, expected_members):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == expected_members
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        assert sum(i.file_size for i in z.infolist()) <= 128 * 1024**2
        inventory = json.loads(z.read('inventory.json'))
        assert set(names) == set(inventory) | {'inventory.json'}
        for name, expected in inventory.items():
            data = z.read(name)
            assert len(data) == expected['bytes'] and digest(data) == expected['sha256'], name
        for name in names: write_exact(destination / name, z.read(name))
    return inventory


def collect_infer(directory, remote):
    out = state(directory); p = core_plan(directory); root = p['run_root'] + '/infer80'
    write_exact(out / 'terminal-receipt.json', read_small(remote, root + '/terminal-receipt.json'))
    receipt = json.loads((out / 'terminal-receipt.json').read_text())
    assert receipt['archive']['path'] == root + '/terminal.zip'
    archive = out / 'terminal.zip'; fetch_archive(remote, receipt['archive'], archive)
    collected = out / 'collected'; unpack_verified(archive, collected, receipt['members'])
    assert (collected / 'plan.json').read_bytes() == (out / 'inference-plan.json').read_bytes()
    for name, path in [('formal-complete.json', out / 'core/formal/complete.json'),
                       ('persistent-ack.json', out / 'core/formal/epoch-001-persistent-ack.json')]:
        assert (collected / 'inputs' / name).read_bytes() == path.read_bytes()
    record(out / 'infer-collected.json', {'files': {'archive': ref(archive), 'inventory': ref(collected / 'inventory.json'),
           'plan': ref(collected / 'plan.json'), 'driver': ref(collected / 'run/driver-result.json'),
           'native_complete': ref(collected / 'run/native/complete.json')}, 'full_inventory_reopen_checked': True})


def bind_quality(directory):
    check_receipt(directory, 'infer-collected.json'); out = state(directory)
    command = [sys.executable, str(DOWN / 'bind.py'), 'quality', '--template', str(DOWN / 'quality-template.json'),
               '--formal-complete', str(out / 'core/formal/complete.json'), '--persistent-ack', str(out / 'core/formal/epoch-001-persistent-ack.json'),
               '--native-directory', str(out / 'collected/run/native'), '--driver-result', str(out / 'collected/run/driver-result.json'),
               '--run-root', str(out / 'quality'), '--output', str(out / 'quality-plan.json')]
    local_once(directory, 'bind-quality', command)


def quality_stage(directory, mode):
    out = state(directory)
    if mode == 'evaluate': assert (out / 'quality/freeze.json').is_file()
    local_once(directory, mode, [sys.executable, str(DOWN / 'quality.py'), '--plan', str(out / 'quality-plan.json'), mode])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('mode', nargs='?', default='core_status', choices=['core_status', 'collect_core', 'bind_infer',
        'deploy_infer', 'start_infer', 'infer_status', 'collect_infer', 'bind_quality', 'assemble', 'evaluate'])
    args = parser.parse_args(); directory = args.directory.resolve()
    remote_modes = {'core_status', 'collect_core', 'deploy_infer', 'start_infer', 'infer_status', 'collect_infer'}
    try:
        remote = control.Remote() if args.mode in remote_modes else None
        if args.mode in {'core_status', 'infer_status'}: inspect(directory, remote, args.mode == 'infer_status')
        elif args.mode in {'assemble', 'evaluate'}: quality_stage(directory, args.mode)
        elif args.mode in remote_modes: globals()[args.mode](directory, remote)
        else: globals()[args.mode](directory)
    except Exception as e:
        save(state(directory) / ('failure-' + args.mode + '-' + str(time.time_ns()) + '.json'),
             {'at_utc': now(), 'error_type': type(e).__name__, 'original_intents_preserved': True})
        raise


if __name__ == '__main__': main()
