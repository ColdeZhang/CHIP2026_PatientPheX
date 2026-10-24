"""Persist the closed alpha05 CPU endpoints; reuse verified bounded transfer logic.
No model, gold access, native decode or scoring. Original two NN ZIPs stay in place.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time
import zipfile

MOUNT = '476f81cf-09e7-445b-a7f0-d0e12b12e50f'
TARGET = 'sequence-round2/patient-balanced-alpha05/endpoint-v1'
CHUNK = 65536


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def sync(path):
    while True:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if path.parent == path:
            break
        path = path.parent


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data
    else:
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    with path.open('rb') as f:
        os.fsync(f.fileno())
    sync(path.parent)


def safe(name):
    p = PurePosixPath(name)
    assert name and '\\' not in name and '\x00' not in name
    assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == name
    return p


def pack(root, out):
    assert not (out/'package.json').exists()
    sources=Path(__file__).resolve().parent;entries={}
    names=['formal-terminal-summary.json','final80/terminal-summary.json','quality/terminal-summary.json','RESULT.md','formal-plan.json','train_patient_balanced.py','mixed_loss.py','interface_recovery/row_loss_scope_v2.py','formal_driver.py','formal_remote.py','formal_collect.py','final80/final_infer.py','final80/final_parser.py','final80/patient_first.py','final80/raw_store.py','final80/infer_driver.py','final80/infer_remote.py','final80/infer_collect.py','final80/final80-plan.json','final80/input-sources.json','final80/binding-result.json','quality/plan.json','quality/support.py','quality/assemble.py','quality/score.py','endpoint_supplement.py']
    for name in names:entries['source/'+name]=sources/name
    for phase in ['formal-v1','final80-v1']:
        for name in ['collection.json','package.json','collection-transfer/archive-binding.json']:
            entries['runs/'+phase+'/'+name]=root/phase/name
    entries['runs/formal-v1/loss-call-coverage.json']=root/'formal-v1/loss-call-coverage.json'
    for p in sorted((root/'quality-v1').rglob('*')):
        if p.is_file():entries['quality/'+p.relative_to(root/'quality-v1').as_posix()]=p
    payload={}
    sensitive=[os.environ.get(k,'').encode() for k in ['MCP_TOKEN','MCP_URL','ALI_PASSWD','ALI_ACCOUNT']]
    for name,p in entries.items():
        safe(name);assert p.is_file() and not p.is_symlink();data=p.read_bytes()
        assert not any(secret and secret in data for secret in sensitive)
        payload[name]=data
    external=[]
    for phase in ['formal-v1','final80-v1']:
        collection=json.loads((root/phase/'collection.json').read_text());raw=collection['remote_archive']
        external.append(dict(**raw,mount_relative_path=str(Path(raw['path']).relative_to('/home/zhangyuheng/projects/CHIP2026_PatientPheX')),mount=MOUNT,packaged=False))
    manifest=dict(schema='patient-balanced-alpha05-closed-endpoint-members-v1',members={name:dict(source_path=str(entries[name].resolve()),bytes=len(data),sha256=digest(data)) for name,data in payload.items()},external_archives=external,exclusions=['raw NN archives already durable on original server','weights and optimizer states','training or held gold inputs','credentials'],no_NN=True,no_score=True,original_files_unmodified=True,transport_only=True)
    manifest_data=encoded(manifest);publish(out/'members-manifest.json',manifest_data)
    archive=out/'endpoint-supplement.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(dict(payload,**{'members-manifest.json':manifest_data}).items()):
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o600)<<16;z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with archive.open('rb') as f:os.fsync(f.fileno())
    sync(out)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(payload)|{'members-manifest.json'}
        for name,data in payload.items():assert z.read(name)==data
    raw=archive.read_bytes();chunks=[]
    for offset in range(0,len(raw),CHUNK):
        data=raw[offset:offset+CHUNK];chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(data),sha256=digest(data)))
    report=dict(archive=ref(archive),manifest=ref(out/'members-manifest.json'),members=len(payload),unpacked_bytes=sum(map(len,payload.values())),target_mount=MOUNT,target=TARGET,chunks=chunks,source=ref(__file__),package_readback_exact=True,all_original_sources_SHA_in_manifest=True,external_archives=external)
    publish(out/'package.json',encoded(report));print(json.dumps(dict(package=ref(out/'package.json'),archive=ref(archive),members=len(payload),chunks=len(chunks))))


REMOTE_VERIFY = r'''
import hashlib,json,os,stat,zipfile
from pathlib import Path,PurePosixPath
cfg=json.loads(CFG);root=Path(cfg['target']);root.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sync(p):
 while True:
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if p.parent==p:break
  p=p.parent
def put(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 with p.open('rb') as f:os.fsync(f.fileno())
 sync(p.parent.resolve())
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
originals=[]
for item in cfg['external_archives']:
 raw=Path(item['mount_relative_path']);assert raw.stat().st_size==item['bytes'] and sha(raw)==item['sha256'];originals.append(ref(raw))
data=b''
for item in cfg['chunks']:
 p=root/item['name'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 with p.open('rb') as f:os.fsync(f.fileno())
 sync(p.parent.resolve());data+=p.read_bytes()
assert len(data)==cfg['archive']['bytes'] and hashlib.sha256(data).hexdigest()==cfg['archive']['sha256']
archive=root/'endpoint-supplement.zip';put(archive,data)
manifest=root/'members-manifest.json';assert manifest.stat().st_size==cfg['manifest']['bytes'] and sha(manifest)==cfg['manifest']['sha256']
contents=json.loads(manifest.read_text());members=contents['members'];files={}
with zipfile.ZipFile(archive) as z:
 names=z.namelist();assert len(names)==len(set(names)) and set(names)==set(members)|{'members-manifest.json'}
 assert z.read('members-manifest.json')==manifest.read_bytes()
 for name,item in members.items():
  pp=PurePosixPath(name);assert name and not pp.is_absolute() and '..' not in pp.parts and pp.as_posix()==name and chr(92) not in name and chr(0) not in name
  zi=z.getinfo(name);assert not zi.is_dir() and not stat.S_ISLNK(zi.external_attr>>16)
  p=root/'members'/name;assert not p.is_symlink() and p.resolve().is_relative_to((root/'members').resolve())
  value=z.read(name);assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
  put(p,value);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];files[name]=ref(p)
with manifest.open('rb') as f:os.fsync(f.fileno())
sync(root.resolve());assert sha(archive)==cfg['archive']['sha256']
receipt=dict(schema='patient-balanced-alpha05-endpoint-persistent-v1',mount=cfg['target_mount'],root=str(root.resolve()),archive=ref(archive),manifest=ref(manifest),members=len(members),member_files=files,all_member_SHA_verified=True,files_and_ancestor_directories_fsynced=True,reopened_all_files_after_fsync=True,zip_safety_checked=True,original_archive_references=originals,original_raw_repackaged=False,no_NN=True,no_score=True)
destination=root/'persistent-receipt.json';payload=(json.dumps(receipt,indent=2)+chr(10)).encode();put(destination,payload)
print(json.dumps(dict(receipt=ref(destination),result=receipt)))
'''


def deploy(root, out, client_dir):
    package = json.loads((out / 'package.json').read_text())
    assert ref(package['archive']['path']) == package['archive']
    assert ref(package['manifest']['path']) == package['manifest']
    receipts = out / 'deployment'
    receipts.mkdir(exist_ok=True)
    for p in receipts.glob('*.backoff.json'):
        assert time.time() >= json.loads(p.read_text())['retry_not_before_unix']
    sys.path.insert(0, str(client_dir.resolve()))
    from mcp_client import Client
    client = Client()
    client.initialize()

    def call(name, args, tag):
        stamp = str(time.time_ns())
        summary = dict(args)
        if 'content' in summary:
            summary['content'] = '[content stored in source package; omitted from receipt]'
        publish(receipts / (tag + '-' + stamp + '.intent.json'), encoded(dict(tool=name, arguments=summary)))
        try:
            response = client.request('tools/call', dict(name=name, arguments=args))
        except Exception as exc:
            publish(receipts / (tag + '-' + stamp + '.error.json'), encoded(dict(type=type(exc).__name__, UNKNOWN=True)))
            raise RuntimeError('UNKNOWN; inventory before any retry') from None
        text = json.dumps(response)
        for key in ['MCP_TOKEN', 'MCP_URL']:
            if os.environ.get(key):
                text = text.replace(os.environ[key], '[REDACTED]')
        response = json.loads(text)
        if response.get('isError'):
            publish(receipts / (tag + '-' + stamp + '.error.json'), encoded(response))
            if 'LUNA_OFFLINE' in text:
                publish(receipts / (tag + '-' + stamp + '.backoff.json'), encoded(dict(retry_not_before_unix=time.time() + 180)))
            raise RuntimeError('Tool error saved; no automatic retry')
        result = response.get('structuredContent')
        if result is None:
            result = json.loads(next(b['text'] for b in response['content'] if b['type'] == 'text'))
        publish(receipts / (tag + '-' + stamp + '.response.json'), encoded(result))
        return result

    code = "import json,hashlib;from pathlib import Path;p=Path(" + repr(TARGET) + ");print(json.dumps({'exists':p.exists(),'files':{str(q.relative_to(p)):{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in p.rglob('*') if q.is_file()} if p.exists() else {}}))"
    result = call('run_command', dict(mount_id=MOUNT, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=30), 'inventory')
    assert result['exit_code'] == 0
    inventory = json.loads(result['stdout'])['files']
    raw = Path(package['archive']['path']).read_bytes()
    for item in package['chunks']:
        if item['name'] in inventory:
            assert inventory[item['name']] == {k: item[k] for k in ['bytes', 'sha256']}
            continue
        data = raw[item['offset']:item['offset'] + item['bytes']]
        result = call('write_file', dict(mount_id=MOUNT, path=TARGET + '/' + item['name'],
            content=base64.b64encode(data).decode(), encoding='base64', create_parents=True), 'chunk-' + str(item['offset']))
        assert result['sha256'] == item['sha256'] and result['size'] == item['bytes']
        print(json.dumps(dict(uploaded_bytes=item['offset'] + item['bytes'], total=len(raw))), flush=True)
    manifest = Path(package['manifest']['path']).read_bytes()
    assert len(manifest) <= CHUNK
    if 'members-manifest.json' in inventory:
        assert inventory['members-manifest.json'] == {k: package['manifest'][k] for k in ['bytes', 'sha256']}
    else:
        result = call('write_file', dict(mount_id=MOUNT, path=TARGET + '/members-manifest.json',
            content=manifest.decode(), encoding='utf8', create_parents=True), 'manifest')
        assert result['sha256'] == package['manifest']['sha256']
    code = REMOTE_VERIFY.replace('CFG', repr(json.dumps(package)))
    result = call('run_command', dict(mount_id=MOUNT, cwd='.', command='python3 -c ' + shlex.quote(code), timeout_seconds=60), 'persist-verify')
    assert result['exit_code'] == 0
    remote = json.loads(result['stdout'])
    report = dict(local_package=ref(out / 'package.json'), local_archive=package['archive'],
        local_manifest=package['manifest'], remote=remote, source=ref(__file__), completed_unix=time.time())
    publish(out / 'persistent-receipt.json', encoded(report))
    print(json.dumps(dict(receipt=ref(out / 'persistent-receipt.json'), remote_receipt=remote['receipt'],
                         archive=remote['result']['archive'], members=remote['result']['members'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['pack', 'deploy'])
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--client-dir', type=Path, default=Path('/workspace/patientphex-tools'))
    a = ap.parse_args()
    os.umask(0o077)
    root = a.run_dir.resolve()
    out = root / 'endpoint-supplement-v1'
    out.mkdir(exist_ok=True)
    lock = (out / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.stage == 'pack':
        pack(root, out)
    else:
        deploy(root, out, a.client_dir)


if __name__ == '__main__':
    main()
