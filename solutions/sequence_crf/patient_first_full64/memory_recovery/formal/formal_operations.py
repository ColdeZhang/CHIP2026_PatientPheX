"""Deploy frozen text/data, start each phase once, collect existing small files."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['deploy','start','collect'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--phase', choices=['train','infer'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--mcp-client-dir', default='/workspace/patientphex-tools')
    args = parser.parse_args(); b = args.directory
    resource = json.loads((b/'resource.json').read_text())
    sys.path.insert(0, args.mcp_client_dir)
    from mcp_client import Client
    client = Client(); client.initialize()
    def save(name, item):
        with (b/name).open('x') as handle:
            json.dump(item, handle, indent=2); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        descriptor = os.open(str(b), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    def call(name, **kwargs):
        result = client.request('tools/call', dict(name=name, arguments=dict(mount_id=resource['mount'], **kwargs)))
        if result.get('isError'): raise RuntimeError(json.dumps(result))
        return result['structuredContent']
    def run(code, timeout=30):
        return call('run_command', command='python3 -c '+shlex.quote(code), cwd='.', timeout_seconds=timeout)
    if args.mode == 'deploy':
        assert not (b/'deployment-intent.json').exists(), 'Inspect prior remote outcome; never blindly deploy twice'
        payload = json.loads((b/'payload.json').read_text()); buffer = io.BytesIO(); members = {}
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, item in payload.items():
                assert Path(name).name == name
                raw = Path(item['path']).read_bytes()
                assert len(raw) == item['bytes'] and digest(raw) == item['sha256']
                if name not in resource.get('reuse_remote',{}): archive.writestr(name, raw)
                members[name] = {k:item[k] for k in ['bytes','sha256']}
        raw = buffer.getvalue(); (b/'deployment.zip').write_bytes(raw)
        save('deployment-intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
             archive_bytes=len(raw), archive_sha256=digest(raw), members=members,
             remote=resource['remote_preparation'], no_GPU=True))
        code = "from pathlib import Path;p=Path(ROOT);assert not p.exists();p.mkdir(parents=True)".replace('ROOT',repr(resource['remote_preparation']))
        result = run(code); assert result['exit_code'] == 0, result
        parts = []
        for index, offset in enumerate(range(0,len(raw),256*1024)):
            name = 'upload.part%03d'%index; parts.append(name)
            call('write_file', path=resource['remote_preparation']+'/'+name, encoding='base64',
                 content=base64.b64encode(raw[offset:offset+256*1024]).decode())
        code = '''from pathlib import Path
import io,zipfile,hashlib,json,os
root=Path(ROOT);expected=MEMBERS;reuse=REUSE
raw=b''.join((root/n).read_bytes() for n in PARTS);assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
z=zipfile.ZipFile(io.BytesIO(raw));assert set(z.namelist())==set(expected)-set(reuse)
for name,item in expected.items():
 data=Path(reuse[name]).read_bytes() if name in reuse else z.read(name)
 assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 assert not (root/name).exists()
 with (root/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 if name.endswith('.py'):compile(data,name,'exec')
print(json.dumps(dict(files=len(expected),archive_sha256=ARCHIVE_SHA,all_SHA_compile=True,no_GPU=True)))
'''.replace('ROOT',repr(resource['remote_preparation'])).replace('MEMBERS',repr(members)).replace('PARTS',repr(parts)).replace('ARCHIVE_SHA',repr(digest(raw))).replace('REUSE',repr(resource.get('reuse_remote',{})))
        result = run(code); assert result['exit_code'] == 0, result
        save('deployment.json', result); print(json.dumps(result)); return
    assert args.phase
    phase = args.phase; phase_path = b/(phase+'-plan.json'); plan = json.loads(phase_path.read_text())
    assert plan['phase'] == phase and plan['gpu_uuid'] == resource['gpu_uuid']
    remote_plan = resource['remote_preparation_absolute']+'/'+phase+'-plan.json'
    control = resource['control_root']+'/'+phase
    if args.mode == 'start':
        assert args.checkpoint and len(args.checkpoint) == 40 and (b/'deployment.json').exists()
        assert not (b/(phase+'-launch-intent.json')).exists() and not (b/(phase+'-job.json')).exists()
        final_identity = {}
        if phase == 'infer':
            training = json.loads((b/'training-collection.json').read_text())
            assert training['wrapper']['returncode'] == 0 and not training['wrapper'].get('endpoint_error')
            assert training['wrapper']['all_epochs_persistent_and_readback_verified']
            receipt = training['wrapper']['durability']['2']['receipt']
            assert receipt['completed_epochs'] == 2 and receipt['global_step'] == 68
            assert receipt['file_directory_fsync'] and receipt['reopened_SHA_all_exact']
            root = resource['run_root']+'/formal2ep'
            assert receipt['checkpoint'] == root+'/epoch-002'
            for name in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']:
                final_identity[root+'/epoch-002/'+name] = receipt['files'][name]['sha256']
            final_identity[root+'/epoch-002/checkpoint.json'] = receipt['checkpoint_metadata_sha256']
            final_identity[root+'/training_complete.json'] = training['wrapper']['training_complete_sha256']
        code = '''from pathlib import Path
import hashlib,json,subprocess,shutil,datetime
p=json.loads(Path(PLAN_PATH).read_text());assert hashlib.sha256(Path(PLAN_PATH).read_bytes()).hexdigest()==PLAN_SHA
for path,digest in FINAL_IDENTITY.items():
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 assert h.hexdigest()==digest
assert not Path(CONTROL).exists() and not Path(p['output_dir']).exists()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
target=[s for s in memory.splitlines() if p['gpu_uuid'] in s];assert len(target)==1 and p['gpu_uuid'] not in apps and int(target[0].split(',')[1])<512
free=shutil.disk_usage(p['storage_parent']).free;assert free>=p['minimum_free_disk_bytes']
Path(CONTROL).mkdir(parents=True)
print(json.dumps(dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=target[0],target_no_process=True,persistent_disk_free=free,plan_sha256=PLAN_SHA)))
'''.replace('PLAN_PATH',repr(remote_plan)).replace('PLAN_SHA',repr(digest(phase_path.read_bytes()))).replace('CONTROL',repr(control)).replace('FINAL_IDENTITY',repr(final_identity))
        result = run(code); assert result['exit_code'] == 0, result
        save(phase+'-fresh-resource.json', result)
        command = shlex.join(['python3','-u',resource['remote_preparation_absolute']+'/formal_driver.py','--plan',remote_plan])
        mount_root = Path(resource['remote_preparation_absolute']).parents[len(Path(resource['remote_preparation']).parts)-1]
        assert str(Path(resource['remote_preparation_absolute']).relative_to(mount_root)) == resource['remote_preparation']
        arguments = dict(command=command,cwd=str(Path(control).relative_to(mount_root)),env=resource['env'])
        save(phase+'-launch-intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
             arguments=arguments,checkpoint=args.checkpoint,plan_sha256=digest(phase_path.read_bytes()),
             once_only=True,unknown_outcome='Inspect original job/files. Never call start again.'))
        result = call('start_process', **arguments)
        started = datetime.fromisoformat(result['started_at'].replace('Z','+00:00'))
        save(phase+'-job.json', dict(result=result,arguments=arguments,checkpoint=args.checkpoint,
             cap_seconds=plan['cap_seconds'],grace_seconds=30,
             approximate_TERM_deadline=(started+timedelta(seconds=plan['cap_seconds'])).isoformat(),
             approximate_KILL_deadline=(started+timedelta(seconds=plan['cap_seconds']+30)).isoformat(),
             actual_deadline='remote driver-started.json'))
        print(json.dumps(result)); return
    assert (b/(phase+'-launch-intent.json')).exists()
    collection_name = 'training-collection.json' if phase == 'train' else 'inference-collection.json'
    assert not (b/collection_name).exists()
    code = '''from pathlib import Path
import json,hashlib,base64
control=Path(CONTROL);out=Path(OUTPUT);phase=PHASE
if not (control/'wrapper-complete.json').exists():
 print(json.dumps(dict(complete=False,control_files=sorted(p.name for p in control.glob('*')))))
else:
 paths={n:control/n for n in ['driver-started.json','wrapper-complete.json','stdout.log','stderr.log']}
 names=['run_manifest.json','token-mask-audit.json','gpu-preflight.json','model-coverage.json','base-loading-info.json','first-nonzero-update.json','progress.json','training_complete.json'] if phase=='train' else ['responses.jsonl','responses.manifest.json','responses.checkpoint.json','responses.loading.json','responses.decisions.json','responses.raw-index.json','generation-failure.json','complete.json']
 paths.update({n:out/n for n in names if (out/n).exists()})
 if phase=='infer':
  for directory in ['raw-rows','decision-rows','generation-rows']:
   paths.update({str(p.relative_to(out)):p for p in (out/directory).glob('*.json') if p.is_file()})
 if phase=='train':
  paths.update({str(p.relative_to(out)):p for p in out.glob('epoch-*/checkpoint.json') if p.parent.is_dir()})
  paths.update({p.name:p for p in out.glob('failure-*.json')})
  paths.update({p.name:p for p in out.glob('durability-epoch-[0-9][0-9][0-9].json') if p.is_file()})
 result={}
 for name,path in paths.items():
  raw=path.read_bytes();assert len(raw)<1024*1024
  result[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
 print(json.dumps(dict(complete=True,files=result)))
'''.replace('CONTROL',repr(control)).replace('OUTPUT',repr(plan['output_dir'])).replace('PHASE',repr(phase))
    result = run(code); assert result['exit_code'] == 0, result
    actual = json.loads(result['stdout'])
    if not actual['complete']:
        print(json.dumps(actual)); return
    target = b/('training-terminal' if phase == 'train' else 'inference-terminal'); target.mkdir(exist_ok=False)
    files = {}
    for name, item in actual['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        raw = base64.b64decode(item['base64']); assert len(raw)==item['bytes'] and digest(raw)==item['sha256']
        path = target/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
        files[name]={k:item[k] for k in ['bytes','sha256']}
    save(collection_name,dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,files=files,
         direct_existing_files_only=True,wrapper=json.loads((target/'wrapper-complete.json').read_text())))
    print(json.dumps(json.loads((b/collection_name).read_text())))


if __name__ == '__main__':
    main()
