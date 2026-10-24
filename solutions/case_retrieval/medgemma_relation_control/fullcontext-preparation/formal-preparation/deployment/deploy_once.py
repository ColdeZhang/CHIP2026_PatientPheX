"""Upload only source/input bundle; never starts a GPU or background process."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import os
import shlex
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
D = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (D/name).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    assert not (D/'deployment-intent.json').exists(), 'Inspect prior deployment; never overwrite/redeploy blindly'
    plan = json.loads((D/'run-plan.json').read_text())
    inventory = json.loads((D/'source-bundle.json').read_text())
    assert sha(D/'source-bundle.json') == plan['source_bundle_sha256']
    payload = Path(inventory['bundle']['path']).read_bytes()
    assert len(payload) == inventory['bundle']['bytes'] and hashlib.sha256(payload).hexdigest() == inventory['bundle']['sha256']
    remote, mount = plan['namespace'], plan['mount_id']
    save('deployment-intent.json', {'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'run_plan_sha256':sha(D/'run-plan.json'),'deployer_sha256':sha(__file__),
                                    'namespace':remote,'mount_id':mount,'no_GPU_or_start_process':True})
    client = Client()
    def call(name, args):
        result = client.request('tools/call', {'name':name, 'arguments':{'mount_id':mount, **args}})
        if result.get('isError'):
            raise RuntimeError('MCP tool error; preserve deployment intent, no automatic retry')
        return result.get('structuredContent') or json.loads(result['content'][0]['text'])
    def command(source):
        result = call('run_command', {'command':'python3 -c '+shlex.quote(source),'cwd':'.','timeout_seconds':30})
        if result['exit_code'] != 0:
            raise RuntimeError('Bounded deployment command failed: '+json.dumps(result))
        return result
    try:
        client.initialize()
        preflight = """from pathlib import Path
import hashlib,json,shutil,os
p=Path(REMOTE)
assert not p.exists(),'Existing namespace must be inspected'
assert not Path(OUTPUT).exists(),'Probe output already exists; no relaunch'
assert shutil.disk_usage('.').free>20000000,'Insufficient tiny-deployment disk space'
reuse=REUSE
for name,r in reuse.items():
 f=Path(name)
 assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],name
assert Path(MODEL).is_dir(),'Existing model directory missing'
p.mkdir(parents=True,mode=0o700)
print(json.dumps(dict(namespace=str(p),cwd=str(Path.cwd()),reused_small_files_SHA_verified=len(reuse),disk_free=shutil.disk_usage('.').free,shm_free=shutil.disk_usage('/dev/shm').free,weights_not_opened=True)))
""".replace('REMOTE',repr(remote)).replace('OUTPUT',repr(plan['probe_output'])).replace('REUSE',repr(plan['reused_source_files'])).replace('MODEL',repr(plan['model']))
        save('remote-preflight.json', command(preflight))
        chunks, receipts = [], []
        for i, offset in enumerate(range(0,len(payload),256*1024)):
            data=payload[offset:offset+256*1024]
            record={'path':remote+f'/bundle.part-{i:03d}','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            receipt=call('write_file',{'path':record['path'],'content':base64.b64encode(data).decode(),'encoding':'base64','create_parents':False})
            chunks.append(record)
            receipts.append({'part':record,'receipt':receipt})
        save('upload-receipts.json',receipts)
        extract = """from pathlib import Path
import gzip,hashlib,io,json,os,tarfile
parts=PARTS
for r in parts:
 b=Path(r['path']).read_bytes()
 assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
payload=b''.join(Path(r['path']).read_bytes() for r in parts)
assert hashlib.sha256(payload).hexdigest()==BUNDLESHA
members=MEMBERS
p=Path(REMOTE)
actual={}
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)),mode='r:') as arc:
 names=arc.getnames()
 assert len(names)==len(set(names)) and set(names)==set(members)
 for info in arc.getmembers():
  assert info.isfile() and not Path(info.name).is_absolute() and '..' not in Path(info.name).parts
  r=members[info.name];data=arc.extractfile(info).read()
  assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
  f=p/info.name;f.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
  with f.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
  back=f.read_bytes()
  actual[info.name]=dict(bytes=len(back),sha256=hashlib.sha256(back).hexdigest())
  assert actual[info.name]==dict(bytes=r['bytes'],sha256=r['sha256'])
directories={p}|{(p/n).parent for n in members}
for directory in directories:
 fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='deployed_all9_fullcontext_formal_SHA_verified_no_GPU_launch',namespace=str(p),actual_files=actual,bundle_sha256=BUNDLESHA,model_weights_copied=False,native_tokenizer_rerun=False)))
"""
        # Keep extraction narrow: no shell archive extraction or archive-controlled paths.
        extract=extract.replace('PARTS',repr(chunks)).replace('BUNDLESHA',repr(inventory['bundle']['sha256'])).replace('MEMBERS',repr(inventory['members'])).replace('REMOTE',repr(remote))
        verified=command(extract)
        save('remote-deployment-verification.json',verified)
        actual=json.loads(verified['stdout'])
        save('deployment-complete.json',{'status':'source_inputs_deployed_owner_GPU_launch_pending',
                                         'mount_id':mount,'namespace':remote,'run_plan_sha256':sha(D/'run-plan.json'),
                                         'bundle_sha256':inventory['bundle']['sha256'],'actual_files':actual['actual_files'],
                                         'remote_verification_sha256':sha(D/'remote-deployment-verification.json'),
                                         'deployer_sha256':sha(__file__),'no_start_process_no_GPU_no_quality':True})
        print(json.dumps({'deployed':len(actual['actual_files']),'run_plan_sha256':sha(D/'run-plan.json'),
                          'deployment_complete_sha256':sha(D/'deployment-complete.json')}),flush=True)
    except BaseException as error:
        save('deployment-interruption.json',{'error_type':type(error).__name__,'error_text':str(error),
                                             'namespace':remote,'GPU_process_not_started':True,
                                             'action':'Inspect same namespace and intent; do not overwrite or automatically retry'})
        raise


if __name__ == '__main__':
    main()
