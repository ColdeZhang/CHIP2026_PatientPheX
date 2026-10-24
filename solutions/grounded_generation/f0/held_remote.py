#!/usr/bin/env python3
"""Complete necessary held public-query transport; frozen policy and original intent recovery.
Frozen source/data plus plan. Original chunk/claim/UNKNOWN recovery and fsync retained.
"""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
import zipfile

MOUNT='1593f159-19bc-41ca-8b4c-2e7455f5db99'
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL='grounded-generation-v1/normalization-v1/held-F0-v1'
ROOT=PROJECT+'/'+REL
REUSE_ROOTS={PROJECT+'/grounded-generation-v1/normalization-v1/query-v1'}
START_MARKERS={'start-intent.json','driver-started.json','child-started.json','wrapper-complete.json',
    'fresh-resource-gate.json','stdout.log','stderr.log'}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def ref(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=digest(raw))


def identity(item):return {key:item[key] for key in ['bytes','sha256']}


def safe_name(name):
    p=PurePosixPath(name)
    assert isinstance(name,str) and name and p.parts and not p.is_absolute() and '..' not in p.parts
    assert p.as_posix()==name and '\\' not in name
    return name


def put(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    parent=path.resolve().parent
    while True:
        descriptor=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if parent.parent==parent:break
        parent=parent.parent


def save(path,obj):put(path,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())


def save_exact(path,obj):
    raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists():assert path.read_bytes()==raw,'Existing receipt differs; preserve both attempts'
    else:put(path,raw)


def checked_plan(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    assert plan['operations_source']==ref(__file__)
    assert plan['schema']=='grounded-fixed-held-public-queries-v1' and plan['complete_necessary_queries']
    assert plan['budget_seconds']==180 and plan['budget_seconds']==plan['authorized_budget_seconds'] and plan['grace_seconds']==30
    assert plan['output_dir']==ROOT+'/output'
    return plan


def payload_sources(plan):
    items=plan['payload_sources']
    assert set(items)=={'strict.py','semantic.py','public_bank.py','policy_v1.json','held_query.py','held_driver.py','queries.json'}
    for name,item in items.items():safe_name(name);assert ref(item['path'])==item
    return items


def expected_remote_files(plan):
    """Build the driver's absolute-path files map without pack/deploy/start."""
    return {ROOT+'/'+name:identity(item) for name,item in payload_sources(plan).items()}


def pack(plan_path,out):
    plan=checked_plan(plan_path);items=payload_sources(plan)
    assert all(plan['files'][ROOT+'/'+name]==identity(item) for name,item in items.items())
    items['held-query-plan.json']=ref(plan_path)
    if out.exists():
        assert (out/'package.json').is_file(),'Partial prior pack is preserved; do not repackage'
        old=json.loads((out/'package.json').read_text())
        assert old['source']==ref(__file__) and old['members']==items and old['plan']==ref(plan_path)
        assert ref(out/'deployment.zip')==old['archive']
        print(json.dumps(dict(existing_package=ref(out/'package.json'),repacked=False)));return
    reuse={};packages=[]
    for package_ref in plan.get('reuse_packages',[]):
        assert ref(package_ref['path'])==package_ref
        package=json.loads(Path(package_ref['path']).read_text())
        assert package['mount']==MOUNT and package['remote'] in REUSE_ROOTS
        packages.append(package_ref)
        for target,item in items.items():
            if target in reuse:continue
            for original,stored in sorted(package['members'].items()):
                safe_name(original)
                if identity(item)==identity(stored):
                    reuse[target]=dict(path=package['remote']+'/'+original,**identity(item),
                        package=package_ref,original_member=original)
                    break
    for target,origin in plan.get('reuse_actual_artifacts',{}).items():
        assert target in items and identity(origin)==identity(items[target])
        assert any(origin['path'].startswith(r+'/') for r in REUSE_ROOTS)
        assert origin['path']==origin['original_root']+'/'+origin['original_member']
        safe_name(origin['original_member']);reuse[target]=origin
    transfer={name:item for name,item in items.items() if name not in reuse}
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,item in sorted(transfer.items()):z.writestr(safe_name(name),Path(item['path']).read_bytes())
    out.mkdir(parents=True,exist_ok=False);put(out/'deployment.zip',archive.getvalue())
    save(out/'package.json',dict(schema='grounded-fixed-held-query-package-v1',archive=ref(out/'deployment.zip'),
        members=items,transfer_members=transfer,reuse=reuse,reuse_packages=packages,
        plan=ref(plan_path),source=ref(__file__),input_sources=plan['input_sources'],remote=ROOT,mount=MOUNT,
        model_weights_not_in_archive=True,old_zip_controllers_not_reused=True))
    print(json.dumps(ref(out/'package.json')))


REMOTE_COMMON=r'''
from pathlib import Path,PurePosixPath
import hashlib,json,os,zipfile,io
root=Path(SPEC['root']);cfg=SPEC['cfg'];markers=set(SPEC['markers'])
def sha(raw):return hashlib.sha256(raw).hexdigest()
def member(name):
 p=PurePosixPath(name)
 assert name and p.parts and not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and chr(92) not in name
 return root/name
def no_symlink(path):
 p=path
 while True:
  assert not p.is_symlink(),str(p)
  if p==root:break
  assert root in p.parents;p=p.parent
def file_ref(path):
 no_symlink(path);raw=path.read_bytes()
 return dict(bytes=len(raw),sha256=sha(raw))
def sync_directory(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def sync_ancestors(path):
 while True:
  sync_directory(path)
  if path.parent==path:break
  path=path.parent
def matching_pids():
 matches=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   argv=(proc/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
   names={'held_driver.py','held_query.py',str(root/'held_driver.py'),str(root/'held_query.py')}
   if any(a in names for a in argv) and (proc/'cwd').resolve()==root:matches.append(int(proc.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return sorted(matches)
'''

INVENTORY=r'''
present={};parts={}
if root.exists():
 assert root.is_dir() and not root.is_symlink()
 for name in cfg['members']:
  p=member(name)
  if p.exists():present[name]=file_ref(p)
 for p in sorted(root.glob('deployment-*.part')):parts[p.name]=file_ref(p)
receipt=root/'deployment-complete.json'
if receipt.exists():no_symlink(receipt)
print(json.dumps(dict(members=present,parts=parts,markers=sorted(n for n in markers if (root/n).exists()),
 matching_pids=matching_pids(),deployment_receipt=json.loads(receipt.read_text()) if receipt.exists() else None,
 output_exists=Path(SPEC['output_dir']).exists())))
'''

DEPLOY=r'''
assert not any((root/n).exists() for n in markers) and not matching_pids()
assert not Path(SPEC['output_dir']).exists()
root.mkdir(parents=True,exist_ok=True);assert not root.is_symlink()
raw=b''
for item in SPEC['chunks']:
 p=member(item['name']);b=p.read_bytes();assert len(b)==item['bytes'] and sha(b)==item['sha256'];raw+=b
assert len(raw)==cfg['archive']['bytes'] and sha(raw)==cfg['archive']['sha256']
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
 assert len(archive.namelist())==len(set(archive.namelist()))
 assert set(archive.namelist())==set(cfg['transfer_members'])
 assert set(cfg['transfer_members'])|set(cfg['reuse'])==set(cfg['members'])
 assert not set(cfg['transfer_members'])&set(cfg['reuse'])
 for name,item in cfg['members'].items():
  p=member(name);no_symlink(p)
  if name in cfg['reuse']:
   origin=cfg['reuse'][name];source=Path(origin['path'])
   origin_root=next(Path(r) for r in SPEC['reuse_roots'] if Path(r) in source.parents)
   assert source==origin_root/origin['original_member']
   check=source
   while True:
    assert not check.is_symlink()
    if check==origin_root:break
    check=check.parent
   b=source.read_bytes()
  else:b=archive.read(name)
  assert len(b)==item['bytes'] and sha(b)==item['sha256']
  p.parent.mkdir(parents=True,exist_ok=True);no_symlink(p)
  if p.exists():assert p.read_bytes()==b
  else:
   with p.open('xb') as handle:handle.write(b);handle.flush();os.fsync(handle.fileno())
  with p.open('rb') as handle:os.fsync(handle.fileno())
  assert file_ref(p)=={k:item[k] for k in ['bytes','sha256']}
  if name.endswith('.py'):compile(b,str(p),'exec')
for p in sorted({member(n).parent for n in cfg['members']},key=lambda x:-len(x.parts)):sync_ancestors(p)
receipt=dict(schema='grounded-fixed-held-query-deployment-v1',package_sha256=SPEC['package_sha256'],
 archive_sha256=cfg['archive']['sha256'],members=len(cfg['members']),transferred_members=len(cfg['transfer_members']),
 same_server_original_SHA_reused_members=len(cfg['reuse']),SHA_fsync_reopen=True,root=str(root),no_model_import=True)
p=root/'deployment-complete.json'
encoded=(json.dumps(receipt,indent=2)+'\n').encode()
if p.exists():assert p.read_bytes()==encoded
else:
 with p.open('xb') as handle:handle.write(encoded);handle.flush();os.fsync(handle.fileno())
sync_ancestors(root)
assert json.loads(p.read_text())==receipt
print(json.dumps(receipt))
'''

CLAIM=r'''
assert not any((root/n).exists() for n in markers) and not matching_pids()
assert not Path(SPEC['output_dir']).exists()
receipt=json.loads((root/'deployment-complete.json').read_text())
assert receipt['package_sha256']==SPEC['package_sha256'] and receipt['SHA_fsync_reopen']
for name,item in cfg['members'].items():assert file_ref(member(name))=={k:item[k] for k in ['bytes','sha256']}
p=root/'start-intent.json'
with p.open('x') as handle:json.dump(SPEC['claim'],handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
sync_ancestors(root)
print(json.dumps(dict(claim_sha256=sha(p.read_bytes()),start_not_yet_called=True)))
'''

OBSERVE=r'''
def small_json(path):
 if not path.exists():return None
 no_symlink(path);size=path.stat().st_size
 if size>32768:return dict(bytes=size,content_omitted='metadata exceeds compact observation cap')
 raw=path.read_bytes()
 try:return dict(bytes=len(raw),sha256=sha(raw),value=json.loads(raw))
 except (ValueError,UnicodeDecodeError):return dict(bytes=len(raw),sha256=sha(raw),not_complete_JSON=True)
def tail(path,limit,lines):
 if not path.exists():return None
 no_symlink(path)
 with path.open('rb') as handle:
  handle.seek(0,2);size=handle.tell();handle.seek(max(0,size-limit));raw=handle.read(limit)
 return dict(total_bytes=size,last_lines=raw.decode(errors='replace').splitlines()[-lines:])
output=Path(SPEC['output_dir']);result={}
for name in ['start-intent.json','driver-started.json','child-started.json','wrapper-complete.json']:
 value=small_json(root/name)
 if value is not None:result[name]=value
for name in ['complete.json','generation-failure.json','run-manifest.json']:
 value=small_json(output/name)
 if value is not None:result['output/'+name]=value
failures=[]
if failures:result['output/'+failures[-1].name]=small_json(failures[-1])
result['stdout_tail']=tail(root/'stdout.log',8192,6)
result['stderr_tail']=tail(root/'stderr.log',4096,3)
result['matching_pids']=matching_pids()
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pack','deploy','start','observe'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--client-dir',type=Path,default=Path('/workspace/patientphex-tools'))
    args=parser.parse_args();out=args.directory.resolve();os.umask(0o077)
    if args.stage=='pack':
        assert args.plan,'Pack requires explicit frozen held-query-plan.json'
        pack(args.plan.resolve(),out);return
    cfg=json.loads((out/'package.json').read_text())
    assert cfg['source']==ref(__file__) and cfg['remote']==ROOT and cfg['mount']==MOUNT
    assert ref(out/'deployment.zip')==cfg['archive'] and ref(cfg['plan']['path'])==cfg['plan']
    plan=checked_plan(cfg['plan']['path'])
    with (out/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backoffs=sorted((out/'transport').glob('offline-backoff-*.json'))
        if backoffs:assert time.time()>=json.loads(backoffs[-1].read_text())['not_before_unix'],'Unified LUNA backoff is active'
        sys.path.insert(0,str(args.client_dir));from mcp_client import Client
        client=Client()
        try:client.initialize()
        except Exception as error:
            save(out/'transport'/('initialize-'+str(time.time_ns())+'.error.json'),
                dict(UNKNOWN=True,error_type=type(error).__name__,no_remote_mutation_called=True))
            if 'LUNA_OFFLINE' in str(error):
                save(out/'transport'/('offline-backoff-'+str(time.time_ns())+'.json'),dict(not_before_unix=time.time()+185))
            raise RuntimeError('Initialization failed; no remote mutation called; preserve error and inventory later') from None
        def call(tool,arguments,tag):
            path=out/'transport'/(tag+'-'+str(time.time_ns()));safe=dict(arguments)
            if 'content' in safe:safe['content']='[frozen bytes omitted]'
            save(path.with_suffix('.intent.json'),dict(tool=tool,arguments=safe))
            try:result=client.request('tools/call',dict(name=tool,arguments=arguments))
            except Exception as error:
                save(path.with_suffix('.error.json'),dict(UNKNOWN=True,error_type=type(error).__name__,automatic_retry=False))
                raise RuntimeError('UNKNOWN; preserve intent and inventory before further action; never replay start') from None
            text=json.dumps(result)
            for key in ['MCP_TOKEN','MCP_URL']:
                if os.environ.get(key):text=text.replace(os.environ[key],'[REDACTED]')
            result=json.loads(text)
            if result.get('isError'):
                save(path.with_suffix('.error.json'),result)
                if 'LUNA_OFFLINE' in text:
                    save(out/'transport'/('offline-backoff-'+str(time.time_ns())+'.json'),dict(not_before_unix=time.time()+185))
                raise RuntimeError('Remote tool error preserved; LUNA requires unified >=185 second backoff')
            actual=result.get('structuredContent')
            if actual is None:actual=json.loads(next(x['text'] for x in result['content'] if x['type']=='text'))
            save(path.with_suffix('.response.json'),actual);return actual
        def command(body,tag,extra=None):
            spec=dict(root=ROOT,cfg=cfg,markers=sorted(START_MARKERS),output_dir=plan['output_dir'],
                package_sha256=ref(out/'package.json')['sha256'],reuse_roots=sorted(REUSE_ROOTS),**(extra or {}))
            code='SPEC='+repr(spec)+'\n'+REMOTE_COMMON+'\n'+body
            actual=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=40),tag)
            assert actual['exit_code']==0,'Remote command failed; preserve response, no automatic replay'
            return json.loads(actual['stdout'])
        if args.stage=='deploy':
            assert not (out/'start-intent.json').exists(),'Never deploy after any local start intent'
            inventory=command(INVENTORY,'inventory-before-deploy')
            assert not inventory['markers'] and not inventory['matching_pids'] and not inventory['output_exists']
            for name,actual in inventory['members'].items():assert actual==identity(cfg['members'][name])
            raw=(out/'deployment.zip').read_bytes();chunks=[]
            for offset in range(0,len(raw),32768):
                data=raw[offset:offset+32768];name='deployment-%06d.part'%offset
                expected=dict(bytes=len(data),sha256=digest(data));chunks.append(dict(name=name,**expected))
                if name in inventory['parts']:assert inventory['parts'][name]==expected;continue
                actual=call('write_file',dict(mount_id=MOUNT,path=REL+'/'+name,content=base64.b64encode(data).decode(),
                    encoding='base64',create_parents=True),'part-'+str(offset))
                assert actual['size']==expected['bytes'] and actual['sha256']==expected['sha256']
            assert set(inventory['parts'])<={item['name'] for item in chunks},'Unrecognized old chunks; preserve namespace'
            actual=command(DEPLOY,'verify-copy-fsync-reopen',dict(chunks=chunks))
            save_exact(out/'deployment.json',dict(actual=actual,package=ref(out/'package.json')))
        elif args.stage=='start':
            assert plan['authorized'] is True and type(plan['budget_seconds']) is int and plan['budget_seconds']>0
            assert plan['grace_seconds']==30 and (out/'deployment.json').is_file()
            assert not (out/'start-intent.json').exists(),'Any prior start intent forbids another start call; observe only'
            deployed=json.loads((out/'deployment.json').read_text())
            assert deployed['package']==ref(out/'package.json') and deployed['actual']['SHA_fsync_reopen']
            inventory=command(INVENTORY,'inventory-before-start')
            assert not inventory['markers'] and not inventory['matching_pids'] and not inventory['output_exists']
            assert inventory['deployment_receipt']['package_sha256']==ref(out/'package.json')['sha256']
            env=plan['environment'];assert not {'MCP_TOKEN','MCP_URL'}&set(env)
            assert env['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and env['PATIENTPHEX_DURABILITY_ROOT']==plan['storage_parent']
            executable=plan['python'];launch=shlex.join([executable,'-u','held_driver.py'])
            intent=dict(command=launch,cwd=REL,environment=env,package=ref(out/'package.json'),plan=cfg['plan'],
                source=ref(__file__),inventory=inventory,no_retry=True,fresh_GPU_gate_inside_driver=True)
            save(out/'start-intent.json',intent)
            claim=command(CLAIM,'claim-unique-start',dict(claim=intent))
            save(out/'remote-start-claim.json',claim)
            actual=call('start_process',dict(mount_id=MOUNT,cwd=REL,command=launch,env=env),'start')
            save(out/'started.json',actual)
        else:
            actual=command(OBSERVE,'observe-compact')
            save(out/('observation-'+str(time.time_ns())+'.json'),actual);print(json.dumps(actual))
    print(json.dumps(dict(stage=args.stage,complete=True,directory=str(out))))


if __name__=='__main__':main()
