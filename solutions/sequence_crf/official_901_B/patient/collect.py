"""Collect the one completed patient B job. Never starts or resumes inference.

Each invocation inventories the same remote archive first. One remote package is
allowed only after the frozen 420-row rc0 endpoint. Failed transfers keep chunks;
UNKNOWN stops without retry, and explicit LUNA_OFFLINE imposes >=180s backoff.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time
import uuid
import zipfile

_RUN_ROOT = None


REMOTE = r'''
import hashlib,json,os,stat,zipfile
from pathlib import Path,PurePosixPath
cfg=json.loads(__PAYLOAD__)
project=Path(cfg['project']);control=Path(cfg['control']);prediction=Path(cfg['output'])
destination=prediction.parent/'complete-collection'
archive=destination/'patient-complete.zip';receipt=destination/'archive.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(p.read_text())
def sync(directory):
 directory=directory.resolve();directory.relative_to(project.resolve())
 while True:
  fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  if directory==project.resolve():break
  directory=directory.parent
def publish(p,b):
 temporary=p.with_name(p.name+'.pending')
 with temporary.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.link(temporary,p);temporary.unlink();sync(p.parent)
def safe(name):
 p=PurePosixPath(name)
 assert name and chr(92) not in name and chr(0) not in name and not p.is_absolute() and '..' not in p.parts
 assert str(p)==name and p.parts[0] in ('control','prediction')
 return p
def inventory_archive():
 assert archive.is_file() and not archive.is_symlink()
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();assert len(names)==len(set(names)) and 'member-index.json' in names
  idx=json.loads(z.read('member-index.json'));assert idx['plan_sha256']==cfg['plan_sha256']
  assert set(names)==set(idx['files'])|{'member-index.json'}
  for name,item in idx['files'].items():
   safe(name);info=z.getinfo(name)
   assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16)
   h=hashlib.sha256();size=0
   with z.open(name) as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b);size+=len(b)
   assert size==item['bytes'] and h.hexdigest()==item['sha256']
  result=dict(schema='patient-B-original-complete-archive-v1',archive=ref(archive),
   member_index_sha256=hashlib.sha256(z.read('member-index.json')).hexdigest(),
   plan_sha256=cfg['plan_sha256'],rows=idx['rows'],files=len(idx['files']),
   wrapper=idx['wrapper'],manifest=idx['manifest'],remote_mapping=idx['remote_mapping'],
   driver_exit=idx['driver_exit'])
 if receipt.exists():assert read(receipt)==result
 return dict(status='ready',**result,archive_preexisting=True,remote_zip_rebuilt=False)
assert control.is_absolute() and prediction.is_absolute() and project.is_absolute()
control.relative_to(project);prediction.relative_to(project)
assert control.name=='infer' and control.parent.name=='control' and prediction.name=='prediction'
assert control.parent.parent==prediction.parent and cfg['rows']==420
assert not control.is_symlink() and not prediction.is_symlink()
# An already published archive is immutable, including after a lost response.
if archive.exists():
 print(json.dumps(inventory_archive()));raise SystemExit(0)
if destination.exists():
 print(json.dumps(dict(status='incomplete_package_inventory_only',entries=sorted(p.name for p in destination.iterdir()),
  remote_zip_rebuilt=False)));raise SystemExit(0)
terminal=control/'wrapper-complete.json'
if not terminal.exists():
 print(json.dumps(dict(status='not_terminal',remote_zip_created=False)));raise SystemExit(0)
wrapper=read(terminal)
assert wrapper['phase']=='infer' and wrapper['returncode']==0 and not wrapper['timed_out'] and not wrapper['killed']
assert 'endpoint_error' not in wrapper and wrapper['complete_coverage'] and wrapper['rows']==420
driver_started=read(control/'driver-started.json')
assert driver_started['plan_sha256']==cfg['plan_sha256']
driver_pid=driver_started['pid'];assert type(driver_pid) is int and driver_pid>1
# wrapper publication precedes the driver's final print. Observe its own PID
# once; do not freeze potentially changing logs while that process can write.
try:
 proc_text=Path('/proc/'+str(driver_pid)+'/stat').read_text()
 proc_state=proc_text[proc_text.rfind(')')+2:].split()[0]
except FileNotFoundError:proc_state='absent'
if proc_state not in ('absent','Z','X'):
 print(json.dumps(dict(status='driver_still_running',pid=driver_pid,state=proc_state,
  remote_zip_created=False,no_polling=True)));raise SystemExit(0)
driver_exit=dict(pid=driver_pid,observed_proc_state=proc_state,no_running_process=True,one_observation=True)
m=read(prediction/'responses.manifest.json')
assert sha(prediction/'responses.manifest.json')==wrapper['manifest_sha256']
assert m['expected_rows']==m['new_rows']==420 and m['cached_rows']==0
assert m['all_B_raw_saved_before_codec'] and m['all_rows_fresh'] and m['no_cache_or_resume'] and m['no_baseline_fallback']
assert m['model_answer_repair'] is False and sum(m['codec_status_counts'].values())==420
assert sha(prediction/'responses.jsonl')==m['output_sha256']==wrapper['responses_sha256']
rows=[json.loads(x) for x in (prediction/'responses.jsonl').read_text().splitlines()]
assert len(rows)==len({r['row_id'] for r in rows})==420
indices=read(prediction/'responses.raw-index.json');decisions=read(prediction/'responses.decisions.json')
assert len(indices)==len(decisions)==420
for name in ['decisions','raw_index']:
 p=Path(m[name]['path']);assert p.parent==prediction
 assert p.stat().st_size==m[name]['bytes'] and sha(p)==m[name]['sha256']
expected={hashlib.sha256(r['row_id'].encode()).hexdigest()+'.json' for r in rows}
for directory in ['raw-rows','generation-rows','decision-rows']:
 assert {p.name for p in (prediction/directory).iterdir()}==expected
for row,index,decision in zip(rows,indices,decisions):
 name=hashlib.sha256(row['row_id'].encode()).hexdigest()+'.json'
 raw=prediction/'raw-rows'/name
 assert Path(index['path'])==raw and raw.stat().st_size==index['bytes'] and sha(raw)==index['sha256']
 content=read(raw);assert content['record']==row and content['raw_before_codec'] and content['origin']=='new_generation_all_B'
 stored=read(prediction/'decision-rows'/name);generation=read(prediction/'generation-rows'/name)
 assert stored['row_id']==decision['row_id']==row['row_id'] and stored['raw']==index
 assert decision==dict(row_id=row['row_id'],**stored['decision'])
 for key in ['row_id','pmc_id','generated_token_ids','input_tokens','output_tokens','hit_token_limit']:
  assert generation[key]==row[key]
 assert generation['decoded_or_parsed'] is False
if cfg['mode']=='inventory':
 print(json.dumps(dict(status='complete_unpacked',rows=420,wrapper=ref(terminal),driver_exit=driver_exit,remote_zip_created=False)));raise SystemExit(0)
assert cfg['mode']=='package'
members={};locations={}
for label,root in [('control',control),('prediction',prediction)]:
 parents={root}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),str(p)
  if p.is_dir():continue
  assert p.is_file(),str(p)
  name=label+'/'+p.relative_to(root).as_posix();safe(name)
  with p.open('rb') as f:os.fsync(f.fileno())
  members[name]=dict(bytes=p.stat().st_size,sha256=sha(p));locations[name]=p
  parents.add(p.parent)
 for directory in parents:sync(directory)
index=dict(schema='patient-B-original-members-v1',plan_sha256=cfg['plan_sha256'],rows=420,
 files=members,wrapper=ref(terminal),manifest=ref(prediction/'responses.manifest.json'),
 remote_mapping={'control':str(control),'prediction':str(prediction)},driver_exit=driver_exit)
index_bytes=(json.dumps(index,indent=2)+'\n').encode()
destination.mkdir(exist_ok=False);sync(destination)
publish(destination/'package-intent.json',(json.dumps(dict(plan_sha256=cfg['plan_sha256'],once_only=True),indent=2)+'\n').encode())
temporary=destination/'patient-complete.zip.pending'
with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for name,p in locations.items():
  z.write(p,name)
  assert p.stat().st_size==members[name]['bytes'] and sha(p)==members[name]['sha256']
 z.writestr('member-index.json',index_bytes)
with temporary.open('rb') as f:os.fsync(f.fileno())
os.link(temporary,archive);temporary.unlink();sync(destination)
result=inventory_archive();persisted={k:v for k,v in result.items() if k not in ('status','archive_preexisting','remote_zip_rebuilt')}
publish(receipt,(json.dumps(persisted,indent=2)+'\n').encode())
print(json.dumps(dict(result,archive_preexisting=False)))
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def sync_chain(directory,root):
    directory=Path(directory).resolve();root=Path(root).resolve();directory.relative_to(root)
    while True:
        descriptor=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if directory==root:break
        directory=directory.parent


def publish(path,data,root):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink();sync_chain(path.parent,root)


def save(path,value,root):
    publish(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode(),root)


def read(path):
    return json.loads(Path(path).read_text())


def safe_name(name):
    path=PurePosixPath(name)
    assert name and chr(92) not in name and chr(0) not in name and not path.is_absolute() and '..' not in path.parts
    assert str(path)==name and path.parts[0] in ('control','prediction')
    return Path(*path.parts)


def redacted(value):
    text=json.dumps(value,ensure_ascii=False)
    for name in ('MCP_TOKEN','MCP_URL'):
        if os.environ.get(name):text=text.replace(os.environ[name],'[REDACTED]')
    return json.loads(text)


def main():
    global _RUN_ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--client-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,help='Default: RUN_DIR/infer-plan.json')
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--chunk-bytes',type=int,default=131072)
    args=parser.parse_args();assert 1<=args.chunk_bytes<=262144
    root=args.run_dir.resolve();assert root.is_dir()
    _RUN_ROOT=root
    plan_path=(args.plan or root/'infer-plan.json').resolve();plan=read(plan_path)
    assert plan['phase']=='infer' and plan['rows']==420 and plan['documents']==100 and plan['patients']==244
    assert plan['cap_seconds']==3600 and plan['grace_seconds']==30
    assert [item['sha256'] for path,item in plan['files'].items() if path.endswith('/infer.py')]==[
        '5b899375fd82121ca8119d85aa5f62a3c0f3c725544a6bbb73516928f77946c7']
    control=Path(plan['control']);prediction=Path(plan['output'])
    assert control.parent.parent==prediction.parent and control.parts[-2:]==('control','infer')
    destination=root/'inference-collection';destination.mkdir(exist_ok=True)
    lock=(destination/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (destination/'offline-backoff.json').exists():
        assert time.time()>=read(destination/'offline-backoff.json')['retry_not_before_unix'], 'LUNA backoff not elapsed'
    for pending in destination.glob('offline-backoff-*.json'):
        assert time.time()>=read(pending)['retry_not_before_unix'], 'LUNA backoff not elapsed'
    receipts=destination/'receipts';receipts.mkdir(exist_ok=True)
    client_path=args.client_dir.resolve()/'mcp_client.py'
    spec=importlib.util.spec_from_file_location('patient_collection_mcp',client_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=module.Client()
    def failure(kind,payload):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        clean=redacted(payload)
        save(receipts/('error-'+stamp+'.json'),dict(at_utc=now(),kind=kind,evidence=clean),root)
        if 'LUNA_OFFLINE' in json.dumps(clean):
            save(destination/('offline-backoff-'+stamp+'.json'),dict(at_utc=now(),retry_not_before_unix=time.time()+180,seconds=180),root)
        raise RuntimeError(kind+': evidence saved; no automatic retry or inference restart')
    def call(name,arguments):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save(receipts/('intent-'+stamp+'.json'),dict(at_utc=now(),tool=name,arguments=redacted(arguments)),root)
        try:result=client.request('tools/call',dict(name=name,arguments=arguments))
        except Exception as error:
            failure('client_request_unknown',dict(type=type(error).__name__))
        if result.get('isError'):failure('tool_error',result)
        value=result.get('structuredContent')
        if value is None:
            try:value=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
            except Exception:failure('unrecognized_tool_response',result)
        if name=='read_file':
            summary={k:v for k,v in value.items() if k!='content'}
            summary['encoded_content_length']=len(value.get('content',''))
        else:summary=value
        save(receipts/('response-'+stamp+'.json'),dict(at_utc=now(),tool=name,result=redacted(summary)),root)
        return value
    try:client.initialize()
    except Exception as error:failure('client_initialize_unknown',dict(type=type(error).__name__))
    cfg={key:plan[key] for key in ('project','control','output','rows')}
    cfg['plan_sha256']=sha(plan_path)
    def remote(mode):
        script=REMOTE.replace('__PAYLOAD__',repr(json.dumps(dict(cfg,mode=mode))))
        response=call('run_command',dict(mount_id=plan['mount'],cwd='.',
            command='python3 -c '+shlex.quote(script),timeout_seconds=60))
        if response.get('exit_code')!=0:failure('remote_collection_check_failed',response)
        return json.loads(response['stdout'])
    # Always inventory first, including after an UNKNOWN packaging response.
    actual=remote('inventory')
    if args.inventory_only:
        print(json.dumps(actual));return
    if actual['status']=='complete_unpacked':actual=remote('package')
    if actual['status']!='ready':
        print(json.dumps(actual));raise SystemExit(2)
    assert actual['rows']==420 and actual['plan_sha256']==cfg['plan_sha256']
    identity=dict(mount=plan['mount'],archive=actual['archive'],plan_sha256=cfg['plan_sha256'],
        member_index_sha256=actual['member_index_sha256'])
    binding=destination/'archive-binding.json'
    if binding.exists():assert read(binding)==identity
    else:save(binding,identity,root)
    remote_zip=actual['archive']
    remote_tool_path=Path(remote_zip['path']).relative_to(Path(plan['project'])).as_posix()
    assert remote_tool_path and not PurePosixPath(remote_tool_path).is_absolute()
    assert '..' not in PurePosixPath(remote_tool_path).parts
    chunks=destination/'chunks';chunks.mkdir(exist_ok=True)
    local_zip=destination/'patient-complete.zip'
    if not local_zip.exists():
        offset=0
        while offset<remote_zip['bytes']:
            wanted=min(args.chunk_bytes,remote_zip['bytes']-offset)
            chunk=chunks/(str(offset)+'.bin');meta=chunks/(str(offset)+'.json')
            if chunk.exists():
                if meta.exists():
                    item=read(meta);assert item['archive_sha256']==remote_zip['sha256'] and item['offset']==offset
                    assert ref(chunk)==item['local']
                else:
                    # A crash after fsync but before the receipt leaves original bytes.
                    save(meta,dict(archive_sha256=remote_zip['sha256'],offset=offset,
                        local=ref(chunk),recovered_fsynced_chunk_receipt=True),root)
                size=chunk.stat().st_size;assert 0<size<=262144 and offset+size<=remote_zip['bytes']
            else:
                value=call('read_file',dict(mount_id=plan['mount'],path=remote_tool_path,offset=offset,limit=wanted))
                assert value['encoding']=='base64', 'Binary ZIP requires base64 transport'
                data=base64.b64decode(value['content'],validate=True);assert len(data)==wanted
                publish(chunk,data,root);size=len(data)
                save(meta,dict(archive_sha256=remote_zip['sha256'],offset=offset,requested_bytes=wanted,local=ref(chunk)),root)
            offset+=size
            print(json.dumps(dict(downloaded_bytes=offset,total_bytes=remote_zip['bytes'])),flush=True)
        temporary=destination/('local-assembly-'+uuid.uuid4().hex+'.part')
        with temporary.open('xb') as stream:
            offset=0
            while offset<remote_zip['bytes']:
                data=(chunks/(str(offset)+'.bin')).read_bytes();stream.write(data);offset+=len(data)
            stream.flush();os.fsync(stream.fileno())
        assert temporary.stat().st_size==remote_zip['bytes'] and sha(temporary)==remote_zip['sha256']
        os.link(temporary,local_zip);temporary.unlink();sync_chain(destination,root)
    assert local_zip.stat().st_size==remote_zip['bytes'] and sha(local_zip)==remote_zip['sha256']
    local=root/'inference-collected';local.mkdir(exist_ok=True)
    with zipfile.ZipFile(local_zip) as archive:
        names=archive.namelist();assert len(names)==len(set(names))
        raw_index=archive.read('member-index.json')
        assert hashlib.sha256(raw_index).hexdigest()==actual['member_index_sha256']
        index=json.loads(raw_index);files=index['files']
        assert index['plan_sha256']==cfg['plan_sha256'] and index['rows']==420
        assert set(names)==set(files)|{'member-index.json'}
        for name,item in files.items():
            relative=safe_name(name);info=archive.getinfo(name)
            assert not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16) and info.file_size==item['bytes']
            target=local/relative;assert not target.is_symlink()
            assert target.resolve().is_relative_to(local.resolve())
            if not target.exists():
                data=archive.read(name)
                assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
                publish(target,data,root)
            assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
            with target.open('rb') as stream:os.fsync(stream.fileno())
            sync_chain(target.parent,root)
        internal=local/'member-index.json'
        if not internal.exists():publish(internal,raw_index,root)
        else:assert internal.read_bytes()==raw_index
    known=set(files)|{'member-index.json'}
    for path in local.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():assert path.relative_to(local).as_posix() in known or path.name.endswith('.pending')
    wrapper=read(local/'control/wrapper-complete.json');manifest=read(local/'prediction/responses.manifest.json')
    assert wrapper['returncode']==0 and wrapper['rows']==420 and wrapper['complete_coverage'] and not wrapper['timed_out']
    assert 'endpoint_error' not in wrapper and sha(local/'prediction/responses.manifest.json')==wrapper['manifest_sha256']
    assert sha(local/'prediction/responses.jsonl')==manifest['output_sha256']==wrapper['responses_sha256']
    report=dict(schema='patient-B-complete-local-collection-v1',local=str(local),files=files,
        archive=ref(local_zip),remote_archive=remote_zip,remote_mapping=index['remote_mapping'],
        driver_exit_before_packaging=index['driver_exit'],
        inference_plan=ref(plan_path),collector_source=ref(__file__),client_source=ref(client_path),
        member_index=ref(internal),rows=420,documents=100,patients=244,
        complete_original_recursive_control_and_prediction=True,all_members_SHA_and_size_verified=True,
        files_and_ancestor_directories_fsynced=True,zip_safety_checked=True,
        no_raw_reconstruction=True,no_NN_or_retry_or_score=True,wrapper=wrapper)
    final=root/'inference-collection.json'
    if final.exists():assert read(final)==report
    else:save(final,report,root)
    print(json.dumps(dict(collection=ref(final),members=len(files),rows=420)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if _RUN_ROOT is not None:
            failure_path=_RUN_ROOT/'inference-collection'/'receipts'/('collector-failure-'+uuid.uuid4().hex+'.json')
            save(failure_path,dict(at_utc=now(),type=type(error).__name__,message=redacted(str(error)),
                partial_original_bytes_preserved=True,automatic_retry=False,NN_or_restart=False),_RUN_ROOT)
        raise
