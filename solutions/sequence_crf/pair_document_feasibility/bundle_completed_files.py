#!/usr/bin/env python3
"""Recover already-hashed small endpoint files in one archive; reuse prior bytes."""
import argparse,base64,datetime,hashlib,io,json,shlex,sys,tarfile,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;r=json.loads((b/'resource.json').read_text())
    assert not (b/'collection.json').exists()
    actual=json.loads((b/'actual-artifact-recovery.json').read_text());assert actual['result']['exit_code']==0
    expected=json.loads(actual['result']['stdout']);assert len(expected)==9 and all(x['name'].endswith('.json') for x in expected)
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    c=Client();c.initialize()
    def call(name,**kwargs):
        value=c.request('tools/call',{'name':name,'arguments':{'mount_id':r['mount'],**kwargs}});assert not value.get('isError'),value;return value['structuredContent']
    code='''from pathlib import Path
import hashlib,json,tarfile,io
expected=EXPECTED
target=Path('endpoint-small-recovery.tar.gz');assert not target.exists()
with tarfile.open(target,'w:gz') as t:
 for item in expected:
  p=Path(item['name']);assert p.is_file() and not p.is_symlink();raw=p.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
  info=tarfile.TarInfo(item['name']);info.size=len(raw);info.mode=0o600;t.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(name=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),members=expected)))
'''.replace('EXPECTED',repr(expected))
    started=time.monotonic();receipt=call('run_command',command=r['python']+' -c '+shlex.quote(code),cwd=r['remote'],env=r['CPU_env'],timeout_seconds=30);assert receipt['exit_code']==0
    manifest=json.loads(receipt['stdout']);(b/'small-recovery-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    parts=[]
    for offset in range(0,manifest['bytes'],512*1024):
        value=call('read_file',path=r['remote']+'/'+manifest['name'],offset=offset,limit=min(512*1024,manifest['bytes']-offset));parts.append(base64.b64decode(value['content']) if value['encoding']=='base64' else value['content'].encode())
    raw=b''.join(parts);assert len(raw)==manifest['bytes'] and hashlib.sha256(raw).hexdigest()==manifest['sha256'];(b/manifest['name']).write_bytes(raw)
    dest=b/'terminal';dest.mkdir(exist_ok=True);byname={x['name']:x for x in expected};reused=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        members=t.getmembers();assert len(members)==len(expected)==len({x.name for x in members})
        for entry in members:
            assert entry.isfile() and entry.name in byname and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
            payload=t.extractfile(entry).read();ref=byname[entry.name];assert len(payload)==ref['bytes'] and hashlib.sha256(payload).hexdigest()==ref['sha256']
            p=dest/entry.name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==payload;reused.append(entry.name)
            else:p.write_bytes(payload)
    record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'items':expected,'remote':r,'archive':manifest,'reused_existing_files':reused,'seconds':time.monotonic()-started,'method':'Bundle of exact existing small files only after original PID/GPU/final-state SHA verification; no training or large checkpoint rehash; managed exitcode unavailable.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_and_optimizer_states_retained_remote':True}
    (b/'collection.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'complete':True,'members':len(expected),'archive_bytes':len(raw),'reused_files':len(reused),'seconds':record['seconds']}))

if __name__=='__main__':main()
