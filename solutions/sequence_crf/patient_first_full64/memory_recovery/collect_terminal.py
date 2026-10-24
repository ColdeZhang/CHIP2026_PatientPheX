"""Collect fixed small terminal members compressed; never read model state bytes."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--phase',choices=['train','infer'],required=True)
    args=ap.parse_args();b=args.directory;phase=args.phase
    resource=json.loads((b/'resource.json').read_text());plan=json.loads((b/(phase+'-plan.json')).read_text())
    name='training-collection.json' if phase=='train' else 'inference-collection.json'
    target=b/('training-terminal' if phase=='train' else 'inference-terminal')
    assert not(b/name).exists() and not target.exists()
    config=dict(control=resource['control_root']+'/'+phase,output=plan['output_dir'],phase=phase)
    code='CFG='+repr(config)+'''
from pathlib import Path
import json,hashlib,base64,zipfile,io
control=Path(CFG['control']);out=Path(CFG['output']);phase=CFG['phase']
assert (control/'wrapper-complete.json').exists(), 'Original job not terminal; collect only after terminal observation'
paths={n:control/n for n in ['driver-started.json','wrapper-complete.json','stdout.log','stderr.log']}
names=['run_manifest.json','token-mask-audit.json','gpu-preflight.json','model-coverage.json','base-loading-info.json','first-nonzero-update.json','progress.json','training_complete.json','selected-loss-binding.json'] if phase=='train' else ['responses.jsonl','responses.manifest.json','responses.checkpoint.json','responses.loading.json','responses.decisions.json','responses.raw-index.json','generation-failure.json','complete.json']
paths.update({n:out/n for n in names if(out/n).exists()})
if phase=='train':
 paths.update({str(p.relative_to(out)):p for p in out.glob('epoch-*/checkpoint.json') if p.parent.is_dir()})
 paths.update({p.name:p for p in out.glob('failure-*.json') if p.is_file()})
 paths.update({p.name:p for p in out.glob('durability-epoch-[0-9][0-9][0-9].json') if p.is_file()})
else:
 for directory in ['raw-rows','decision-rows','generation-rows']:
  paths.update({str(p.relative_to(out)):p for p in (out/directory).glob('*.json') if p.is_file()})
assert all(p.suffix not in ['.pt','.safetensors','.bin'] for p in paths.values())
files={};buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for n,p in paths.items():
  raw=p.read_bytes();assert len(raw)<1024*1024
  files[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());z.writestr(n,raw)
raw=buffer.getvalue();assert len(raw)<=512*1024
print(json.dumps(dict(files=files,archive=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))))
'''
    compile(code,'fixed-small-terminal-collection','exec')
    save(b/(phase+'-compressed-collection-intent.json'),dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_only=True,unknown_outcome='Only collection, no remote mutation or launch; retain receipt and recover same files'))
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize();result=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id=resource['mount'],command='python3 -c '+shlex.quote(code),cwd='.',timeout_seconds=45)))
    save(b/(phase+'-compressed-collection-execution.json'),result);assert not result.get('isError')
    actual=result['structuredContent'];assert actual['exit_code']==0,actual.get('stderr')
    payload=json.loads(actual['stdout']);archive=payload['archive'];raw=base64.b64decode(archive['base64'])
    assert len(raw)==archive['bytes'] and hashlib.sha256(raw).hexdigest()==archive['sha256']
    with(b/(phase+'-terminal.zip')).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    target.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw))as z:
        assert set(z.namelist())==set(payload['files'])
        for n,item in payload['files'].items():
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            data=z.read(n);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            path=target/n;path.parent.mkdir(exist_ok=True,parents=True)
            with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    wrapper=json.loads((target/'wrapper-complete.json').read_text())
    save(b/name,dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,files=payload['files'],wrapper=wrapper,direct_existing_files_only=True,compressed_transport=True,archive={k:v for k,v in archive.items()if k!='base64'}))
    print(json.dumps(dict(phase=phase,members=len(payload['files']),archive_bytes=len(raw),wrapper_returncode=wrapper['returncode'],collection=str(b/name))))


if __name__=='__main__':main()
