"""Read/adopt an exact already-published final ACK after an unknown mutation."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controller',type=Path,required=True);a=ap.parse_args();C=a.controller
    plan=json.loads((C/'controller-plan.json').read_text());raw=(C/'epoch-002-export-ack.json').read_bytes();expected=hashlib.sha256(raw).hexdigest()
    assert not (C/'epoch-002-source-ack-copy.json').exists()
    code='''from pathlib import Path
import hashlib,json,os
p=Path(P['path'])
if not p.exists():print(json.dumps(dict(exists=False,path=str(p))));raise SystemExit(0)
assert p.is_file() and not p.is_symlink();b=p.read_bytes();assert len(b)==P['bytes'] and hashlib.sha256(b).hexdigest()==P['sha256']
with p.open('rb') as f:os.fsync(f.fileno())
fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(exists=True,path=str(p),bytes=len(b),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),preexisting_exact=True,no_content_write=True)))
'''
    params=dict(path=plan['output_dir']+'/export-ack-epoch-002.json',bytes=len(raw),sha256=expected)
    client=Client();client.initialize();r=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id=plan['mount_id'],command='python3 -c '+shlex.quote('P='+repr(params)+'\n'+code),timeout_seconds=20)))
    with (C/('final-ack-inspect-'+str(time.time_ns())+'-tool.json')).open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if r.get('isError'):raise RuntimeError('Original source ACK read unknown; do not infer mutation result')
    v=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert v['exit_code']==0;proof=json.loads(v['stdout'])
    if proof['exists']:
        assert proof['sha256']==expected
        with (C/'epoch-002-source-ack-copy.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(proof))


if __name__=='__main__':main()
