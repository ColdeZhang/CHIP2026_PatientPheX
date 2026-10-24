"""Back up the completed small quality bundle to the original durable server."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import sys
import zipfile
from common import ref,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());quality=Path(plan['output']);result=json.loads((quality/'evaluation/result.json').read_text())
    assert result['status']=='COMPLETE_UNIQUE_FIXED_UNION_F1' and result['new_evaluator_calls']==1 and result['single_head_evaluator_calls']==0
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);files={}
    for p in sorted(quality.rglob('*')):
        if p.is_file():assert p.stat().st_size<2*1048576;files['quality/'+str(p.relative_to(quality))]=p
    files['quality-plan.json']=Path(a.plan);files['canonical.jsonl']=Path(plan['actual']['canonical']['path'])
    for p in Path(__file__).parent.glob('*.py'):files['source/'+p.name]=p
    archive=out/'quality.zip';index={name:ref(p) for name,p in files.items()}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,p in files.items():z.write(p,name)
        z.writestr('member-index.json',json.dumps(index,indent=2)+'\n')
    wanted=ref(archive);relative='research/canonical-names-v1/qwen3-candidate-v1/quality-F1-v1/quality.zip'
    mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c';project='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(name,args,tag):
        safe={k:v for k,v in args.items() if k!='content'};write(out/(tag+'.intent.json'),dict(tool=name,args=safe,archive=wanted))
        r=client.request('tools/call',dict(name=name,arguments=args));write(out/(tag+'.response.json'),r)
        assert not r.get('isError'),r
        return r.get('structuredContent') or json.loads(next(v['text'] for v in r['content'] if v['type']=='text'))
    call('write_file',dict(mount_id=mount,path=relative,content=base64.b64encode(archive.read_bytes()).decode(),encoding='base64',create_parents=True),'write-once')
    code="""from pathlib import Path
import hashlib,json,os,zipfile
p=Path(PATH);raw=p.read_bytes();assert len(raw)==SIZE and hashlib.sha256(raw).hexdigest()==SHA
with zipfile.ZipFile(p) as z:
 idx=json.loads(z.read('member-index.json'));assert set(z.namelist())==set(idx)|{'member-index.json'}
 for n,r in idx.items():
  b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
with p.open('rb') as h:os.fsync(h.fileno())
fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
print(json.dumps(dict(status='DURABLE_QUALITY_ARCHIVE_ALL_MEMBERS_SHA_FSYNC_REOPEN',path=str(p),bytes=SIZE,sha256=SHA,members=len(idx))))
"""
    code='PATH='+repr(project+'/'+relative)+'\nSIZE='+repr(wanted['bytes'])+'\nSHA='+repr(wanted['sha256'])+'\n'+code
    response=call('run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=30),'verify-once')
    assert response['exit_code']==0,response.get('stderr');ack=json.loads(response['stdout']);write(out/'ack.json',dict(local=wanted,remote=ack))
    print(json.dumps(ref(out/'ack.json')))

if __name__=='__main__':main()
