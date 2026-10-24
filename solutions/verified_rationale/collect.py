"""Collect one completed pilot, with full member hashes; never invokes a model."""
import argparse, base64, datetime, hashlib, importlib.util, io, json, os, shlex, tarfile
from pathlib import Path

REMOTE = r'''
import datetime,hashlib,io,json,os,pathlib,subprocess,tarfile
p=pathlib.Path('.'); done=json.loads((p/'driver-complete.json').read_text());result=json.loads((p/'terminal/result.json').read_text())
assert done['returncode']==0 and not done['TERM_sent'] and not done['KILL_sent']
assert result['status']=='COMPLETE_FIXED15_TEACHER_RESPONSES_SEMANTIC_REVIEW_PENDING' and len(result['rows'])==15
assert not pathlib.Path('/proc/'+str(done['child_pid'])).exists(),'child still exists'
files=sorted(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and not f.name.startswith('collection.'))
items={str(f):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}
inv=json.dumps({'files':items,'payload_count':len(items)},indent=2).encode()+b'\n'
target=p/'collection.tar.gz'
assert not target.exists(),'Existing collection must be recovered, not overwritten'
with tarfile.open(target,'w:gz') as t:
 for f in files:t.add(f,arcname=str(f),recursive=False)
 ti=tarfile.TarInfo('inventory.json');ti.size=len(inv);t.addfile(ti,io.BytesIO(inv))
with target.open('rb') as f:os.fsync(f.fileno())
b=target.read_bytes();receipt={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':{'name':str(target),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'payload_count':len(items),'members':len(items)+1,'fsync_reopened':True,'driver':done,'child_absent':True,'gpu':subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout,'apps':subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout}
with (p/'collection.receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(receipt))
'''

def save(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

def ref(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--bind-dir',required=True);p.add_argument('--mcp-client',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);e=json.loads((Path(a.bind_dir)/'execution.json').read_text());remote=Path(e['references']['inputs_gz']['path']).parent;rel=str(remote.relative_to('/home/zhangyuheng/projects/CHIP2026_PatientPheX'));mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
    s=importlib.util.spec_from_file_location('collect_mcp',a.mcp_client);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);c=m.Client();c.initialize()
    def call(tool,args,label):
        r=c.request('tools/call',{'name':tool,'arguments':args});save(out/(label+'.json'),r)
        assert not r.get('isError'),'Original request outcome saved; recover original collection after UNKNOWN'
        return r.get('structuredContent') or json.loads(r['content'][0]['text'])
    save(out/'intent.json',{'source':ref(__file__),'remote':str(remote),'command_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'no_GPU_no_quality':True})
    r=call('run_command',{'mount_id':mount,'command':shlex.quote(e['python'])+' -c '+shlex.quote(REMOTE),'cwd':rel,'env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':30},'assembly-response');assert r['exit_code']==0
    receipt=json.loads(r['stdout']);save(out/'remote-receipt.json',receipt);chunks=[]
    for offset in range(0,receipt['archive']['bytes'],524288):
        x=call('read_file',{'mount_id':mount,'path':rel+'/'+receipt['archive']['name'],'offset':offset,'limit':min(524288,receipt['archive']['bytes']-offset)},'chunk-'+str(offset));b=base64.b64decode(x['content']) if x['encoding']=='base64' else x['content'].encode();chunks.append(b)
    archive=b''.join(chunks);assert len(archive)==receipt['archive']['bytes'] and hashlib.sha256(archive).hexdigest()==receipt['archive']['sha256'];ap=out/'collection.tar.gz'
    with ap.open('xb') as f:f.write(archive);f.flush();os.fsync(f.fileno())
    payload=out/'payload';payload.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        members=t.getmembers();names=[x.name for x in members];assert len(names)==len(set(names))==receipt['members'];inv=json.load(t.extractfile('inventory.json'));assert set(names)==set(inv['files'])|{'inventory.json'}
        for name in names:
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            info=t.getmember(name);assert info.isfile();b=t.extractfile(info).read()
            if name!='inventory.json':assert {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}==inv['files'][name]
            target=payload/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    save(out/'complete.json',{'status':'FULL_COMPLETED_PILOT_COLLECTION_ALL_MEMBERS_SHA_VERIFIED','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':ref(ap),'remote_receipt':ref(out/'remote-receipt.json'),'members':receipt['members'],'payload_count':receipt['payload_count'],'inventory':ref(payload/'inventory.json'),'source':ref(__file__),'GPU_calls':0,'quality_calls':0})
    print(json.dumps({'status':'COMPLETE','members':receipt['members'],'bytes':len(archive)}))

if __name__=='__main__':main()
