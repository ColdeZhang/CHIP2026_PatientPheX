from pathlib import Path
import sys,json,base64,hashlib,time,zipfile,os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
D=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/patient-competition-calibration-v1';ready=json.loads((D/'terminal-ready.json').read_text());parts=D/'terminal-parts';parts.mkdir(exist_ok=True)
def get(offset):
 size=min(524288,ready['bytes']-offset);p=parts/(str(offset)+'.bin')
 if p.exists():assert p.stat().st_size==size;return
 c=Client();c.initialize();v=c.request('tools/call',{'name':'read_file','arguments':{'mount_id':M,'path':rel+'/terminal-evidence.zip','offset':offset,'limit':size}});(D/('terminal-read-'+str(offset)+'-'+str(time.time_ns())+'.json')).write_text(json.dumps(v,indent=2));assert not v.get('isError'),v;x=v.get('structuredContent') or json.loads(v['content'][0]['text']);b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)==size
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(get,range(0,ready['bytes'],524288)))
b=b''.join((parts/(str(i)+'.bin')).read_bytes() for i in range(0,ready['bytes'],524288));assert len(b)==ready['bytes'] and hashlib.sha256(b).hexdigest()==ready['sha256'];p=D/'terminal-evidence.zip';p.write_bytes(b);members={}
with zipfile.ZipFile(p) as z:
 inv=json.loads(z.read('terminal-inventory.json'));assert len(inv)==ready['members']
 for name,r in inv.items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts;b=z.read(name);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];q=D/'collected'/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256'];members[name]={'path':str(q),'bytes':len(b),'sha256':r['sha256']}
(D/'collection-receipt.json').write_text(json.dumps({'status':'ALL_ORIGINAL_TERMINAL_MEMBERS_BYTES_SHA_PASS','archive':ready,'members':members,'new_model_or_quality':False},indent=2)+'\n');print(json.dumps({'members':len(members),'archive_bytes':ready['bytes'],'archive_sha256':ready['sha256']}))
