"""Collect only completed immutable original full80 bundle in resumable128KiB read chunks."""
from pathlib import Path
import sys,json,time,hashlib,base64,shlex,os,gzip
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
from poll_original import main as poll,M
F=Path(__file__).parent

def sha(b):return hashlib.sha256(b).hexdigest()
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 assert poll(),'Original process not terminal; this collector never starts/stops it'
 result=json.loads((F/'recovered/terminal/result.json').read_text());wrapper=json.loads((F/'recovered/wrapper-result.json').read_text())
 assert result['status']=='COMPLETE_FIXED80_RESPONSES_2_REUSED_78_NEW_NO_QUALITY_EVALUATION' and wrapper['returncode']==0
 assert result['new_generations']==78 and result['reused_generations']==2 and len(result['rows'])==80
 r=result['response_bundle'];dest=F/'recovered/terminal/responses.raw.jsonl.gz';parts=F/'collection-parts';parts.mkdir(exist_ok=True)
 c=Client();c.initialize();records=[]
 for start in range(0,r['bytes'],131072):
  count=min(131072,r['bytes']-start);p=parts/('%09d.bin'%start);m=p.with_suffix('.json')
  if p.exists() and m.exists():
   e=json.loads(m.read_text());assert e['source_sha256']==r['sha256'] and e['offset']==start and e['bytes']==count and len(p.read_bytes())==count and sha(p.read_bytes())==e['sha256'];records.append(e);continue
  assert not p.exists() and not m.exists(),'Unknown local partial chunk; inspect rather than overwrite'
  code="from pathlib import Path;import os,json,hashlib,base64;p=Path("+repr(r['path'])+");assert p.stat().st_size=="+str(r['bytes'])+";f=p.open('rb');f.seek("+str(start)+");b=f.read("+str(count)+");assert len(b)=="+str(count)+";print(json.dumps({'offset':"+str(start)+",'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}))"
  raw=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)})
  if raw.get('isError'):(F/'collection-error.json').write_text(json.dumps({'at':time.time(),'offset':start,'tool':raw},indent=2)+'\n');raise RuntimeError('Read interrupted; verified chunks retained')
  v=un(raw);assert v['exit_code']==0;e=json.loads(v['stdout']);data=base64.b64decode(e.pop('base64'));assert len(data)==count and sha(data)==e['sha256'];e['source_sha256']=r['sha256']
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  with m.open('x') as f:json.dump(e,f);f.write('\n');f.flush();os.fsync(f.fileno())
  records.append(e)
 data=b''.join((parts/('%09d.bin'%x['offset'])).read_bytes() for x in records);assert len(data)==r['bytes'] and sha(data)==r['sha256'];unpacked=gzip.decompress(data);assert sha(unpacked)==result['response_uncompressed_sha256']
 rows=[json.loads(x) for x in unpacked.decode().splitlines()];assert len(rows)==80
 for i,(item,rr) in enumerate(zip(rows,result['rows'])):
  assert item['original_index']==rr['original_index']==i and item['row_id']==rr['row_id']
  assert sha(json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())==rr['canonical_item_sha256']
 if dest.exists():assert dest.read_bytes()==data
 else:
  with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 manifest={'status':'COMPLETE_RESPONSE_BUNDLE_FULL_SHA_GZIP_AND_ALL80_CANONICAL_BINDINGS','source':r,'local':{'path':str(dest),'bytes':len(data),'sha256':sha(data)},'rows':80,'new':78,'reused':2,'read_chunks':records,'no_NN_gold_or_quality':True}
 (F/'response-collection.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='read_chunks'}))
if __name__=='__main__':main()
