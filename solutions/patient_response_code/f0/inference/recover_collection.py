"""Read-only continuation of an ambiguous original terminal-zip collection.

Never generates a model answer or rewrites the remote archive. The original
collector's exact archive contract is reused after requiring its zip to exist.
"""
from pathlib import Path
import argparse, ast, base64, hashlib, io, json, os, shlex, sys, time, zipfile
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

def digest(b): return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--directory',type=Path,required=True); a=ap.parse_args(); B=a.directory
 assert (B/'collect-intent.json').is_file() and not (B/'terminal-manifest.json').exists()
 plan=json.loads((B/'plan.json').read_text()); root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+plan['remote']
 source=Path(__file__).with_name('operations.py'); tree=ast.parse(source.read_text())
 texts=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
 archive_code=next(s for s in texts if "target=p/'collected-terminal.zip'" in s)
 start=archive_code.index('if not target.exists():'); end=archive_code.index('with zipfile.ZipFile(target) as z:',start)
 archive_code=archive_code[:start]+"assert target.is_file(), 'Original archive absent; stop without recreating ambiguous mutation'\n"+archive_code[end:]
 part_code=next(s for s in texts if "with Path(PARAM['path']).open('rb') as f:f.seek" in s)
 compile(archive_code,'read-only-manifest','exec'); compile(part_code,'read-only-part','exec')
 stamp=str(time.time_ns()); R=B/('collect-recovery-'+stamp); R.mkdir()
 def save(path,x):
  with path.open('x') as f: json.dump(x,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 save(R/'intent.json',dict(at_unix=time.time(),source_sha256=digest(Path(__file__).read_bytes()),original_source_sha256=digest(source.read_bytes()),remote=root,remote_read_only=True,original_intent_sha256=digest((B/'collect-intent.json').read_bytes())))
 client=Client(); client.initialize()
 def remote(code,params,label):
  for attempt in range(2):
   r=client.request('tools/call',{'name':'run_command','arguments':{'mount_id':plan['mount_id'],'command':'python3 -c '+shlex.quote('PARAM='+repr(params)+'\n'+code),'timeout_seconds':30}})
   save(R/(label+'-'+str(attempt)+'-tool.json'),r)
   if r.get('isError'):
    s=json.dumps(r)
    if 'UNKNOWN_OUTCOME' in s and attempt==0: time.sleep(30); continue
    raise RuntimeError('Read-only collection unavailable; preserve all original bytes and respect explicit LUNA cooldown')
   out=r.get('structuredContent') or json.loads(r['content'][0]['text']); assert out['exit_code']==0,out.get('stderr'); return json.loads(out['stdout'])
  raise AssertionError('unreachable')
 if (B/'collect-remote-manifest.json').exists(): metadata=json.loads((B/'collect-remote-manifest.json').read_text())
 else:
  metadata=remote(archive_code,{'root':root},'manifest'); save(B/'collect-remote-manifest.json',metadata)
 pieces=[]
 for offset in range(0,metadata['zip_bytes'],32768):
  size=min(32768,metadata['zip_bytes']-offset); path=B/f'collect-part-{offset:09d}.bin'; receipt=B/f'collect-part-{offset:09d}-recovered.json'
  if path.exists():
   if not receipt.exists():
    original=B/f'collect-part-{offset:09d}-tool.json'; tool=json.loads(original.read_text()); assert not tool.get('isError')
    result=tool.get('structuredContent') or json.loads(tool['content'][0]['text']); assert result['exit_code']==0
    prior=json.loads(result['stdout']); data=path.read_bytes(); assert data==base64.b64decode(prior['b64'])
    assert len(data)==size==prior['bytes'] and digest(data)==prior['sha256']
    save(receipt,dict(bytes=len(data),sha256=digest(data),original_successful_tool_sha256=digest(original.read_bytes()),original_collector_piece_reused=True))
   ref=json.loads(receipt.read_text()); data=path.read_bytes(); assert len(data)==size==ref['bytes'] and digest(data)==ref['sha256']
  else:
   r=remote(part_code,{'path':metadata['remote_archive'],'offset':offset,'size':size},f'part-{offset:09d}'); data=base64.b64decode(r['b64']); assert len(data)==size==r['bytes'] and digest(data)==r['sha256']
   save(receipt,dict(bytes=len(data),sha256=digest(data)))
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  pieces.append(data)
 payload=b''.join(pieces); assert len(payload)==metadata['zip_bytes'] and digest(payload)==metadata['zip_sha256']
 def exact_write(path,data):
  path.parent.mkdir(exist_ok=True,parents=True)
  if path.exists(): assert path.read_bytes()==data
  else:
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 exact_write(B/'terminal.zip',payload); dest=B/'terminal'; expected={r['name']:r for r in metadata['files']};assert len(expected)==len(metadata['files'])
 with zipfile.ZipFile(io.BytesIO(payload)) as z:
  assert len(z.namelist())==len(expected) and set(z.namelist())==set(expected)
  for n,r in expected.items():
   p=Path(n); assert not p.is_absolute() and '..' not in p.parts
   assert len(p.parts)==1 or len(p.parts)==2 and p.parts[0] in ['generated-ids','raw-responses','canonical-records']
   data=z.read(n);assert len(data)==r['bytes'] and digest(data)==r['sha256'];exact_write(dest/p,data)
 m={'status':'ACTUAL_UNIQUE_FULL58_TERMINAL_COLLECTED',**metadata,'remote':root,'plan_sha256':digest((B/'plan.json').read_bytes()),'model_calls_repeated':False,'quality_calls':0,'collection_recovery':str(R),'original_archive_reused_no_repack':True}
 save(B/'terminal-manifest.json',m);save(R/'complete.json',dict(files=len(expected),zip_bytes=metadata['zip_bytes'],zip_sha256=metadata['zip_sha256'],remote_mutations=0,model_calls=0,quality_calls=0)); print(json.dumps({'status':m['status'],'files':len(expected),'zip_bytes':metadata['zip_bytes'],'zip_sha256':metadata['zip_sha256']}))
if __name__=='__main__': main()
