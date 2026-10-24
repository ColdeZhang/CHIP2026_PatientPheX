"""Bounded CPU-only original-file read fallback, at most32768 bytes per RPC.
No remote file mutation/service/GPU; never regenerates or repairs responses.
"""
import base64,hashlib,json,os,shlex,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));from direct_collect import put,save,ref
D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-weight-average-v1/F0/infer-v1');W=D/'direct-recovery';C=W/'command-read'
M='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def main():
 backoffs=sorted(W.glob('offline-backoff-*.json'))
 if backoffs:assert time.time()>=json.loads(backoffs[-1].read_text())['not_before_unix']
 expected=json.loads((D/'terminal/output/complete.json').read_text())['responses'];assert expected['path'].endswith('/grounded-weight-average-v1/F0/infer-v1/output/responses.jsonl') and expected['bytes']<2_000_000
 C.mkdir(exist_ok=True);sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
 c=Client();c.initialize();chunks=[]
 for offset in range(0,expected['bytes'],32768):
  target=C/('%08d.bin'%offset);wanted=min(32768,expected['bytes']-offset)
  if target.exists():assert target.stat().st_size==wanted;chunks.append(target.read_bytes());continue
  code='from pathlib import Path\nimport json,base64,hashlib\np=Path('+repr(expected['path'])+');b=p.read_bytes();assert len(b)=='+str(expected['bytes'])+' and hashlib.sha256(b).hexdigest()=='+repr(expected['sha256'])+'\nx=b['+str(offset)+':'+str(offset+wanted)+'];print(json.dumps(dict(offset='+str(offset)+',bytes=len(x),sha256=hashlib.sha256(x).hexdigest(),content=base64.b64encode(x).decode())))'
  args=dict(mount_id=M,command='python3 -c '+shlex.quote(code),timeout_seconds=30);stem=C/str(time.time_ns());save(stem.with_suffix('.intent.json'),dict(tool='run_command',args=args,read_only=True))
  r=c.request('tools/call',dict(name='run_command',arguments=args));save(stem.with_suffix('.response.json'),r)
  if r.get('isError'):
   if 'LUNA_OFFLINE' in json.dumps(r):save(W/('offline-backoff-'+str(time.time_ns())+'.json'),dict(not_before_unix=time.time()+185,error_receipt=str(stem.with_suffix('.response.json')),scope='original-file CPU read fallback only; original model complete',no_GPU_restart=True))
   raise RuntimeError('Original read-only fallback receipt preserved; no immediate retry')
  v=r.get('structuredContent') or json.loads(next(x['text'] for x in r['content'] if x['type']=='text'));assert v['exit_code']==0;v=json.loads(v['stdout']);b=base64.b64decode(v['content']);assert v['offset']==offset and len(b)==wanted==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];put(target,b);chunks.append(b)
 raw=b''.join(chunks);assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'];put(D/'terminal/output/responses.jsonl',raw)
 save(C/'complete.json',dict(status='COMPLETE_ORIGINAL_RAW_CPU_READ_ONLY',source=ref(__file__),original=expected,local=ref(D/'terminal/output/responses.jsonl'),chunk_bytes_max=32768,chunks=len(chunks),remote_mutation=False,model_or_decode=False));print(json.dumps(ref(C/'complete.json')))
if __name__=='__main__':main()
