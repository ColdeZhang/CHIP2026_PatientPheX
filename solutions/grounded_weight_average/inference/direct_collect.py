"""Direct original-file recovery; byte-identical row reconstruction, never model/decode.

Two read-only directory inventories returned UNKNOWN. No remote archive is claimed.
"""
import base64,hashlib,json,os,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-weight-average-v1/F1/infer-v1');W=D/'direct-recovery';T=D/'terminal'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/grounded-weight-average-v1/F1/infer-v1';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
sys.path.insert(0,str(S.parent));from synthesize import ref
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client

def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 q=p.parent
 while True:
  fd=os.open(q,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if q.parent==q:break
  q=q.parent
 assert p.read_bytes()==b

def save(p,v):put(p,(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def same(raw,item):return len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
def main():
 c=Client();c.initialize();receipts=W/'receipts';receipts.mkdir(exist_ok=True)
 def fetch(name,expected=None,wanted_sha=None):
  target=T/name
  if target.exists():
   b=target.read_bytes()
   if expected:assert same(b,expected)
   if wanted_sha:assert hashlib.sha256(b).hexdigest()==wanted_sha
   return ref(target)
  parts=[];offset=0;limit=131072
  while True:
   args=dict(mount_id=MOUNT,path=Path(ROOT+'/'+name).relative_to(P).as_posix(),offset=offset,limit=min(limit,expected['bytes']-offset) if expected else limit)
   stem=receipts/str(time.time_ns());save(stem.with_suffix('.intent.json'),dict(tool='read_file',arguments=args))
   r=c.request('tools/call',dict(name='read_file',arguments=args));save(stem.with_suffix('.response.json'),r)
   assert not r.get('isError'),'Original direct-read response preserved; no automatic retry'
   v=r.get('structuredContent') or json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
   b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode();parts.append(b);offset+=len(b)
   if expected:
    assert b and offset<=expected['bytes']
    if offset==expected['bytes']:break
   else:
    assert len(b)<limit,'Small ancillary file unexpectedly large; do not silently truncate';break
  raw=b''.join(parts)
  if expected:assert same(raw,expected)
  if wanted_sha:assert hashlib.sha256(raw).hexdigest()==wanted_sha
  put(target,raw);return ref(target)
 complete=json.loads((W/'complete.json').read_text());assert ref(W/'complete.json')['sha256']=='41c4bb5bd77c04c71fdd7d99c112e9c9f76c7408c283ef4bad5ab5ce65bd9175'
 put(T/'output/complete.json',(W/'complete.json').read_bytes());fetch('output/responses.jsonl',complete['responses'])
 records=[json.loads(l) for l in (T/'output/responses.jsonl').read_text().splitlines()];assert len(records)==39
 rawrefs={Path(x['path']).name:x for x in complete['raw_refs']};genrefs={Path(x['path']).name:x for x in complete['generation_refs']};rebuilt=[]
 for record in records:
  name=hashlib.sha256(record['row_id'].encode()).hexdigest()+'.json';assert record['generation']==genrefs[name]
  gen={k:record[k] for k in ['row_id','pmc_id','target_patient_id','generated_token_ids','input_tokens','output_tokens','actual_last_token_eos','hit_token_limit','seconds']};gen['decoded_or_parsed']=False
  for kind,obj,item in [('raw',record,rawrefs[name]),('generation',gen,genrefs[name])]:
   raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();assert same(raw,item)
   target=T/'output'/kind/name;assert item['path']==ROOT+'/'+target.relative_to(T).as_posix();put(target,raw);rebuilt.append(dict(local=ref(target),remote_original=item,mode='byte-identical JSON serialization from original responses; no token decode or model call'))
 assert len(rebuilt)==78
 latest=json.loads(sorted(D.glob('observation-*.json'))[-1].read_text());assert latest['matching_pids']==[]
 for name in ['driver-started.json','child-started.json','wrapper-complete.json']:
  item=latest[name];raw=(json.dumps(item['value'],indent=2)+'\n').encode();assert same(raw,item);put(T/name,raw)
 wrapper=json.loads((T/'wrapper-complete.json').read_text());assert wrapper['returncode']==0 and not wrapper['timed_out'] and wrapper['complete39_actual_members_verified']
 assert wrapper['complete_sha256']==ref(T/'output/complete.json')['sha256']
 for name in ['fresh-resource-gate.json','stdout.log','stderr.log','output/loading.json','output/run-manifest.json']:
  fetch(name,wanted_sha='6c9fdebe1df7e5a2af5f93aee7dafa0aea7f840774f4d6840173431792f82d2e' if name.endswith('run-manifest.json') else None)
 files={p.relative_to(T).as_posix():ref(p) for p in sorted(T.rglob('*')) if p.is_file()}
 proof=W/'byteexact-reconstruction.json';save(proof,dict(original_complete=ref(T/'output/complete.json'),original_responses=ref(T/'output/responses.jsonl'),members=rebuilt,source=ref(__file__),no_model_token_decode_or_parser=True))
 col=dict(schema='grounded-held39-complete-collection-v1',collection_mode='direct_original_files_and_byteexact_row_reconstruction',local=str(T),files=files,source=ref(__file__),original_remote=ROOT,mount=MOUNT,package=ref(D/'package.json'),wrapper=wrapper,child_exit_before_freeze=dict(no_running_process=True,original_observation=ref(sorted(D.glob('observation-*.json'))[-1])),inference_reported_success=True,all_actual_SHA_verified=True,all39_raw_and_ID_bytes_verified=True,all_original_output_files_included=False,files_and_ancestor_directories_fsynced=True,remote_archive_frozen=False,archive=None,remote_archive=None,reconstruction=ref(proof),scope='All39 manifest-listed originals/IDs plus required controller/loading/run metadata. No remote directory enumeration/archive assertion.',collection_unix=time.time())
 save(D/'collection.json',col);print(json.dumps(dict(collection=ref(D/'collection.json'),files=len(files),byteexact_rows=len(rebuilt),EOS_rows=complete['EOS_rows'],cap_rows=complete['cap_rows'])))
if __name__=='__main__':main()
