"""Freeze new F1 source/evidence into one small durable server backup, no NN/quality."""
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
from pathlib import Path
from synthesize import ref,save
S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F1';B=D/'final-backup-v1'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';REL='research/grounded-weight-average/F1-final-backup-v1';MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'

def build():
 B.mkdir(exist_ok=False);result=json.loads((D/'quality-v1/result.json').read_text());freeze=json.loads((D/'quality-v1/freeze.json').read_text())
 named={
 'weight_CPU_collection':D/'cpu-v1/collection.json','weight_CPU_complete':D/'cpu-v1/collected/output/complete.json',
 'weight_actual_parents':D/'cpu-v1/collected/output/actual-parents.json','weight_tiny_PEFT':D/'cpu-v1/collected/output/tiny-peft.json',
 'inference_collection':D/'infer-v1/collection.json','inference_complete':D/'infer-v1/terminal/output/complete.json',
 'inference_wrapper':D/'infer-v1/terminal/wrapper-complete.json','loaded_model_manifest':D/'infer-v1/terminal/output/run-manifest.json',
 'byteexact_reconstruction':D/'infer-v1/direct-recovery/byteexact-reconstruction.json',
 'extraction':D/'quotes-v1/manifest.json','normalization_preparation':D/'normalization-preparation/manifest.json',
 'query_collection':D/'Sap-CPU100-v1/collection.json','query_complete':D/'Sap-CPU100-v1/terminal/output/complete.json',
 'query_runtime':D/'Sap-CPU100-v1/terminal/output/encoder-runtime.json','query_vector_locator':D/'Sap-CPU100-v1/terminal/member-index.json',
 'normalization':D/'normalized-v1/manifest.json','freeze':D/'quality-v1/freeze.json','evaluation_intent':D/'quality-v1/evaluation-intent.json',
 'result':D/'quality-v1/result.json','metrics':D/'quality-v1/metrics.json','format':D/'quality-v1/format.json','quality_bound_inputs':D/'quality-v1/bound-inputs.json',
 'primary_prediction':D/'quality-v1/prediction.jsonl','unscored_pure_mean':D/'quality-v1/mean-new-only.jsonl',
 'normalization_amendment':S/'normalization-amendment.json','actual_verifier':S/'verify_direct.py','quality_source':S/'postprocess.py'}
 entry=dict(status='COMPLETE_FIXED_F1_JOINT_WEIGHT_MEAN_CPU_QUERY_CONTROL',refs={k:ref(v) for k,v in named.items()},
  result=result,coverage=dict(documents=16,patients=39,entities=1189,raw_rows=39,EOS=35,cap=4,anchors=402,necessary_queries=174),
  mean_relations=189,intersection=156,added=19,unchanged_B4_relations=524,formula=freeze['formula'],new_v2=1,old_v2=0,
  no_originaldev_or_official=True,no_checkpoint_or_alpha_or_precision_selection=True,
  provenance_notes=['Delta mean is synthesized rank32/alpha32, not trained epoch2.',
   'Raw39 direct recovery: original responses plus78 byte-identical JSON serializations checked against original complete SHA; no remote directory/archive assertion.',
   'Joint Sap query FP16 CUDA to FP32 CPU16threads amendment, same public bank/policy; no numerical equivalence or pure weight-mean causality claimed.',
   'F0 separately authorized after strict positive fixed F1 gate; this entry only F1.'],
  inherited_reviews={'weight_CPU':'/workspace/patientphex-runs/round2-20261023/validation/grounded-weight-average-F1-final-review/source-cpu-check.json'},
  model_and_embeddings_retained_remotely=True,entry_time_unix=time.time())
 save(D/'final-review-entry.json',entry)
 files={}
 for p in S.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json','.md'] and '__pycache__' not in p.parts:files['source/'+p.relative_to(S).as_posix()]=ref(p)
 for folder in ['quality-v1','quotes-v1','normalization-preparation','normalized-v1','Sap-CPU100-v1/terminal']:
  for p in (D/folder).rglob('*'):
   if p.is_file():files['actual/'+p.relative_to(D).as_posix()]=ref(p)
 for p in list(named.values())+[D/'final-review-entry.json',D/'infer-v1/terminal/output/responses.jsonl',D/'cpu-v1/cpu-evidence.zip']:
  if D in p.parents:files['actual/'+p.relative_to(D).as_posix()]=ref(p)
 for path in ['schema.py','inference/decode.py','inference/extract_quotes.py','quality/assemble.py','quality/support.py','normalization/strict.py','normalization/semantic.py','normalization/public_bank.py','normalization/policy_v1.json','normalization/held_prepare.py','normalization/held_finalize.py']:
  files['inherited/'+path]=ref(G/path)
 manifest=dict(files=files,not_self_contained_large_dependencies=['public model/base+both parent LoRAs+new mean adapter','fixed HPO/train32/native47-or39 inputs','public Sap model and bank; new FP32 query vectors remain at exact member-index locator'],no_gold_file_in_bundle=True)
 raw=io.BytesIO()
 with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
  for name,item in sorted(files.items()):assert ref(item['path'])==item;z.writestr(name,Path(item['path']).read_bytes())
  z.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
 with (B/'bundle.zip').open('xb') as h:h.write(raw.getvalue());h.flush();os.fsync(h.fileno())
 save(B/'plan.json',dict(archive=ref(B/'bundle.zip'),manifest=manifest,mount=MOUNT,remote=P+'/'+REL,source=ref(__file__),review_entry=ref(D/'final-review-entry.json')))
 print(json.dumps(dict(plan=ref(B/'plan.json'),bytes=len(raw.getvalue()),members=len(files)+1)))

def transfer():
 p=json.loads((B/'plan.json').read_text());assert ref(B/'bundle.zip')==p['archive']
 sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
 c=Client();c.initialize();(B/'transport').mkdir(exist_ok=True)
 def call(name,args):
  stem=B/'transport'/str(time.time_ns());safe=dict(args)
  if 'content' in safe:safe['content']='[fixed archive chunk omitted]'
  save(stem.with_suffix('.intent.json'),dict(tool=name,args=safe));r=c.request('tools/call',dict(name=name,arguments=args));save(stem.with_suffix('.response.json'),r);assert not r.get('isError'),'Original I/O receipt retained; no model/quality action'
  return r.get('structuredContent') or json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 raw=(B/'bundle.zip').read_bytes();chunks=[]
 for offset in range(0,len(raw),65536):
  block=raw[offset:offset+65536];name=f'part-{offset:08}';v=call('write_file',dict(mount_id=MOUNT,path=REL+'/'+name,content=base64.b64encode(block).decode(),encoding='base64',create_parents=True));assert v['size']==len(block) and v['sha256']==hashlib.sha256(block).hexdigest();chunks.append(name)
 code='from pathlib import Path\nimport hashlib,json,os\np=Path('+repr(P+'/'+REL)+');b=b"".join((p/n).read_bytes() for n in '+repr(chunks)+');assert len(b)=='+str(len(raw))+' and hashlib.sha256(b).hexdigest()=='+repr(p['archive']['sha256'])+'\nq=p/"bundle.zip"\nif q.exists():assert q.read_bytes()==b\nelse:\n with q.open("xb") as h:h.write(b);h.flush();os.fsync(h.fileno())\nwith q.open("rb") as h:os.fsync(h.fileno())\nassert q.read_bytes()==b\nf=p\nwhile True:\n fd=os.open(f,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\n if f.parent==f:break\n f=f.parent\nprint(json.dumps(dict(path=str(q),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),file_and_parent_chain_fsynced=True,reopened_exact=True)))'
 v=call('run_command',dict(mount_id=MOUNT,command='python3 -c '+shlex.quote(code),timeout_seconds=30));assert v['exit_code']==0;remote=json.loads(v['stdout']);parts=[]
 for offset in range(0,len(raw),131072):
  v=call('read_file',dict(mount_id=MOUNT,path=REL+'/bundle.zip',offset=offset,limit=min(131072,len(raw)-offset)));parts.append(base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode())
 restored=b''.join(parts);assert restored==raw
 with zipfile.ZipFile(io.BytesIO(restored)) as z:
  assert set(z.namelist())==set(p['manifest']['files'])|{'manifest.json'}
  for name,item in p['manifest']['files'].items():b=z.read(name);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
 save(B/'complete.json',dict(status='COMPLETE_ACTUAL_SERVER_BACKUP_AND_READBACK',plan=ref(B/'plan.json'),remote=remote,archive=ref(B/'bundle.zip'),members=len(p['manifest']['files'])+1,all_archive_members_SHA_verified=True,actual_download_readback_bytes_equal=True,source=ref(__file__),no_model_fit_quality=True))
 print(json.dumps(ref(B/'complete.json')))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['build','transfer']);v=a.parse_args();build() if v.stage=='build' else transfer()
