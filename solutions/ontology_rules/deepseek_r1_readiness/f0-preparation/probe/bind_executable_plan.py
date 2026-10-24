"""Bind actual copy ACK to already frozen F0 CPU/source plan, no remote/NN/quality action."""
from pathlib import Path
import json,hashlib,os,datetime
P=Path(__file__).parent;F=P.parent;A=F/'asset-copy/formal';C=F/'cpu'
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(p,r):assert ref(p)['bytes']==r['bytes'] and ref(p)['sha256']==r['sha256']
complete=json.loads((A/'complete.json').read_text());ackpath=A/'asset-copy-ack.json';ack=json.loads(ackpath.read_text());dep=json.loads((P/'actual-deployment.json').read_text())
assert complete['status']=='ALL9_FULL_SHA_FSYNC_R1_PRIVATE_COPY_COMPLETE_SHM_NOT_PERSISTENT' and complete['ack_sha256']==ref(ackpath)['sha256'];assert dep['status']=='ACTUAL_DEPLOYMENT_17_SHA_COMPILE_RUNTIME_SOURCE_PASS_NO_GPU'
plan=json.loads((P/'plan.pending.json').read_text());assert plan['execution_ready'] is False and plan['target_copy_ACK'] is None
expected=json.loads((F/'asset-copy/expected-transport-manifest.json').read_text());assert ack['files']==expected['files'] and len(ack['files'])==9 and ack['fsync_all_files'] and not ack['target_is_persistent'] and ack['total_bytes']==16388603517
assert ack['target_directory']==plan['model_dir'] and ack['MS_revision']==plan['MS_revision'] and ack['transport_manifest_sha256']==plan['transport_manifest']['sha256']
for name,digest in plan['sources'].items():assert ref(F/'probe-source'/name)['sha256']==digest
for v in json.loads((P/'deployment-manifest.json').read_text())['files']:
 checked(v['local']['path'],v['local']);assert dep['files'][v['name']]=={k:v['local'][k] for k in ['bytes','sha256']}
checked(C/'actual-result.json',plan['CPU_result']);assert plan['selected_CPU_indices']==[30] and plan['input_tokens']==[6028] and plan['row_ids']==['3563609:relations:0']
plan['target_copy_ACK']={**ref(ackpath),'path':plan['model_dir']+'/asset-copy-ack.json'};plan['execution_ready']=True;plan['status']='F0_LONGEST_FIXED_6028_ACTUAL_CPU_COPY_ACK_READY_ONE_FORMAT_COST_PROBE';plan['authorization']='Root conditional authorization after actual full58 CPU/full9 copy ACK/source/fresh GPU: single original longest,4096/240+TERM30,first raw persisted then strict+normalized; no gold/full58/quality.'
plan['actual_deployment']=ref(P/'actual-deployment.json');plan['actual_copy_complete']=ref(A/'complete.json');plan['same_CPU_expected_target_runtime']=dep['runtime'];plan['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (P/'plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
refs=[F/'cpu-terminal-manifest.json',C/'actual-result.json',C/'collection-manifest.json',F/'probe-source/source-check.json',P/'deployment-manifest.json',P/'deployment-upload.json',P/'post-UNKNOWN-actual-inventory.json',P/'actual-deployment.json',P/'actual-deployment-tool.json',P/'plan.json',P/'start_once.py',P/'poll_original.py',P/'bind_executable_plan.py',A/'asset-copy-ack.json',A/'complete.json',A/'run-plan.json',A/'controller-interruption.json']+[F/'probe-source'/n for n in plan['sources']]
manifest={'status':'F0_SINGLE_LONGEST_READY_NOT_STARTED_NO_QUALITY','files':[ref(p) for p in refs],'CPU_and_asset_scope':'actual full58 native input and exact9 persisted original to ephemeral SHM, no repeated model profiling','budget':{'main_seconds':240,'TERM_seconds':30,'max_new_tokens':4096},'single_row':{'original_index':30,'row_id':'3563609:relations:0','input_tokens':6028},'H32':'Original copy monitor RPC error did not stop/restart original bounded writers; source stop only after confirmed fetch terminal','H49':'New runner IDs/endstats before decode, raw text/IDs before parse, immutable raw file and separate parsed output; current F1 old live source unchanged'}
with (P/'entry-manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'plan':ref(P/'plan.json'),'manifest':ref(P/'entry-manifest.json')}))
