import json,pathlib,sys,shlex,hashlib,datetime
R=pathlib.Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
rr=json.loads((R/'download-plan.json').read_text())['remote'];c=Client();c.initialize()
def call(n,**a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':rr['mount_id'],**a}});assert not r.get('isError');return r.get('structuredContent') or json.loads(r['content'][0]['text'])
code="""import pathlib,json,hashlib
p=pathlib.Path('.');f=p/'runtime-repair-pip-report.json';d=json.loads(f.read_text());print(json.dumps({'pip_report_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'install':[{'name':a['metadata']['name'],'version':a['metadata']['version'],'download_info':a.get('download_info')} for a in d['install']]}))"""
x=call('run_command',command=shlex.quote(rr['python'])+' -c '+shlex.quote(code),cwd=rr['cwd'],env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},timeout_seconds=10);assert x['exit_code']==0
(R/'runtime-repair-dependency-manifest.json').write_text(json.dumps(json.loads(x['stdout']),indent=2)+'\n')
x=call('list_processes');ids=[]
for n in ['download-job.json','download-recovery-job.json','runtime-job.json','runtime-repair-job.json','cpu-job.json','cpu-attempt2-job.json']:
 d=json.loads((R/n).read_text());v=d.get('result',d.get('job',{}));
 if 'id' in v:ids.append(v['id'])
# Keep only owned job records, never expose unrelated command lines.
items=x if isinstance(x,list) else x.get('processes',[]);owned=[v for v in items if v.get('id') in ids]
(R/'owned-terminal-processes.json').write_text(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_ids':ids,'owned_processes':owned},indent=2)+'\n')
f=R/'cdn-runtime.private.json'
if f.exists():f.unlink()
s=json.loads((R/'attempt2/CPU-state.json').read_text());prior=json.loads((R/'CPU-state.json').read_text());assert s['status']=='complete_native_synthetic_cpu'
report='''# F-coref readiness: complete, CPU synthetic only

Fixed author model `biu-nlp/f-coref@e91dfff12879495d882ba9460d0c5d5dd44ade59` loaded successfully with unchanged inference source `ae224139196d122b225af5a7e73b5fc0b6e1076d`. The 362,162,101-byte weight SHA256 is `30ea6efe2bce49a857fe8aba41dcfe28b4147f77e3dec9f14100950bf4de1aa5`; all eight model files verified. Native unique parameters: **90,526,466** (82,118,400 backbone + 8,408,066 head). Together with current 9,597,467,990: **9,687,994,456**; this does not include a prospective NCR model.

The direct HF request failed before weight bytes (75.162s). Recovery used the fixed revision's official CDN and completed in 42.515s, inside the original 600s deadline. Signed URL runtime files were removed locally/remotely and never committed. Private dependency overlays used 109.500s plus 46.770s repair (156.269s cumulative). Shared environment and vendor sources remained unchanged.

Initial native attempt failed after 2.956s importing missing `click`, before model load. Private repair pinned `click==8.1.8`, `typer==0.16.1`. The unchanged native source's second attempt started 2026-10-24T00:33:39.518838Z and completed in **6.629952s**, before original deadline 00:34:10.916789Z; cumulative native time **9.586185s**. No attempt3 was needed. Torch2.7.1+cu126, Transformers4.51.3, spaCy3.8.7, NumPy2.2.6; CPU BF16, one thread, blank English spaCy, no spaCy neural model. All missing/unexpected/mismatched/error loading lists were empty.

All four synthetic examples returned valid original character slices. Exact-span mapping A attached **zero clusters**: native model prefers `Patient A` or larger NPs such as `The proband, Zoë,`, while supplied patient anchors are names. The repeated-name fixture returned no clusters. These are descriptive outputs, not evidence of clinical attribution quality. Original output and failure traces are retained.

The input-only anchor audit covered exactly train32 F2+F3 (87 patients /255 original mentions) and F1 (39/162), with no shared exact anchor coordinates or missing patient anchors. Literal alias strings are shared across patients in 5 training docs and2 F1 docs, so aliases cannot identify patients. No corpus text was sent through the model in this completed readiness phase; no gold labels, quality scores, GPU or LLM calls were used.

See `attempt2/CPU-state.json`, `attempt2/synthetic-native-clusters.json`, `attempt2/CPU.stderr.log`, `download-complete.json`, dependency manifests and owned terminal processes. Upstream vendor trailing whitespace/EOF formatting is intentionally preserved to retain exact source hashes. Later input-only native census, if executed, has its own separately frozen plan and results.
'''
(R/'READINESS-REPORT.md').write_text(report)
names=['READINESS-REPORT.md','finalize_readiness.py','download-initial-failure.json','download-complete.json','weight-source-manifest.json','download-recovery-status.json','download.stderr.log','download-recovery.stdout.log','runtime-state.json','runtime-dependency-manifest.json','runtime-repair-state.json','runtime-repair-dependency-manifest.json','CPU-state.json','CPU.stderr.log','cpu-job.json','cpu-attempt2-job.json','owned-terminal-processes.json','attempt2/CPU-state.json','attempt2/CPU.stderr.log','attempt2/synthetic-native-clusters.json']
manifest={'files':[{'path':n,'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest()} for n in names]}
(R/'readiness-actual-backup-filelist.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'files':len(names),'owned_processes':owned,'private_local_removed':not f.exists(),'native_cumulative_seconds':s['seconds']+prior['seconds']}))
