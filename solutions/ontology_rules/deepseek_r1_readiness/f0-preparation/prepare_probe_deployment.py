from pathlib import Path
import json,hashlib,tarfile,io,shutil,statistics
D=Path(__file__).resolve().parents[1];F=D/'f0-preparation';S=F/'probe-source';P=F/'probe';P.mkdir(exist_ok=False);stage=P/'deployment';stage.mkdir()
remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/f0-longest-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
plan=json.loads((S/'plan.template.json').read_text());manifest=[]
for name,digest in plan['sources'].items():
 p=S/name;assert sha(p)==digest;shutil.copyfile(p,stage/name);manifest.append({'name':name,'local':ref(p),'remote':remote+'/'+name})
bysha={}
for p in D.rglob('*'):
 if p.is_file() and p.stat().st_size<200000 and not str(p).startswith(str(P)):
  bysha.setdefault((p.stat().st_size,sha(p)),p)
for key in ['assets_result','header_inventory','expected_files','yarn_result','derived_shapes','derived_config','codec_CPU_result','transport_manifest']:
 old=plan[key];p=bysha[(old['bytes'],old['sha256'])];name='metadata/'+key+'.json';dst=stage/name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(p,dst);plan[key]={**old,'path':remote+'/'+name};manifest.append({'name':name,'local':ref(p),'remote':remote+'/'+name})
C=F/'cpu';cpu=json.loads((C/'actual-result.json').read_text());audit=json.loads((C/'recovered/all58-input-audit.json').read_text());longest=max(range(58),key=lambda i:audit[i]['native_input_tokens']);assert longest==30
for key,p in [('CPU_result',C/'actual-result.json'),('all58_audit',C/'recovered/all58-input-audit.json'),('selected_prompts',C/'recovered/selected-prompts.jsonl'),('selected_encoded',C/'recovered/selected-encoded.json')]:
 name='metadata/'+key+('.jsonl' if key=='selected_prompts' else '.json');shutil.copyfile(p,stage/name);plan[key]={**ref(p),'path':remote+'/'+name};manifest.append({'name':name,'local':ref(p),'remote':remote+'/'+name})
plan.update(selected_pair_index=cpu['selected_indices'].index(longest),selected_CPU_indices=[longest],row_ids=[audit[longest]['row_id']],input_tokens=[audit[longest]['native_input_tokens']])
plan['status']='F0_CPU_BINDING_COMPLETE_WAITING_EXACT9_COPY_ACK_NOT_EXECUTABLE';assert plan['execution_ready'] is False and plan['target_copy_ACK'] is None
save(P/'plan.pending.json',plan);save(P/'deployment-manifest.json',{'remote':remote,'files':manifest,'no_GPU_start':True,'status':'SOURCE_AND_REAL_F0_CPU_ONLY_COPY_ACK_PENDING'})
with tarfile.open(P/'deployment.tar.gz','w:gz') as t:
 for v in manifest:
  p=stage/v['name'];info=t.gettarinfo(str(p),arcname=v['name']);info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o600
  with p.open('rb') as f:t.addfile(info,f)
summary={'status':cpu['status'],'actual_CPU_result':ref(C/'actual-result.json'),'actual_CPU_tool':ref(C/'actual-cpu-tool.json'),'source':ref(C/'prepare_cpu.py'),'plan':ref(C/'plan.json'),'collection':ref(C/'collection-manifest.json'),'baseline_refs':ref(C/'f0-baseline-references.json'),'program_seconds':cpu['program_seconds'],'selected':cpu['selected'],'lengths':cpu['native_lengths'],'conventional_median':statistics.median([a['native_input_tokens'] for a in audit]),'probe_source_check':ref(S/'source-check.json'),'probe_template':ref(S/'plan.template.json'),'pending_plan':ref(P/'plan.pending.json'),'no_new_GPU_gold_quality':True}
save(F/'cpu-terminal-manifest.json',summary);print(json.dumps({'CPU_manifest':ref(F/'cpu-terminal-manifest.json'),'deployment_bundle':ref(P/'deployment.tar.gz'),'members':len(manifest),'longest':longest,'pending_plan':ref(P/'plan.pending.json')}))
