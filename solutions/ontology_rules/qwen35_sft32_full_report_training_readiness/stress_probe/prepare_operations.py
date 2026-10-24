from pathlib import Path
import hashlib,json,tarfile,io,shutil
B=Path(__file__).parent;R=Path('/workspace/patientphex-runs/round2-20261023');W=Path('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/qwen35_sft32_full_report_training_readiness/stress_probe')
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(n,x):
 with (B/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
s=(R/'research/qwen35-sft32-full-report-feasibility/f0-preparation/full58-v1/operations.py').read_text()
s=s.replace('Single full58 launch','Single 420s stress launch').replace('full58','stress').replace('1200','420').replace("'infer_stress.py'","'probe.py'").replace('ACTUAL_ALL13_SOURCE_INPUT_CACHE_SHA_COMPILE_FSYNC','ACTUAL_STRESS_SOURCE_DATA_SHA_COMPILE_FSYNC')
s=s.replace("'summary.json','failure.json'","'started.json','run-manifest.json','step.json','complete.json','failure.json'")
s=s.replace("out['completed_canonical_files']=len(list((p/'canonical-records').glob('*.json')))","out['completed_backward_rows']=len(list(p.glob('stage-*-after.json')))\nout['latest_stages']=sorted(f.name for f in p.glob('stage-*.json'))[-4:]")
start="names=['actual-input-and-assets.json','base-loading.json','summary.json','complete.json','failure.json','fresh-gpu.json','driver-started.json','child-started.json','wrapper-complete.json','gpu-release.json','responses.jsonl','canonical.jsonl','stderr.log','stdout.log']"
new="names=['started.json','run-manifest.json','step.json','complete.json','failure.json','fresh-gpu.json','driver-started.json','child-started.json','wrapper-complete.json','gpu-release.json','first-update.json','stderr.log','stdout.log']"
assert start in s;s=s.replace(start,new)
s=s.replace("for folder in ['generated-ids','raw-responses','canonical-records']:paths.extend(sorted((p/folder).glob('*.json')))","paths.extend(sorted(p.glob('stage-*.json')))")
s=s.replace("assert len(path.parts) == 1 or (len(path.parts) == 2 and path.parts[0] in ['generated-ids', 'raw-responses', 'canonical-records'])","assert len(path.parts) == 1")
s=s.replace('ACTUAL_UNIQUE_FULL58_TERMINAL_COLLECTED','ACTUAL_UNIQUE_STRESS420_TERMINAL_COLLECTED')
# Fresh source/15base asset ACK is a separate no-model read before unique start.
s=s.replace("choices=['deploy', 'start', 'inspect', 'collect']","choices=['deploy', 'preflight', 'start', 'inspect', 'collect']")
preflight='''elif a.action == 'preflight':
    assert not (B/'preflight-intent.json').exists()
    save('preflight-intent.json',dict(at_epoch=time.time(),plan_sha256=digest((B/'plan.json').read_bytes()),no_model=True))
    code = """from pathlib import Path
import hashlib,json,os,subprocess,time
p=Path(PARAM['root']);plan=json.loads((p/'plan.json').read_text());model=Path(plan['model'])
assert not (p/'driver-started.json').exists()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as r:
  for b in iter(lambda:r.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((p/'model-source-manifest.json').read_text())
files=manifest['files'];actual={}
for item in files:
 name=item['name'];f=model/name;expected=item['sha256'];assert f.is_file() and f.stat().st_size==item['size'] and sha(f)==expected
 actual[name]=dict(bytes=f.stat().st_size,sha256=expected)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
target=[l for l in gpu.splitlines() if l.split(',')[1].strip()==plan['gpu_uuid']];assert len(target)==1
idx,uuid,used,free,total=[x.strip() for x in target[0].split(',')]
assert idx=='2' and int(used)<=100 and int(free)>=78000 and int(total)==81920 and uuid not in apps
print(json.dumps(dict(status='ACTUAL_FRESH_GPU_AND_ALL_BASE_SHA_NO_MODEL',asset_count=len(actual),model_files=actual,gpu=target[0],no_process=True,project_free=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize,shm_free=os.statvfs(model).f_bavail*os.statvfs(model).f_frsize,at_epoch=time.time(),plan_sha256=sha(p/'plan.json'))))
"""
    result=remote_python(code,{'root':root},'preflight-tool.json',60)
    save('preflight.json',result);print(json.dumps({k:v for k,v in result.items() if k!='model_files'}))
'''
s=s.replace("elif a.action == 'start':",preflight+"elif a.action == 'start':\n    preflight=json.loads((B/'preflight.json').read_text());assert preflight['status']=='ACTUAL_FRESH_GPU_AND_ALL_BASE_SHA_NO_MODEL' and preflight['plan_sha256']==digest((B/'plan.json').read_bytes())\n    assert time.time()-preflight['at_epoch']<300 and preflight['asset_count']==15")
(B/'operations.py').write_text(s);compile(s,'operations.py','exec')
plan=json.loads((B/'plan.json').read_text());members=list(plan['files'])+['plan.json']
refs={n:{k:v for k,v in ref(B/n).items() if k!='path'} for n in members}
with tarfile.open(B/'deployment.tar.gz','w:gz') as archive:
 for name in members:archive.add(B/name,arcname=name,recursive=False)
save('deployment-manifest.json',dict(files=refs,archive=ref(B/'deployment.tar.gz'),plan=ref(B/'plan.json'),no_model_started=True))
save('entry.json',dict(source=ref(B/'probe.py'),launch=ref(B/'launch.py'),plan=ref(B/'plan.json'),operations=ref(B/'operations.py'),steps=['deploy','preflight','start once','inspect original','collect once after terminal'],budget_seconds=420,grace_seconds=30,formal_training_authorized=False,no_quality=True))
W.mkdir(parents=True,exist_ok=False)
for p in B.iterdir():
 if p.suffix=='.py' or p.name in ['plan.json','entry.json','probe-data.json','preparation-import-correction.json','deployment-manifest.json']:
  shutil.copyfile(p,W/p.name)
for p in B.glob('*.py'):compile(p.read_text(),str(p),'exec')
save('source-manifest.json',dict(status='PREPARED_NOT_STARTED',files={p.name:ref(p) for p in B.iterdir() if p.is_file()},repo=str(W),excluded_from_git=['probe-encoded.json.gz','CPU-result.json','native-result.json','deployment.tar.gz'],single_start=True))
shutil.copyfile(B/'source-manifest.json',W/'source-manifest.json')
print(json.dumps(dict(source=ref(B/'probe.py'),plan=ref(B/'plan.json'),operations=ref(B/'operations.py'),entry=ref(B/'entry.json'),archive=ref(B/'deployment.tar.gz'))))
