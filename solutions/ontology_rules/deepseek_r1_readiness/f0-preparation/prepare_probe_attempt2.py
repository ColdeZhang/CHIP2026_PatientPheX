"""Preserve actual zero-NN gate failure; single reported memory.total field repair."""
from pathlib import Path
import json,hashlib,shutil,difflib,importlib.util,copy,time,datetime,tarfile
F=Path(__file__).parent;old=F/'probe';S=F/'probe-source';P=F/'probe-attempt2';P.mkdir(exist_ok=False)
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
w=json.loads((old/'recovered/wrapper-result.json').read_text());assert w['status']=='failed_wrapper_preserved' and 'GPU_main_started_at' not in w and w['target_UUID_has_no_compute_process'];assert not (old/'recovered/terminal/result.json').exists()
for name in ['runner.py','launcher.py','strict_final.py','sparse_final.py','normalized_final.py']:shutil.copyfile(S/name,P/name)
p=P/'launcher.py';a=p.read_text();b=a.replace("int(values[5])==81156","int(values[5])==81920");assert a!=b and a.count('int(values[5])==81156')==1;p.write_text(b);(P/'launcher.diff').write_text(''.join(difflib.unified_diff(a.splitlines(keepends=True),b.splitlines(keepends=True),fromfile='original',tofile='single-total-field-fixed')))
t=time.monotonic();spec=importlib.util.spec_from_file_location('f0_actual_gate_cpu',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);gate=w['fresh_GPU2'];uuid='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc';m.validate(gate,uuid);cases=[]
for label,index,value in [('excess_used',3,'101'),('insufficient_free',4,'79999'),('wrong_total_old_free_value',5,'81156'),('wrong_index',0,'1')]:
 g=copy.deepcopy(gate);parts=[x.strip() for x in g['gpu'].split(',')];parts[index]=value;g['gpu']=', '.join(parts)
 try:m.validate(g,uuid)
 except AssertionError:cases.append(label)
 else:raise AssertionError('negative accepted: '+label)
g=copy.deepcopy(gate);g['compute_apps']+='\n'+uuid+', 12345'
try:m.validate(g,uuid)
except AssertionError:cases.append('target_GPU_process_present')
else:raise AssertionError('busy target accepted')
CPU={'status':'PASS_ACTUAL_NVIDIA_SMI_FIELD_ORDER_CPU_ONLY_SINGLE_TOTAL_FIX','source':ref(p),'prior_source':ref(S/'launcher.py'),'actual_wrapper_failure':ref(old/'recovered/wrapper-result.json'),'actual_gate':gate,'positive_actual_gate_passed':True,'negative_rejections':cases,'actual_fixture_seconds':time.monotonic()-t,'only_changed_semantic':'memory.total exact81920 instead of historical memory.free81156; original used<=100/free>=80000/UUID and process check unchanged','no_GPU_model_or_native_tokenizer':True,'original_probe_NN_calls':0};save(P/'CPU-gate-fix.json',CPU)
plan=json.loads((old/'plan.json').read_text());plan['sources']['launcher.py']=ref(p)['sha256'];plan['status']='F0_LONGEST_ATTEMPT2_SINGLE_GPU_TOTAL_FIELD_FIX_READY';plan['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();plan['previous_zero_NN_wrapper_failure']=ref(old/'recovered/wrapper-result.json');plan['CPU_actual_gate_fix']=ref(P/'CPU-gate-fix.json');plan['actual_deployment_scope']='Original immutable metadata/runner/parser deployment. New launcher only differs in memory.total literal; attempt2 actual-prestart checks new sources+plan+all same original immutable refs.';plan['budget_scope']='Original240s main process never started in attempt1; preserve .236604s wrapper-only failure and run the sole model attempt with unchanged240+TERM30.'
save(P/'plan.json',plan)
for name in ['start_once.py','poll_original.py']:
 text=(old/name).read_text().replace('readiness/f0-longest-v1','readiness/f0-longest-v2');(P/name).write_text(text)
shutil.copyfile(old/'freeze_terminal.py',P/'freeze_terminal.py')
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
with tarfile.open(P/'sources.tar.gz','w:gz') as tf:
 for name in plan['sources']:
  p=P/name;info=tf.gettarinfo(str(p),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o600
  with p.open('rb') as f:tf.addfile(info,f)
save(P/'entry-manifest.json',{'status':'SINGLE_TOTAL_FIELD_FIX_NO_NN_STARTED','original_zero_NN_attempt':[ref(old/'plan.json'),ref(old/'recovered/wrapper-result.json'),ref(old/'recovered/wrapper-started.json'),ref(old/'recovered/collection-manifest.json')],'CPU_fix':ref(P/'CPU-gate-fix.json'),'plan':ref(P/'plan.json'),'source_diff':ref(P/'launcher.diff'),'sources':[ref(P/n) for n in plan['sources']],'same_model_inputs_codec_budget':True,'only_fresh_GPU_after_checkpoint':True})
print(json.dumps({'plan':ref(P/'plan.json'),'entry':ref(P/'entry-manifest.json'),'CPU':ref(P/'CPU-gate-fix.json'),'source':ref(P/'launcher.py'),'diff':(P/'launcher.diff').read_text()}))
