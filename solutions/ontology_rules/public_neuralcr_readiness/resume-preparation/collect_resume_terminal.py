from pathlib import Path
import sys,json,hashlib,shlex
R=Path(__file__).resolve().parent
assert not (R/'terminal-inventory.json').exists()
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(n,a):
 x=c.request('tools/call',{'name':n,'arguments':{'mount_id':mount,**a}});assert not x.get('isError'),x;return x['structuredContent']
job=json.loads((R/'managed-job.json').read_text())['result'];processes=call('list_processes',{});processes=processes if isinstance(processes,list) else processes['processes'];entry=[p for p in processes if p['id']==job['id']];assert len(entry)==1 and not entry[0]['running']
code=r'''
from pathlib import Path
import json,hashlib,subprocess,shlex,re,os,datetime
R=Path.cwd();plan=json.loads((R/'resume-plan.json').read_text());state=json.loads((R/'resume-state.json').read_text());job=json.loads((R/'resume-job-state.json').read_text());assert job['status']=='failed_preserved' and job['returncode']==1 and not(R/'downloads/complete.json').exists()
writers=[]
for line in subprocess.run(['ps','-eo','pid=,ppid=,args='],text=True,capture_output=True,check=True).stdout.splitlines():
 p=line.split(None,2)
 if len(p)!=3 or int(p[0])==os.getpid():continue
 try:a=shlex.split(p[2])
 except ValueError:a=[]
 if int(p[0]) in [223215,job['child_pid']] or any(x in ['resume_assets.py','launch_resume.py'] or x.endswith('/resume_assets.py') or x.endswith('/launch_resume.py') for x in a):writers.append({'pid':p[0],'args':p[2]})
assert not writers,writers
def record(path):
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return {'path':str(path.resolve()),'bytes':before.st_size,'sha256':h.hexdigest(),'stable_during_hash':True}
old=[]
for a in plan['assets']:
 r=record(Path(a['prefix']['path']));assert r['bytes']==a['prefix']['bytes'] and r['sha256']==a['prefix']['sha256'];old.append(r)
actual=[];published={(x['filename'],x['start'],x['end_inclusive']):x for x in state['chunks_complete']}
for f in sorted((R/'chunks').iterdir()):
 m=re.fullmatch(r'(.+)\.(\d+)-(\d+)\.(done|part)',f.name);assert m,f.name
 name,start,end,suffix=m.groups();start=int(start);end=int(end);r=record(f);assert 0<=r['bytes']<=end-start+1
 if suffix=='done':assert r['bytes']==end-start+1
 r.update(filename=name,start=start,planned_end_inclusive=end,actual_end_exclusive=start+r['bytes'],suffix=suffix,published_completed=(name,start,end) in published)
 if r['published_completed']:p=published[(name,start,end)];assert r['sha256']==p['sha256'] and r['bytes']==p['bytes']
 actual.append(r)
assert len([r for r in actual if r['published_completed']])==len(published)
coverage=[]
for a in plan['assets']:
 intervals=[(0,a['prefix']['bytes'])]+[(r['start'],r['actual_end_exclusive']) for r in actual if r['filename']==a['filename'] and r['bytes']]
 intervals.sort();last=0;gaps=[];have=0
 for begin,end in intervals:
  assert begin>=last
  if begin>last:gaps.append([last,begin])
  have+=end-begin;last=end
 if last<a['bytes']:gaps.append([last,a['bytes']])
 assert have+sum(e-b for b,e in gaps)==a['bytes']
 coverage.append({'filename':a['filename'],'asset_id':a['asset_id'],'etag':a['etag'],'expected_bytes':a['bytes'],'verified_existing_bytes':have,'missing_bytes':a['bytes']-have,'missing_ranges_half_open':gaps})
stdout=(R/'stdout.log').read_text();stderr=(R/'stderr.log').read_text();assert len(stdout)<20000 and len(stderr)<20000
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'state_sha256':hashlib.sha256((R/'resume-state.json').read_bytes()).hexdigest(),'writers':writers,'prefixes':old,'chunks':actual,'coverage':coverage,'stdout':stdout,'stderr':stderr,'source_sha256':hashlib.sha256((R/'resume_assets.py').read_bytes()).hexdigest(),'complete_exists':False,'no_model_or_NN':True}))
'''
x=call('run_command',{'command':'../../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'public-neuralcr-readiness/resume-attempt2','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});assert x['exit_code']==0,x
report=json.loads(x['stdout']);(R/'terminal-small-files/stdout.log').write_text(report.pop('stdout'));(R/'terminal-small-files/stderr.log').write_text(report.pop('stderr'));report.update(status='failed_transport_complete_fragment_inventory_preserved',managed_process_terminal=entry[0],collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inspection_code_sha256=hashlib.sha256(code.encode()).hexdigest(),no_original_or_chunk_mutations=True)
(R/'terminal-inventory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'coverage':report['coverage'],'chunks':len(report['chunks']),'sha256':hashlib.sha256((R/'terminal-inventory.json').read_bytes()).hexdigest()}))
