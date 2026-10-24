"""Read-only final observation after original downloader stops; never infer its exit code."""
from pathlib import Path
import sys,json,hashlib,shlex,datetime
D=Path(__file__).resolve().parent
assert not (D/'download-terminal.json').exists()
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
c=Client();c.initialize();m='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':m,**a}});assert not r.get('isError'),r;return r['structuredContent']
job=json.loads((D/'download-job.json').read_text())['result']
processes=call('list_processes',{});processes=processes if isinstance(processes,list) else processes['processes']
matches=[x for x in processes if x['id']==job['id']];assert len(matches)==1 and matches[0]['running'] is False, matches
code=r'''
from pathlib import Path
import json,hashlib,subprocess,shlex,datetime,os
R=Path.cwd(); pid=ORIGINAL_PID
ps=subprocess.run(['ps','-eo','pid=,ppid=,args='],capture_output=True,text=True,check=True)
writers=[]
for line in ps.stdout.splitlines():
 parts=line.split(None,2)
 if len(parts)!=3:continue
 p,pp,args=parts
 if int(p)==os.getpid():continue
 try:argv=shlex.split(args)
 except ValueError:argv=[]
 if int(p)==pid or any(a=='download_assets.py' or a.endswith('/download_assets.py') for a in argv):writers.append({'pid':int(p),'ppid':int(pp),'args':args})
assert not writers, writers
raw=(R/'downloads/download-state.json').read_bytes();state=json.loads(raw)
plan=json.loads((R/'download-plan.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
etags={'HP.tar.gz':'"0x8D9B91C61B87AEC"','pmc_model_new.bin':'"0x8D9B91C5F51715C"'}
assets=[]
for a in plan['assets']:
 full=R/'downloads'/a['filename'];p=full if full.exists() else full.with_name(full.name+'.part')
 before=p.stat();digest=sha(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 assets.append({'filename':a['filename'],'path':str(p.relative_to(R)),'official_asset_id':a['asset_id'],'official_url':a['url'],'expected_bytes':a['bytes'],'actual_bytes':before.st_size,'sha256':digest,'complete_by_length':before.st_size==a['bytes'],'last_state':state['files'].get(a['filename']),'frozen_official_ETag':etags[a['filename']],'ETag_evidence_sha256':plan['access_evidence']['HEAD_and_range_sha256'],'stable_during_hash':True})
print(json.dumps({'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ps_original_PID_or_exact_downloader_argv_matches':writers,'assets':assets,'raw_state_sha256':hashlib.sha256(raw).hexdigest(),'raw_state_utf8':raw.decode(),'complete_report_exists':(R/'downloads/complete.json').exists(),'no_remote_files_modified':True}))
'''.replace('ORIGINAL_PID',str(job['pid']))
command='../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code)
receipt=call('run_command',{'command':command,'cwd':'public-neuralcr-readiness','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':40})
assert receipt['exit_code']==0,receipt
result=json.loads(receipt['stdout']);raw=result.pop('raw_state_utf8').encode();assert hashlib.sha256(raw).hexdigest()==result['raw_state_sha256']
(D/'download-final-state.json').write_bytes(raw)
result.update(status='complete' if result['complete_report_exists'] else 'incomplete_partial_preserved',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inspection_code_sha256=hashlib.sha256(code.encode()).hexdigest(),original_job=job,original_process_terminal=matches[0],original_process_exit_code=None,exit_code_note='MCP does not expose original return code; no rc124 inference',original_deadline_UTC='2026-10-24T00:30:44.810536407Z',TERM_grace_deadline_UTC='2026-10-24T00:31:14.810536407Z',inspection_only_rc=receipt['exit_code'])
(D/'download-terminal.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
