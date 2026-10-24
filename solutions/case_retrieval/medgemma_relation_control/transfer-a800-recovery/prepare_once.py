from pathlib import Path
import json,sys,hashlib,shlex,datetime
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
Q=Path(__file__).resolve().parent;E=Q.parent;F=E/'transfer-a800'/'formal';p=json.loads((F/'run-plan.json').read_text())
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert not (Q/'deployment-complete.json').exists()
x=json.loads((F/'fetch-after-controller503.json').read_text())['result']['structuredContent'];assert not x['running']
assert json.loads((F/'server-stopped.json').read_text())['stopped']
t=json.loads(json.loads((Q/'target-partial-freeze-actual.json').read_text())['stdout']);s=json.loads(json.loads((Q/'source-stats-actual.json').read_text())['stdout']);ready=json.loads((F/'server-ready.json').read_text())
r={'target_root':p['destination'],'initial_finalized':t['finalized'],'initial_partials':t['partials'],'source_stats':s['files'],'original_transport_manifest_sha256':ready['manifest_sha256'],
   'original_window':{'source_job':'proc_af670473-434e-44f9-b19b-157139e9cd7a','fetch_job':'proc_f99147b3-27ed-4698-a5c1-4c4e0b6d0948','absolute_deadline_unix':1792810949.3566287,'status':'interrupted_controller_HTTP503_finally_source_stop_fetch_connection_refused','both_actually_stopped':True,'first_fetch_source_sha256':p['target_fetcher_sha256'],'source_ready_sha256':sha(F/'server-ready.json')},
   'verification_contract':'Initial finalized files inherited full SHA before atomic rename; only exact frozen stat reused without reread; partials never called complete; new completions full SHA; original persistent .100 retained'}
(Q/'resume-state.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
remote='research/medgemma-relation-control/transfer-a800-recovery'
files=['serve_assets_recovery.py','fetch_assets_recovery.py','resume-state.json']
refs={n:dict(bytes=(Q/n).stat().st_size,sha256=sha(Q/n)) for n in files}
receipts=[]
for mount,names in [(p['source_mount_id'],['serve_assets_recovery.py','resume-state.json']),(p['target_mount_id'],['fetch_assets_recovery.py','resume-state.json'])]:
 io.M=mount
 for n in names:io.upload(str(Q/n),remote+'/'+n)
 code='from pathlib import Path;import hashlib,json;refs='+repr({n:refs[n] for n in names})+';root=Path('+repr(remote)+');assert all((root/n).stat().st_size==r["bytes"] and hashlib.sha256((root/n).read_bytes()).hexdigest()==r["sha256"] for n,r in refs.items());print(json.dumps(dict(files=refs)))'
 got=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':15});assert got['exit_code']==0;receipts.append({'mount':mount,'actual':got})
p.update({'source_server':remote+'/serve_assets_recovery.py','target_fetcher':remote+'/fetch_assets_recovery.py','remote_recovery':remote,'source_server_sha256':refs['serve_assets_recovery.py']['sha256'],'target_fetcher_sha256':refs['fetch_assets_recovery.py']['sha256'],'resume_state_sha256':refs['resume-state.json']['sha256'],'original_manifest_sha256':ready['manifest_sha256'],'controller_sha256':sha(Q/'run_once.py'),'recovery_semantics':'Separate once-only window; original failure and partial retained; no new public download; monitor failure preserves remote jobs until known terminal or original deadline+30'})
p.pop('archive');p.pop('files')
p['local_evidence']=[dict(path=str(x),sha256=sha(x)) for x in [F/'server-stopped.json',F/'fetch-after-controller503.json',Q/'target-partial-freeze-actual.json',Q/'source-stats-actual.json',Q/'resume-state.json']]
(Q/'formal').mkdir();(Q/'formal'/'run-plan.json').write_text(json.dumps(p,indent=2)+'\n')
(Q/'deployment-complete.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipts':receipts,'files':refs,'no_jobs_started':True},indent=2)+'\n')
print('RECOVERY_SOURCES_DEPLOYED_NO_JOBS')
