"""Start the separately authorized single full public cache after owner checkpoint."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parent
F=B/'full'
assert not (F/'job.json').exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
deployment=json.loads((F/'deployment.json').read_text())
plan=json.loads((F/'inference-plan.json').read_text())
assert deployment['plan_sha256']==sha(F/'inference-plan.json')
assert plan['driver_sha256']==sha(B/'run_native_pairs.py')
assert plan['rows']==135000 and plan['time_cap_seconds']==600 and plan['TERM_grace_seconds']==15
checkpoint=json.loads((F/'owner-checkpoint.json').read_text())
assert len(checkpoint['commit'])==40 and checkpoint['source_saved_before_launch'] is True
assert checkpoint['source_owner']=='/root/round2_sequence'
(F/'driver-snapshot.py').write_bytes((B/'run_native_pairs.py').read_bytes())
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
c=Client();c.initialize()
args={'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f',
      'command':'timeout --signal=TERM --kill-after=15s 600s /dataset/miniforge3/envs/vlm-det/bin/python run_native_pairs.py',
      'cwd':'phenobert-pair-train64','env':plan['env']}
r=c.request('tools/call',{'name':'start_process','arguments':args});assert not r.get('isError'),r
job={'plan_sha256':sha(F/'inference-plan.json'),'owner_checkpoint_sha256':sha(F/'owner-checkpoint.json'),
     'source_sha256':sha(F/'driver-snapshot.py'),'arguments':args,'result':r['structuredContent']}
(F/'job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(job))
