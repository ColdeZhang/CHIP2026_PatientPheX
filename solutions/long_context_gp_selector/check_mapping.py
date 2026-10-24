"""Independent comparison to already-used original GP matrix addresses; no NN/gold."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/long-context-gp-selector-readiness-v1/formal-v1';B=R/'research/biomedical-long-context-readiness-20261024/formal-v1/baseline-targets.json'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 job=json.load(open(D/'data.json'))['jobs'][-1];old=json.load(open(B));newpos=defaultdict(list);oldpos=defaultdict(list)
 for row,ts in job['window_targets'].items():
  for t in ts:
   x=job['targets'][t['target_index']];k=(x['pmc_id'],x['offset'],x['length']);newpos[k].append((row,t['block_index'],t['i'],t['j']))
 byid={r['target_id']:r for r in old['records']}
 for row,ts in old['window_targets'].items():
  for t in ts:
   x=byid[t['target_id']];k=(x['pmc_id'],x['offset'],x['length']);oldpos[k].append((row,t['block_index'],t['i'],t['j']))
 for r in old['records']:
  k=(r['pmc_id'],r['offset'],r['length']);assert sorted(set(newpos[k]))==sorted(set(oldpos[k])) and r['representable']==bool(newpos[k])
 out=dict(status='ORIGINAL1313_MATRIX_COORDINATES_EXACT_IN_NEW_FULL16417_MAP',source=ref(__file__),data=ref(D/'data.json'),old_mapping=ref(B),old_targets=len(old['records']),new_targets=len(job['targets']),no_model=True,no_gold=True,no_quality=True)
 p=D/'mapping-contract.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
