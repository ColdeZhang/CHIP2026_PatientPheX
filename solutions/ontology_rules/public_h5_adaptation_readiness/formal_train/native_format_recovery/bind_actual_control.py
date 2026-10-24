"""Bind only actual terminal identities and original shared deadline; no scoring."""
from pathlib import Path
import hashlib,json,time
S=Path(__file__).parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/public-h5-train32-formal-v1/native-format-recovery')
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest=json.loads((D/'terminal-manifest.json').read_text())
assert manifest['state']['status']=='GPU_AND_NATIVE_COMPLETE_CPU_ENDPOINTS_PENDING' and manifest['state']['returncode']==0
assert manifest['state']['owned_child_absent'] and not manifest['state'].get('timed_out')
for n,r in manifest['members'].items():
    p=D/'terminal'/n
    assert p.stat().st_size==r['bytes'] and ref(p)['sha256']==r['sha256']
trained=json.loads((D/'terminal/actual/complete.json').read_text())
assert trained['optimizer_iterations']==20 and trained['examples']==1252 and trained['epochs']==1
assert trained['original_H5_unchanged'] and not trained['held_input_or_gold_read']
assert trained['complete_state_restored']['model_all_exact'] and trained['complete_state_restored']['optimizer_all_exact']
assert trained['complete_state_restored']['new_object_global_step']==20
native=D/'terminal/held-final'
cache=json.loads((native/'manifest.json').read_text())
assert cache['model_weight_sha256']==trained['trained_H5']['sha256'] and cache['public_model_training_on_task']
assert cache['counts']['documents']==16 and len(cache['input_ids'])==len(set(cache['input_ids']))==16
plan=json.loads((S/'control-plan-template.json').read_text())
plan['deadline_epoch']=manifest['state']['deadline_epoch']
assert time.time()<plan['deadline_epoch']
plan['actual_terminal_manifest']=ref(D/'terminal-manifest.json')
plan['actual_training_terminal']=ref(D/'terminal/actual/complete.json')
plan['actual_native_manifest']=ref(native/'manifest.json')
plan['binder']=ref(Path(__file__))
with (D/'actual-control-plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps(dict(plan=ref(D/'actual-control-plan.json'),native_directory=str(native),remaining_seconds=plan['deadline_epoch']-time.time(),quality_calls=0)))
