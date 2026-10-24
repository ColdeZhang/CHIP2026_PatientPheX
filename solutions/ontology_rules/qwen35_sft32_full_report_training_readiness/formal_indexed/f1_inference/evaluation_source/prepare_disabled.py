"""Attach the newly ACK-bound inference plan, still disabled and no quality."""
from pathlib import Path
import datetime,importlib.util,json
S=Path(__file__).parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/qwen35-fulltext32-indexed-F1-full80-v1')
spec=importlib.util.spec_from_file_location('new_fulltext_eval',S/'evaluate_full80.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(S/'unbound-template.json');assert p['execution_ready'] is False
p['inference_references']={n:m.ref(D/f) for n,f in [('inference_source','infer_full80.py'),('inference_launcher','launch.py'),('inference_plan','plan.json')]}
p['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
p['template_source']=m.ref(S/'unbound-template.json')
m.reference_gate(p,require_ready=False)
m.save(D/'evaluation-template.json',p)
print(json.dumps(dict(status='DISABLED_TEMPLATE_BOUND_NO_GOLD',plan=m.ref(D/'evaluation-template.json'))))
