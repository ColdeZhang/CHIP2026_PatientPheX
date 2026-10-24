"""Freeze authorized two-endpoint policy/source now; bind only actual terminal refs later."""
from pathlib import Path
import json,hashlib,importlib.util
F=Path(__file__).parent;D=F.parent;R=D.parent.parent

def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
spec=importlib.util.spec_from_file_location('prepared_r1_eval',F/'evaluate_full80.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=json.loads((R/'research/medgemma-relation-control/evaluation-preparation/evaluation-template.json').read_text())
refs={k:old['references'][k] for k in m.SHARED_SHA if k in old['references']}
refs.update(normalized_final=ref(F/'normalized_final.py'),strict_final=ref(F/'strict_final.py'),input_audit=ref(D/'published-interface-v1/cpu/recovered/all80-input-audit.json'))
assert set(refs)==set(m.SHARED_SHA) and all(v['sha256']==m.SHARED_SHA[k] for k,v in refs.items())
endpoints={k:None for k in m.DYNAMIC_NAMES};endpoints.update(inference_source=ref(F/'infer_full80.py'),inference_launcher=ref(F/'launch_full80.py'),inference_plan=ref(F/'plan.json'),cached_row0=ref(D/'published-interface-v1/gpu/recovered/terminal/row0.json'),cached_row73=ref(D/'published-interface-v1/codec-longest-v1/gpu/recovered/terminal/row0.json'))
obj={'schema':m.SCHEMA,'control_sha256':ref(F/'evaluate_full80.py')['sha256'],'execution_ready':False,'policy':m.POLICY,'references':refs,'endpoint_references':endpoints,'gold_path':old['gold_path'],'historical_F1_gold_sha256':m.GOLD_SHA256,'output_dir':str(F/'quality')}
with (F/'evaluation-template.json').open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
print(json.dumps(ref(F/'evaluation-template.json')))
