"""Bind the CPU contract to actual frozen native function bodies; no execution."""
from pathlib import Path
import ast,hashlib
D=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
model=R/'research/qwen35-feasibility/independent-review/modeling_qwen3_5.py'
native=R/'research/patient-first-full64-formal-preparation/actual-runtime-source/loss_utils.py'
selected=D.parent/'stress_probe/selected_loss.py'
assert hashlib.sha256(model.read_bytes()).hexdigest()=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
assert hashlib.sha256(native.read_bytes()).hexdigest()=='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'
def extract(path,name,cls=None):
 nodes=ast.parse(path.read_text()).body
 if cls:nodes=next(x for x in nodes if isinstance(x,ast.ClassDef) and x.name==cls).body
 node=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name);node.decorator_list=[]
 return ast.unparse(node)
head='from __future__ import annotations\nimport json,os,sys,time,hashlib\nfrom types import SimpleNamespace\nimport torch\nfrom torch import nn\nimport indexed_head as indexed\nCausalLMOutputWithPast=SimpleNamespace\n'
head+=extract(native,'fixed_cross_entropy')+'\nnative_fixed_cross_entropy=fixed_cross_entropy\n'+extract(selected,'selected_causal_loss')+'\n'+extract(model,'forward','Qwen3_5ForCausalLM')+'\n'
body=(D/'cpu_checks_body.py').read_text()
(D/'cpu_checks.py').write_text(head+body)
