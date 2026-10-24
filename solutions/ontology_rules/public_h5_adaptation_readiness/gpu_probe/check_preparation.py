"""No model imports: verify prepared CPU metadata, references and failure receipt contract."""
from pathlib import Path
import ast,gzip,hashlib,json,math,os,tempfile
S=Path(__file__).parent.resolve();plan=json.loads((S/'plan.json').read_text());checks={}
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for n,r in plan['sources'].items():assert ref(S/n)==r;checks['source_'+n]=True
for p in S.glob('*.py'):compile(p.read_text(),str(p),'exec')
checks['all_source_compile']=True
# Explicit keys used by probe/driver must actually exist, preventing pre-run key drift.
def pathkeys(node):
 if isinstance(node,ast.Name) and node.id=='plan':return []
 if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
  base=pathkeys(node.value)
  if base is not None:return base+[node.slice.value]
 return None
for name in ['probe.py','bounded_driver.py']:
 tree=ast.parse((S/name).read_text())
 for node in ast.walk(tree):
  keys=pathkeys(node)
  if keys:
   value=plan
   for key in keys:value=value[key]
checks['constant_plan_paths_resolve']=True
sampling=json.loads((S/'sampling.json').read_text());assert ref(S/'sampling.json')==plan['sampling'];rows=[json.loads(x) for x in gzip.open(plan['dataset']['path'],'rt')];assert len(rows)==1252
order=sampling['single_epoch_order'];assert order==sorted(range(1252),key=lambda i:hashlib.sha256(('20261024|epoch1|'+rows[i]['token_sha256']).encode()).hexdigest());checks['actual_deterministic_order']=True
assert [x['indices'] for x in sampling['probe_batches']]==[order[:64],order[64:128],order[-36:]];assert len(set(sum([x['indices'] for x in sampling['probe_batches']],[])))==164;checks['two_actual_full_batches_and_real36_tail']=True
for batch in sampling['probe_batches']:
 chosen=[rows[i] for i in batch['indices']];assert len(chosen)==batch['size'];assert [x['token_sha256'] for x in chosen]==batch['token_sha256']
 assert all(len(x[k])==32 for x in chosen for k in ['input_ids','token_type_ids','attention_mask']);assert all(x['target_columns'] and all(type(c)is int and 0<=c<17266 for c in x['target_columns']) for x in chosen)
 assert sum(x['target_columns']==[17265] for x in chosen)==batch['None_rows']>0
 assert sum(len(x['target_columns'])>1 for x in chosen)==batch['equivalent_column_rows']
checks['all_frozen_queries_and_targets_actual']=True;assert sum(b['equivalent_column_rows'] for b in sampling['probe_batches'])==4;checks['None_and_equivalent_columns_covered']=True
cov=json.loads(Path(plan['prior_variable_coverage']['path']).read_text());assert len(cov)==201;assert [x['model_variable'] for x in cov if '/pooler/dense/'in x['model_variable']]==plan['expected_disconnected_variables'];checks['unused_pooler2_original_identity']=True
assert plan['cap_seconds']==240 and plan['TERM_grace_seconds']==30 and plan['Adam']['fresh_iterations']==0 and not plan['Adam']['reset_between_three_steps'];checks['budget_and_state_contract']=True
# H52: nonfinite observations must survive on disk before a gate can reject them.
source=(S/'probe.py').read_text();tree=ast.parse(source);funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['safe','write']];env=dict(math=math,Path=Path,os=os,json=json);exec(compile(ast.Module(body=funcs,type_ignores=[]),'only_safe_receipt_functions','exec'),env)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'failure.json';env['write'](p,dict(loss=float('nan'),gradient=[float('inf'),-float('inf')],checks={'finite':False}));d=json.loads(p.read_text());assert d['loss']=={'nonfinite':'nan'} and d['checks']['finite'] is False
checks['NaN_Inf_receipt_written_without_invalid_JSON']=True
# Compile any remote Python command template; this executes no MCP or template.
for node in ast.walk(ast.parse((S/'operations.py').read_text())):
 if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('from pathlib import Path\n'):compile(node.value,'unexecuted_remote_template','exec')
checks['remote_template_syntax']=True
assert not (Path('/workspace/patientphex-runs/round2-20261023/research/public-h5-train32-probe-readiness')/'activation.json').exists();checks['activation_absent']=True
for name in ['probe.py','loss.py','bounded_driver.py']:
 assert 'evaluate_report' not in (S/name).read_text() and 'save_weights(' not in (S/name).read_text()
checks['no_quality_or_probe_checkpoint_entry']=True
result=dict(status='PASS_STATIC_AND_EXISTING_DATA_CONTRACT_ONLY',checks=checks,plan=ref(S/'plan.json'),sampling=ref(S/'sampling.json'),source=ref(Path(__file__).resolve()),GPU_executed=False,TF_imported=False,remote_called=False,quality_performed=False)
with (S/'static-checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),plan=result['plan']['sha256'])))
