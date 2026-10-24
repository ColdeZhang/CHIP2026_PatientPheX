"""Bind one full64 H51 probe to the actual native329 CPU receipt."""
from pathlib import Path
import ast,gzip,hashlib,json,random
S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
B=R/'research/patient-response-code-full64-readiness-v1'
D=B/'probe-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 D.mkdir(exist_ok=False);P=S/'probe';CPU=B/'native100-attempt2/native-terminal'
 result=json.loads((CPU/'native-result.json').read_text());assert result['status']=='PASS_NATIVE_CPU_RESPONSE_CODE_FULL64_CENSUS_NO_MODEL'
 assert not result['cuda_initialized'] and not result['model_loaded']
 a=json.loads((CPU/'native-row-audit.json').read_text());audit={x['index']:x for x in a if x['group']=='train' and x['arm']=='response_code'}
 rows=[json.loads(l) for l in (B/'train.response-code.jsonl').read_text().splitlines()]
 assert len(rows)==len(audit)==265
 order=list(range(265));random.Random(20261023+1).shuffle(order)
 indices=order[:8]+[207]+[i for i in order[8:] if i not in order[:8] and i not in [207,137]][:2]+[137]
 assert indices==[75,132,219,88,253,201,118,74,207,99,199,137] and len(set(indices))==12
 (D/'probe-input.json.gz').write_bytes(gzip.compress(json.dumps([dict(original_index=i,row=rows[i]) for i in indices],ensure_ascii=False).encode(),mtime=0))
 save(D/'selected-native-audit.json',[audit[i] for i in indices])
 plan=json.loads((S.parent/'probe/plan.json').read_text())
 for name in ['schema','scope','budget_basis','formal_training_authorized','authorization','limitations']:
  plan.pop(name,None)
 plan.update(schema='patient-response-code-full64-H51-v1',diagnostic_indices=indices,diagnostic_tokens=sum(audit[i]['full_tokens'] for i in indices),supervised_tokens=sum(audit[i]['supervised_tokens'] for i in indices),output_dir='/home/zhangyuheng/projects/CHIP2026_PatientPheX/patient-response-code/full64-probe-v1/probe-output',operations_source=ref(P/'operations.py'),collection_source=ref(P/'collect.py'),probe_data=ref(D/'probe-input.json.gz'),actual_native_CPU=ref(CPU/'native-result.json'),actual_native_collection=ref(CPU.parent/'native-terminal-manifest.json'),original_recipe=dict(epochs=2,optimizer_calls=68,warmup_ratio=.05,warmup_updates=4,tail_each_epoch=1,batch_size=1,accumulation=8,rank=16,seed=20261023,learning_rate=1e-4,standard_CE=True,input_rows=265,supervised_tokens=17399,changed_output_representation=True),input_sources=dict(train=ref(B/'train.response-code.jsonl'),CPU=ref(CPU/'native-result.json')),scope='One first original shuffled group8 real Adam update, then3 accumulated rows including Pareto4800/128, then Pareto6462/125; fresh base/LoRA, state discarded, no inference or quality.',budget_basis='Actual12 rows52201tokens, original same-A100 indexed12row44582token153.634s is cost proxy only; fixed240+TERM30.',formal_training_authorized='Conditional root authorization: actual engineering PASS with real memory margin and projected total<=5400 permits fresh265x2/68updates plus2fullstate durableACKs; never resume probe.',authorization='Root20261024 full64; unique240+30H51 then conditionally fresh5400+30 formal and pureoriginaldev64 inference1200+30, no quality here.',limitations='Actual CPU .100Torch2.7.1 vs intended A100Torch2.6; this probe independently encodes every selected raw row and matches all nativeCPU input/label SHA before forward. Full trainer will verify all265. No claim of full-backbone bitwise identity or unchanged output representation.')
 names=['probe.py','probe_driver.py','indexed_head.py','selected_loss.py','slot_model.py','qwen35_runner.py','qwen35_relation_cost.py','qwen35_smoke.py','model-source-manifest.json']
 payload={n:ref(P/n) for n in names};payload.update({'probe-input.json.gz':ref(D/'probe-input.json.gz'),'native-result.json':ref(CPU/'native-result.json'),'selected-native-audit.json':ref(D/'selected-native-audit.json')})
 plan['payload_sources']=payload;plan['remote_files']={n:{k:v for k,v in r.items() if k!='path'} for n,r in payload.items()};prefix='/home/zhangyuheng/projects/CHIP2026_PatientPheX/patient-response-code/full64-probe-v1/'
 plan['files']={prefix+n:{k:v for k,v in r.items() if k!='path'} for n,r in payload.items()};plan['reuse_packages']=[]
 save(P/'plan.json',plan);save(D/'plan.json',plan)
 for n in names:
  if n.endswith('.py'):ast.parse((P/n).read_text())
 old=ast.parse((S.parent/'probe/probe.py').read_text());new=ast.parse((P/'probe.py').read_text())
 def indexed_call(t):return [ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='indexed_causal_loss']
 assert indexed_call(old)==indexed_call(new)
 save(P/'source-checks.json',dict(status='SOURCE_SYNTAX_AND_INDEXED_LOSS_CALL_EXACT',plan=ref(P/'plan.json'),indices=indices,diagnostic_tokens=plan['diagnostic_tokens'],supervised_tokens=plan['supervised_tokens'],pareto_both_covered=True,first8_real_Adam_before_remaining4=True,all_data_refs_actual=True,train_guard=6656,future_warmup_ratio=.05,future_warmup_updates=4,probe_executed=False))
 print(json.dumps(dict(plan=ref(P/'plan.json'),diagnostic_tokens=plan['diagnostic_tokens'],indices=indices)))
if __name__=='__main__':main()
