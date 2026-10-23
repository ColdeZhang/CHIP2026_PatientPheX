from pathlib import Path
import json,hashlib,sys,os,statistics,ast,re
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/workspace/patientphex-runs/day1-20261023/semantic-neural/venv/lib/python3.12/site-packages')
from tokenizers import Tokenizer
q=Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation');out=q/'successor-readonly-review'
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
tok=Tokenizer.from_file(str(q/'verified-small/tokenizer.json'));tok.no_padding();tok.no_truncation()
rows=[json.loads(l) for l in (q/'full64-relation-cpu-preparation/train.jsonl').read_text().splitlines()]
counts={k:[] for k in ['old_suffix','ordered_patient_sets_suffix','ordered_binary_matrix_suffix']};positive=multi=0
for row in rows:
 obj=json.loads(row['answer']);cids=list(row['code_map']);pids=row['patient_ids']
 assert set(obj)==set(cids);assert all(set(v)<=set(pids) for v in obj.values())
 sets=[obj[k] for k in cids];matrix=[[int(p in obj[k]) for p in pids] for k in cids]
 assert {(k,p) for k,vs in zip(cids,sets) for p in vs}=={(k,p) for k,bits in zip(cids,matrix) for p,b in zip(pids,bits) if b}=={(k,p) for k,vs in obj.items() for p in vs}
 positive+=sum(map(len,sets));multi+=sum(len(v)>1 for v in sets)
 for key,answer in [('old_suffix',row['answer']),('ordered_patient_sets_suffix',json.dumps(sets,separators=(',',':'))),('ordered_binary_matrix_suffix',json.dumps(matrix,separators=(',',':')))]:counts[key].append(len(tok.encode(answer+'<|im_end|>\n',add_special_tokens=False).ids))
assert sum(counts['old_suffix'])==15410
manifests=[]
for p in sorted(q.rglob('*.manifest.json')):
 try:d=json.loads(p.read_text())
 except Exception:continue
 if d.get('base_parameters')==8953803264:
  f=d['fingerprint'];assert f['thinking'] is False
  manifests.append({'file':ref(p),'thinking':f['thinking'],'elapsed_seconds':d.get('elapsed_seconds')})
config=json.loads((q/'verified-small/tokenizer_config.json').read_text());think={k:v for k,v in config['added_tokens_decoder'].items() if v['content'] in ['<think>','</think>']}
parser=q/'relation-quality-plan/merge_pair_rag.py';fn=next(n for n in ast.parse(parser.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parse_json');ns={'json':json,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_parser','exec'),ns);parse=ns['parse_json']
closed='Reasoning {"C000":[]}\n</think>\n{"C000":["P1"]}'
unclosed='Reasoning {"C000":["P1"]}'
checks={'closed_returns_final':parse(closed)=={'C000':['P1']},'unclosed_without_open_marker_can_return_thought_json':parse(unclosed)=={'C000':['P1']},'explicit_unclosed_open_marker_rejected':parse('<think>'+unclosed) is None}
result={'scope':'CPU only on fixed train64 targets and source/manifests; no heldout labels or quality evaluation','rows':len(rows),'candidate_values':sum(len(r['code_map']) for r in rows),'possible_pairs':sum(len(r['code_map'])*len(r['patient_ids']) for r in rows),'positives':positive,'multi_patient_values':multi,'serialization_counts':{k:{'total':sum(v),'max':max(v),'median':statistics.median(v)} for k,v in counts.items()},'roundtrip_assignment_sets_exact':True,'new_prompt_instruction_not_built_or_tokenized':True,'training_or_GPU':False,'actual_qwen35_manifests':manifests,'thinking_tokens':think,'thinking_prefix_token_ids':{str(b):tok.encode('<|im_start|>assistant\n'+('<think>\n\n</think>\n\n' if not b else '<think>\n'),add_special_tokens=False).ids for b in [False,True]},'original_parser_synthetic_boundary_checks':checks,'sources':[ref(q/'full64-relation-cpu-preparation/train.jsonl'),ref(q/'verified-small/tokenizer.json'),ref(q/'verified-small/tokenizer_config.json'),ref(q/'verified-small/chat_template.jinja'),ref(q/'verified-small/README.md'),ref(parser),ref(Path(__file__))]}
p=out/'cpu-findings.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['sources','actual_qwen35_manifests']}));print('manifests',len(manifests))
