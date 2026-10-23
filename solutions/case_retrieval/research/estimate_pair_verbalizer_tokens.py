"""CPU-only planning on original train48 messages; no model weights or heldout gold."""
from pathlib import Path
import hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import importlib.util,tokenizers
assert importlib.util.find_spec('torch') is None, 'Planning environment expected to have no Torch installed'
from tokenizers import Tokenizer
B=Path(__file__).parent;Q=B.parents[1]
source=Q/'llm-tree-cpu-preparation/train204.prompts.jsonl';raw=Q/'llm-tree-base204/base204.responses.jsonl';model=Q/'verified-small'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()];known={x['row_id']:x for x in map(json.loads,raw.read_text().splitlines())}
def encode(messages):
 assert [m['role'] for m in messages]==['system','user'] and all(isinstance(m['content'],str) for m in messages)
 prompt=''.join('<|im_start|>'+m['role']+'\n'+m['content'].strip()+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
 return tokenizer.encode(prompt,add_special_tokens=False).ids
assert len(rows)==204 and all('answer' not in x for x in rows)
assert all(len(encode(r['messages']))==known[r['row_id']]['input_tokens'] for r in rows)
system='Decide whether the supplied target patient is actually observed to have the supplied candidate clinical phenotype in this report. Use supplied patient identities, HPO meaning and evidence. Answer yes or no only. Reject negated or normal findings, hypothetical possibilities, general syndrome descriptions and other cited cases. Do not infer symptoms from a diagnosis. A supplied target can be a relative; report that patient\'s own findings.'
counts=[];naive_repeated=0;pair_count=0;start=time.time()
for row in rows:
 user=row['messages'][-1]['content'];anchors,candidates=user.split('\n\nCandidate phenotypes:\n',1);candidates=json.loads(candidates)
 assert {c['code'] for c in candidates}==set(row['code_map'])
 naive_repeated+=known[row['row_id']]['input_tokens']*len(candidates)*len(row['patient_ids'])
 for candidate in candidates:
  for patient in row['patient_ids']:
   messages=[{'role':'system','content':system},{'role':'user','content':anchors+'\n\nTarget patient ID: '+patient+'\n\nCandidate phenotype:\n'+json.dumps(candidate,ensure_ascii=False)+'\n\nDoes this target patient have this phenotype?'}]
   ids=encode(messages);counts.append({'row_id':row['row_id'],'code':candidate['code'],'patient_id':patient,'tokens':len(ids)});pair_count+=1
assert pair_count==4127
summary={'scope':'CPU tokenizer-only planning; originaltrain48 204 label-free messages only, no originaldev messages/gold, no weights or GPU','source_sha256':sha(source),'old_raw_sha256':sha(raw),'estimator_source_sha256':sha(__file__),'transformers':None,'template_implementation':'Exact frozen simple-text system+user branch; no tools/vision;204 actual prefix lengths independently match','tokenizers':tokenizers.__version__,'old204_prefix_tokens_reproduced_exactly':sum(known[r['row_id']]['input_tokens'] for r in rows),'pairs':pair_count,'single_concept_all_global_anchors_tokens':sum(x['tokens'] for x in counts),'single_concept_max_tokens':max(x['tokens'] for x in counts),'single_concept_min_tokens':min(x['tokens'] for x in counts),'naive_fullbatch_prefix_repeated_per_pair_tokens_before_new_question':naive_repeated,'single_concept_input_ratio_to_old204':sum(x['tokens'] for x in counts)/648409,'yes_no_standalone_token_ids':{s:tokenizer.encode(s,add_special_tokens=False).ids for s in ['yes','no',' yes',' no','true','false']},'new_formula_note':'softmax([logit_no,logit_yes])[yes] is a real conditional two-verbalizer probability, not guaranteed calibrated relation probability. No logits were computed here.','input_change_caveat':'Single-concept variant removes other candidates local contexts, retaining all global anchors and this candidate complete evidence. This changes context as well as output protocol and needs separate quality evaluation.','torch_installed':False,'cuda_initialized':False,'tokenization_seconds_after_old_exact_check':time.time()-start}
(B/'pair-token-budget.json').write_text(json.dumps(summary,indent=2)+'\n');(B/'pair-token-budget.rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in counts));print(json.dumps(summary,indent=2))
