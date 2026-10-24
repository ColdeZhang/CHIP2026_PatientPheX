"""CPU cached tokenizer: confirm old actual counts, new cap, no model weights/GPU."""
import pathlib,json,hashlib,time,os
P=pathlib.Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic()
import torch,transformers
from transformers import AutoTokenizer
assert transformers.__version__=='5.20.2';torch.set_num_threads(1);assert not torch.cuda.is_initialized()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();cfg=json.loads((P/'cpu-plan.json').read_text());model=pathlib.Path(cfg['model']);manifest=json.loads(pathlib.Path(cfg['model_source']).read_text());files={x['name']:x for x in manifest['files']};actual={}
for n in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt']:
 f=model/n;actual[n]={'bytes':f.stat().st_size,'sha256':sha(f)};assert actual[n]=={'bytes':files[n]['size'],'sha256':files[n]['sha256']}
for n,h in cfg['source_sha256'].items():assert sha(P/n)==h
new=[json.loads(x) for x in (P/'prompts.jsonl').read_text().splitlines()];old=[json.loads(x) for x in (P/'old-prompts.jsonl').read_text().splitlines()];oldactual=json.loads((P/'old-actual-tokens.json').read_text());assert len(new)==len(old)==80
tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True);audit=[]
for before,after in zip(old,new):
 assert before['row_id']==after['row_id'];vals=[]
 for row in [before,after]:
  rendered=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False);vals.append((len(ids),hashlib.sha256(rendered.encode()).hexdigest(),hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
 assert vals[0][0]==oldactual[before['row_id']];assert vals[1][0]<=14000,(before['row_id'],vals[1][0]);audit.append({'row_id':after['row_id'],'pmc_id':str(after['pmc_id']),'old_input_tokens':vals[0][0],'input_tokens':vals[1][0],'added_tokens':vals[1][0]-vals[0][0],'rendered_prefix_sha256':vals[1][1],'input_ids_sha256':vals[1][2]})
assert not torch.cuda.is_initialized();d={'status':'actual_CPU_token_audit_no_weights_no_GPU','prompts':80,'concepts':587,'patient_concept_pairs':1557,'documents':16,'truncated':0,'input_cap':14000,'output_cap':1600,'max_tokens':max(x['input_tokens'] for x in audit),'old_max_tokens':max(x['old_input_tokens'] for x in audit),'total_tokens':sum(x['input_tokens'] for x in audit),'old_total_tokens':sum(x['old_input_tokens'] for x in audit),'old_actual_counts_exact':True,'rows':audit,'tokenizer_files':actual,'prompts_sha256':sha(P/'prompts.jsonl'),'old_prompts_sha256':sha(P/'old-prompts.jsonl'),'source_sha256':sha(__file__),'cpu_plan_sha256':sha(P/'cpu-plan.json'),'runtime':{'torch':torch.__version__,'transformers':transformers.__version__},'cuda_initialized':False,'seconds':time.monotonic()-start};(P/'token-audit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ['rows','tokenizer_files']}))
