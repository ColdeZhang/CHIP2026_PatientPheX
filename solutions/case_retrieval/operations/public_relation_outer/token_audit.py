#!/usr/bin/env python3
"""Actual cached tokenizer only; original prompt bytes, no labels, no weights/GPU."""
from pathlib import Path
import argparse,hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--prompts',required=True);p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();assert transformers.__version__=='5.20.2';torch.set_num_threads(1);assert not torch.cuda.is_initialized()
source=json.loads(Path(a.source).read_text());files={v['name']:v for v in source['files']};actual={}
for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt']:
 f=Path(a.model)/name;actual[name]={'bytes':f.stat().st_size,'sha256':sha(f)};assert actual[name]=={'bytes':files[name]['size'],'sha256':files[name]['sha256']}
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);data=[json.loads(l) for l in Path(a.prompts).read_text().splitlines()];audit=[]
assert len(data)==len({x['row_id'] for x in data})>0 and len({str(x['pmc_id']) for x in data})==16
for r in data:
 assert set(r)=={'row_id','pmc_id','code_map','patient_ids','messages'} and [m['role'] for m in r['messages']]==['system','user']
 rendered=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)<=14000
 audit.append({'row_id':r['row_id'],'pmc_id':str(r['pmc_id']),'input_tokens':len(ids),'rendered_prefix_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
assert not torch.cuda.is_initialized();report=dict(status='actual_CPU_token_audit_no_weights_no_GPU',source_sha256=sha(__file__),prompts_sha256=sha(a.prompts),model_source_sha256=sha(a.source),tokenizer_files=actual,runtime={'torch':torch.__version__,'transformers':transformers.__version__},cuda_initialized=False,rows=audit,documents=16,prompts=len(data),concepts=sum(len(r['code_map']) for r in data),patient_concept_pairs=sum(len(r['code_map'])*len(r['patient_ids']) for r in data),total_tokens=sum(r['input_tokens'] for r in audit),max_tokens=max(r['input_tokens'] for r in audit),input_cap=14000,truncated=0,elapsed_seconds=time.monotonic()-start)
with Path(a.output).open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','tokenizer_files']}))
