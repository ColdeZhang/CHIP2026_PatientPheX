#!/usr/bin/env python3
"""Actual frozen tokenizer/mask, CPU only; no cap truncation or model loading."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoTokenizer
import qwen35_smoke as helper
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--manifest',required=True);p.add_argument('--old-audit',required=True);p.add_argument('--source',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
start=time.monotonic(); assert transformers.__version__=='5.20.2'; torch.set_num_threads(1); assert not torch.cuda.is_initialized()
assert sha(helper.__file__)=='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
assert sha(a.source)=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
m=json.loads(Path(a.manifest).read_text()); assert sha(a.train)==m['artifacts']['train.jsonl']['sha256']; rows=[json.loads(x) for x in Path(a.train).read_text().splitlines()]; assert len(rows)==m['stats']['combined']['rows']==212
source=json.loads(Path(a.source).read_text()); files={r['name']:r for r in source['files']}; actual={}
for name in helper.CPU_FILES:
 f=Path(a.model)/name; actual[name]=dict(bytes=f.stat().st_size,sha256=sha(f)); assert actual[name]==dict(bytes=files[name]['size'],sha256=files[name]['sha256'])
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False); audit=[]
for row in rows:
 assert [v['role'] for v in row['messages']]==['system','user']; en=helper.encode(row,tok)
 audit.append(dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),full_tokens=len(en['input_ids']),prefix_tokens=en['prefix_tokens'],supervised_tokens=en['supervised_tokens'],eos_token_id=en['eos_token_id'],rendered_prefix_sha256=en['rendered_prefix_sha256'],rendered_full_sha256=en['rendered_full_sha256'],input_ids_sha256=hashlib.sha256(json.dumps(en['input_ids']).encode()).hexdigest(),labels_sha256=hashlib.sha256(json.dumps(en['labels']).encode()).hexdigest()))
old=json.loads(Path(a.old_audit).read_text()); assert old['train_sha256']==m['old_train']['sha256'] and audit[:124]==old['rows']; assert not torch.cuda.is_initialized()
def stats(a):
 ls=sorted(r['full_tokens'] for r in a)
 return dict(rows=len(a),tokens=sum(ls),prefix_tokens=sum(r['prefix_tokens'] for r in a),supervised_tokens=sum(r['supervised_tokens'] for r in a),minimum=min(ls),median=statistics.median(ls),p90=ls[int((len(ls)-1)*.90)],p95=ls[int((len(ls)-1)*.95)],maximum=max(ls),longest_row=max(a,key=lambda r:r['full_tokens'])['row_id'],exceed_counts={str(cap):sum(n>cap for n in ls) for cap in [5523,6144,6400,7168,8192,14000]})
out=dict(status='actual_CPU_token_mask_EOS_complete_no_model_no_GPU_no_training',source_sha256=sha(__file__),preparation_sha256=sha(a.manifest),train_sha256=sha(a.train),old_audit_sha256=sha(a.old_audit),source_manifest_sha256=sha(a.source),helper_sha256=sha(helper.__file__),tokenizer_files=actual,runtime=dict(torch=torch.__version__,transformers=transformers.__version__),cuda_initialized=False,old124_all_encoded_fields_exact=True,rows=audit,groups={'old':stats(audit[:124]),'new':stats(audit[124:]),'combined':stats(audit)},truncated=0,prompt_mask_and_assistant_EOS_exact=True,projected_epochs=2,projected_steps_per_epoch=(len(rows)+7)//8,projected_optimizer_calls=2*((len(rows)+7)//8),elapsed_seconds=time.monotonic()-start)
with Path(a.output).open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','tokenizer_files']},indent=2))
