"""CPU-only actual Qwen3.5 tokenizer: no model weights, no truncation, no GPU."""
from pathlib import Path
import argparse,hashlib,json,os,statistics,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from transformers import AutoTokenizer
import transformers
import qwen35_smoke as helper
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--excluded',required=True);p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert sha(helper.__file__)=='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
assert sha(a.train)=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
assert sha(a.excluded)=='2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
assert sha(a.source)=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
assert transformers.__version__=='5.20.2';torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
rows=[json.loads(x) for x in Path(a.train).read_text().splitlines() if x.strip()];excluded=[json.loads(x) for x in Path(a.excluded).read_text().splitlines() if x.strip()]
assert len(rows)==124 and len({x['row_id'] for x in rows})==124
assert len({str(x['pmc_id']) for x in rows})==32 and not {str(x['pmc_id']) for x in rows}&{str(x['pmc_id']) for x in excluded}
assert all('entities' not in x and 'association' not in x for x in excluded)
source=json.loads(Path(a.source).read_text());expected={x['name']:x for x in source['files']};actual={}
for name in helper.CPU_FILES:
 f=Path(a.model)/name;actual[name]={'bytes':f.stat().st_size,'sha256':sha(f)}
 assert actual[name]=={'bytes':expected[name]['size'],'sha256':expected[name]['sha256']}
start=time.monotonic();tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
audit=[]
for row in rows:
 assert [m['role'] for m in row['messages']]==['system','user']
 encoded=helper.encode(row,tokenizer)
 audit.append(dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),full_tokens=len(encoded['input_ids']),prefix_tokens=encoded['prefix_tokens'],supervised_tokens=encoded['supervised_tokens'],eos_token_id=encoded['eos_token_id'],rendered_prefix_sha256=encoded['rendered_prefix_sha256'],rendered_full_sha256=encoded['rendered_full_sha256'],input_ids_sha256=hashlib.sha256(json.dumps(encoded['input_ids']).encode()).hexdigest(),labels_sha256=hashlib.sha256(json.dumps(encoded['labels']).encode()).hexdigest()))
assert not torch.cuda.is_initialized();lens=[r['full_tokens'] for r in audit]
report=dict(status='actual_cpu_token_mask_eos_audit_no_model_no_gpu_no_training',source_sha256=sha(__file__),helper_sha256=sha(helper.__file__),train_sha256=sha(a.train),excluded_sha256=sha(a.excluded),source_manifest_sha256=sha(a.source),tokenizer_files=actual,torch=torch.__version__,transformers=transformers.__version__,cuda_initialized=torch.cuda.is_initialized(),rows=audit,examples=len(audit),training_documents=32,total_tokens=sum(lens),total_prefix_tokens=sum(r['prefix_tokens'] for r in audit),supervised_tokens=sum(r['supervised_tokens'] for r in audit),minimum=min(lens),median=statistics.median(lens),max_tokens=max(lens),longest_row=max(audit,key=lambda r:r['full_tokens'])['row_id'],over6144=[r['row_id'] for r in audit if r['full_tokens']>6144],over8192=[r['row_id'] for r in audit if r['full_tokens']>8192],prefix_mask_eos_passed=True,truncated_rows=0,steps_per_epoch=(len(rows)+7)//8,epochs=2,total_optimizer_steps=2*((len(rows)+7)//8),elapsed_seconds=time.monotonic()-start)
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','tokenizer_files']}))
