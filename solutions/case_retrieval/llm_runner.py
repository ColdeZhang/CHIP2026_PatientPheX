#!/usr/bin/env python3
"""Batched local Qwen inference with strict immutable prompt/config resume checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--prompts',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--max-input-tokens',type=int,default=10000);ap.add_argument('--max-new-tokens',type=int,default=1600)
    ap.add_argument('--batch-size',type=int,default=3);ap.add_argument('--model-revision',default='b968826d9c46dd6066d109eabc6255188de91218')
    args=ap.parse_args();start=time.time();torch.set_num_threads(2)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fingerprint={'configuration':vars(args),'prompts_sha256':digest(args.prompts),'code_sha256':digest(__file__),'model_config_sha256':digest(Path(args.model)/'config.json')}
    checkpoint=out.with_suffix('.checkpoint.json')
    if out.exists() and out.stat().st_size:
        assert checkpoint.exists() and json.loads(checkpoint.read_text())==fingerprint,'Resume fingerprint mismatch'
    checkpoint.write_text(json.dumps(fingerprint,indent=2)+'\n')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Need exactly one allocated GPU'
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    count=sum(p.numel() for p in model.parameters());assert count<=10_000_000_000
    rows=[json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    done={json.loads(x)['row_id'] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    rows=[x for x in rows if x.get('row_id',str(x['pmc_id'])) not in done]
    # Process similarly sized prompts together; output carries explicit row ID, so order is irrelevant.
    prepared=[]
    for row in rows:
        messages=row['messages'];removed=0
        while True:
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            length=len(tok.encode(prompt,add_special_tokens=False))
            if length<=args.max_input_tokens:break
            if len(messages)<=2:raise ValueError(f"Prompt too long; explicit rechunk required: {row.get('row_id',row['pmc_id'])}")
            messages=[messages[0]]+messages[3:];removed+=1
        prepared.append((length,row,prompt,removed))
    prepared.sort(key=lambda x:x[0])
    with out.open('a') as f:
        for begin in range(0,len(prepared),args.batch_size):
            batch=prepared[begin:begin+args.batch_size];encoded=tok([x[2] for x in batch],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda:0');width=encoded.input_ids.shape[1];began=time.time()
            with torch.inference_mode():
                generated=model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id,use_cache=True)
            for item,output in zip(batch,generated):
                length,row,prompt,removed=item;answer=tok.decode(output[width:],skip_special_tokens=True)
                record={'row_id':row.get('row_id',str(row['pmc_id'])),'pmc_id':row['pmc_id'],'response':answer,'input_tokens':length,'output_tokens':int((output[width:]!=tok.pad_token_id).sum()),'removed_demonstrations':removed,'hit_token_limit':int((output[width:]!=tok.pad_token_id).sum())>=args.max_new_tokens and int(output[-1])!=tok.eos_token_id,'batch_seconds':round(time.time()-began,3)}
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'completed':begin+len(batch),'remaining_total':len(prepared),'batch_seconds':round(time.time()-began,2),'padded_input':width}),flush=True)
    meta={'configuration':vars(args),'fingerprint':fingerprint,'parameter_count':count,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':round(time.time()-start,3),'output_sha256':digest(out)}
    out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
