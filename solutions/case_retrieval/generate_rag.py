#!/usr/bin/env python3
"""Run Qwen3-8B association prompts on one explicitly allocated GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--prompts',required=True)
    ap.add_argument('--model-revision',default='b968826d9c46dd6066d109eabc6255188de91218')
    ap.add_argument('--output',required=True);ap.add_argument('--max-input-tokens',type=int,default=11500)
    ap.add_argument('--max-new-tokens',type=int,default=1300)
    args=ap.parse_args()
    start=time.time();torch.set_num_threads(2)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    fingerprint={'prompts_sha256':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
                 'model_revision':args.model_revision,
                 'model_config_sha256':hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest(),
                 'max_input_tokens':args.max_input_tokens,'max_new_tokens':args.max_new_tokens,
                 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checkpoint=output.with_suffix('.checkpoint.json')
    if output.exists() and output.stat().st_size:
        if not checkpoint.exists() or json.loads(checkpoint.read_text())!=fingerprint:
            raise RuntimeError('Existing responses do not match prompts/model/generation configuration')
    checkpoint.write_text(json.dumps(fingerprint,indent=2)+'\n')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one allocated visible CUDA device is required')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda:0',
                                             attn_implementation='sdpa',local_files_only=True)
    model.eval();parameter_count=sum(p.numel() for p in model.parameters())
    assert parameter_count<=10_000_000_000
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():done.add(str(json.loads(line)['pmc_id']))
    rows=[json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    with output.open('a') as f:
        for i,row in enumerate(rows):
            if str(row['pmc_id']) in done:continue
            prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=tok(prompt,return_tensors='pt',add_special_tokens=False)
            was_truncated=False
            if encoded['input_ids'].shape[1]>args.max_input_tokens:
                # Preserve instructions and the full query before sacrificing demonstrations.
                messages=[row['messages'][0],row['messages'][-1]]
                prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                encoded=tok(prompt,return_tensors='pt',add_special_tokens=False)
                was_truncated=True
            if encoded['input_ids'].shape[1]>args.max_input_tokens:
                # Explicitly marked input truncation, rather than an unrecorded overflow.
                keep=args.max_input_tokens-1200
                encoded={k:torch.cat([v[:,:keep],v[:,-1200:]],dim=1) for k,v in encoded.items()}
                was_truncated=True
            encoded={k:v.to('cuda:0') for k,v in encoded.items()};input_tokens=encoded['input_ids'].shape[1]
            began=time.time()
            with torch.inference_mode():
                generated=model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,
                                         pad_token_id=tok.eos_token_id,use_cache=True)
            answer=tok.decode(generated[0,input_tokens:],skip_special_tokens=True)
            result={'pmc_id':row['pmc_id'],'response':answer,'input_tokens':input_tokens,
                    'output_tokens':int(generated.shape[1]-input_tokens),'prompt_truncated':was_truncated,
                    'elapsed_seconds':round(time.time()-began,3)}
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'completed':i+1,'documents':len(rows),'seconds':result['elapsed_seconds'],
                              'input_tokens':input_tokens,'output_tokens':result['output_tokens']}),flush=True)
    metadata={'model':'Qwen/Qwen3-8B','parameter_count':parameter_count,'torch':torch.__version__,
              'dtype':'bfloat16','device':torch.cuda.get_device_name(0),'configuration':vars(args),'fingerprint':fingerprint,
              'elapsed_seconds':round(time.time()-start,3),'prompts_sha256':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
              'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
    output.with_suffix('.manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
