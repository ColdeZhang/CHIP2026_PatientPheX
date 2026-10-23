#!/usr/bin/env python3
"""Fixed98 teacher-forced continuation scores, no cache/generation/training/gold."""
from pathlib import Path
import argparse,json,gzip,hashlib,time,os,importlib.util

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--model',required=True);a.add_argument('--source-manifest',required=True);a.add_argument('--adapter',required=True);p=a.parse_args();b=p.directory;plan=json.loads((b/'remote-plan.json').read_text());assert sha(__file__)==plan['forward_sha256'];assert not (b/'scores.jsonl').exists();start=time.monotonic()
 import torch,transformers,peft
 from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
 from peft import PeftModel
 torch.set_num_threads(1);assert transformers.__version__=='5.20.2' and peft.__version__=='0.21.2';assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
 encoded=b/'selected98-encoded.json.gz';assert sha(encoded)==plan['encoded_sha256'];branches=json.loads(gzip.decompress(encoded.read_bytes()));assert len(branches)==98 and len({(x['row_id'],x['cid']) for x in branches})==12
 for x in branches:
  assert x['score_labels']==x['input_ids'][x['score_start']:] and x['logits_to_keep']==len(x['score_labels']) and len(x['input_ids'])<=4161
  assert x['subset']==[pid for i,pid in enumerate(x['patient_ids']) if x['mask']>>i&1]
 spec=importlib.util.spec_from_file_location('frozen_base_verify',str(b/'qwen35_runner.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);assert sha(b/'qwen35_runner.py')==plan['runner_sha256'];actualbase=helper.verify_model(p.model,p.source_manifest)
 adapter={n:sha(Path(p.adapter)/n) for n in ['adapter_config.json','adapter_model.safetensors']};assert adapter==plan['adapter_files']
 config=Qwen3_5TextConfig.from_pretrained(p.model,local_files_only=True)
 model,loading=Qwen3_5ForCausalLM.from_pretrained(p.model,config=config,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
 assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);assert sum(p.numel() for p in model.parameters())==8953803264
 model=PeftModel.from_pretrained(model,p.adapter,is_trainable=False);model.eval();assert sum(p.numel() for p in model.parameters())==8997081600 and not any(p.requires_grad for p in model.parameters())
 save(b/'run_manifest.json',dict(plan_sha256=sha(b/'remote-plan.json'),source_sha256=sha(__file__),encoded_sha256=sha(encoded),model_files=actualbase,adapter_files=adapter,runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_uuid=plan['gpu_uuid']),configuration=dict(dtype='bfloat16',log_softmax='float32',batch_size=1,use_cache=False,no_generate=True,no_training=True),loading=loading,trainable_parameters=0))
 branches.sort(key=lambda x:(-len(x['input_ids']),x['row_id'],x['cid'],x['mask']));torch.cuda.reset_peak_memory_stats();forward_start=time.monotonic();done=[]
 with (b/'scores.jsonl').open('x') as file,torch.inference_mode():
  for i,x in enumerate(branches):
   torch.cuda.synchronize();tick=time.monotonic();ids=torch.tensor([x['input_ids'][:-1]],device='cuda',dtype=torch.long);labels=torch.tensor(x['score_labels'],device='cuda',dtype=torch.long);K=x['logits_to_keep']
   assert ids.shape[1]-K==x['score_start']-1
   output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=K,return_dict=True);assert output.past_key_values is None and output.logits.shape[:2]==(1,K)
   lp=torch.log_softmax(output.logits[0].float(),dim=-1).gather(1,labels[:,None]).squeeze(1);assert torch.isfinite(lp).all() and (lp<=0).all();total=lp.sum();assert total.dtype==torch.float32 and torch.isfinite(total)
   row={k:x[k] for k in ['row_id','pmc_id','cid','concept','patient_ids','mask','subset','score_start','logits_to_keep','delimiter']};row.update(input_tokens=len(x['input_ids']),input_ids_sha256=hashlib.sha256(json.dumps(x['input_ids']).encode()).hexdigest(),score_label_ids=x['score_labels'],token_logprobs=lp.tolist(),sum_logprob=total.item())
   del output,lp,total,ids,labels;torch.cuda.synchronize();row['forward_seconds']=time.monotonic()-tick;file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush();done.append(row)
   if i==0:save(b/'longest-actual.json',dict(input_tokens=row['input_tokens'],score_tokens=K,finite_sum=row['sum_logprob'],forward_seconds=row['forward_seconds'],allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),continued_same_job=True))
   if i%12==0:print(json.dumps(dict(completed=i+1,total=98,elapsed_seconds=time.monotonic()-start)),flush=True)
 for key in {(x['row_id'],x['cid']) for x in branches}:
  group=[x for x in done if (x['row_id'],x['cid'])==key];assert {x['mask'] for x in group}==set(range(1<<len(group[0]['patient_ids'])))
 elapsed=time.monotonic()-start;assert elapsed<300
 save(b/'complete.json',dict(status='all98_float32_conditional_scores_complete',rows=98,CIDs=12,score_sha256=sha(b/'scores.jsonl'),manifest_sha256=sha(b/'run_manifest.json'),plan_sha256=sha(b/'remote-plan.json'),longest_sha256=sha(b/'longest-actual.json'),elapsed_seconds=elapsed,forward_seconds=time.monotonic()-forward_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),no_gold=True,no_cache=True,no_generation=True,no_training=True));print('COMPLETE98',flush=True)
if __name__=='__main__':main()
