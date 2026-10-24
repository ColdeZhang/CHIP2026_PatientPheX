"""One actual longest-row native update plus two cache steps; never a quality run.

No GPU execution is authorized by this source. Root chooses one card/budget after
exact CPU and complete asset readiness. Probe state is discarded; formal is fresh.
"""
from pathlib import Path
import argparse,hashlib,inspect,json,os,random,subprocess,time,sys
import train_medgemma32 as training
import medgemma_interface as interface
helpers=training.helpers

def run(a):
 out=Path(a.output_dir)
 if out.exists() and any(out.iterdir()):raise ValueError('Fresh unique probe output required')
 out.mkdir(parents=True,exist_ok=True);start=time.time();stage='source_and_data'
 def save(n,o):helpers.write(out/n,o)
 try:
  for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
  os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise ValueError('Require exactly allocated UUID in CUDA_VISIBLE_DEVICES')
  rows,train_ids,excluded=training.verify_sources_and_data(a)
  assets=interface.model_files(a.model,a.model_source_manifest,True)
  import numpy as np
  import torch,transformers,peft
  from transformers import AutoTokenizer,AutoConfig,Gemma3ForConditionalGeneration
  if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':raise ValueError('Use fixed cached native runtime')
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
  if any(x.split(',')[0].strip()==a.gpu_uuid for x in apps.splitlines()):raise ValueError('Allocated card occupied')
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise ValueError('Require single BF16 GPU')
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  random.seed(training.CONFIG['seed']);np.random.seed(training.CONFIG['seed']);torch.manual_seed(training.CONFIG['seed']);torch.cuda.manual_seed_all(training.CONFIG['seed']);torch.backends.cudnn.benchmark=False
  tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,padding_side='right')
  encoded=[interface.encode(r,tok) for r in rows]
  cpu=json.loads(Path(a.cpu_reference).read_text());complete=json.loads((Path(a.cpu_reference).parent/'cpu_complete.json').read_text())
  if cpu['source_sha256']!=helpers.sha(inspect.getfile(training)) or complete['source_sha256']!=cpu['source_sha256'] or complete['cuda_initialized'] or not complete['not_training']:
   raise ValueError('Need complete same-source CPU/meta audit')
  if cpu['encoded_sha256']!=helpers.json_sha(encoded):raise ValueError('Actual token/mask differs from CPU freeze')
  selected=max(range(len(encoded)),key=lambda i:len(encoded[i]['input_ids']))
  e=encoded[selected];r=rows[selected]
  if len(e['input_ids'])>training.CONFIG['max_seq_length']:raise ValueError('Longest row exceeds rejection cap')
  cfg=AutoConfig.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.text_config.use_cache=False
  save('selection.json',{'row_id':e['row_id'],'pmc_id':r['pmc_id'],'full_tokens':len(e['input_ids']),'prefix_tokens':e['prefix_tokens'],'supervised_tokens':e['supervised_tokens'],
       'selection_rule':'first maximum full length among exact124 training rows, independent of correctness','encoded_sha256':cpu['encoded_sha256'],'assets':assets,
       'probe_source_sha256':helpers.sha(__file__),'trainer_source_sha256':helpers.sha(inspect.getfile(training)),'modeling_source_sha256':helpers.sha(inspect.getsourcefile(Gemma3ForConditionalGeneration)),
       'cuda_uuid':a.gpu_uuid,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__})
  stage='native_load';torch.cuda.reset_peak_memory_stats();load_start=time.perf_counter()
  base,loading=Gemma3ForConditionalGeneration.from_pretrained(a.model,config=cfg,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',
       generation_config=interface.greedy_config(transformers),local_files_only=True,trust_remote_code=False,output_loading_info=True)
  save('loading-info.json',loading)
  if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Incomplete native full bundle load')
  model,named,coverage=interface.attach_adapter(base,torch,peft);guards=interface.forbid_vision_forward(base)
  save('model-coverage.json',coverage);load_seconds=time.perf_counter()-load_start
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
  if not model.is_gradient_checkpointing:raise ValueError('Gradient checkpointing inactive')
  params=list(named.values());optim=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
  probes={}
  for n,p in named.items():
   if '.lora_B.' in n and interface.family(n) not in probes:probes[interface.family(n)]=(n,p,p.detach().clone())
  batch=training.collate([e],tok.pad_token_id,torch);stage='longest_native_update';torch.cuda.synchronize();update_start=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**batch,use_cache=False)
  native=output.loss
  active=batch['labels'][:,1:]!=-100
  selected_logits=output.logits[:,:-1,:].detach()[active].float();selected_labels=batch['labels'][:,1:][active]
  manual=torch.nn.functional.cross_entropy(selected_logits,selected_labels,reduction='mean')
  if not bool(torch.isfinite(native)) or not torch.allclose(native.detach().float(),manual,rtol=1e-5,atol=1e-5):raise ValueError('Same-logits assistant CE differs')
  native_value=float(native.detach());manual_value=float(manual);del selected_logits,selected_labels,manual
  native.backward()
  norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not bool(torch.isfinite(norm)) or float(norm)==0:raise ValueError('Nonfinite/zero LoRA gradient')
  if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen bundle/head/embed received gradient')
  families={interface.family(n) for n,p in named.items() if p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().max()>0)}
  if families!={'attention','MLP'}:raise ValueError('Missing finite nonzero language family gradient')
  optim.step();torch.cuda.synchronize();update_seconds=time.perf_counter()-update_start
  if any(torch.equal(before,p.detach()) for _,p,before in probes.values()):raise ValueError('Diagnostic nonzero LR did not update language B probes')
  save('longest-update.json',{'native_loss':native_value,'same_logits_manual_loss':manual_value,'rtol':1e-5,'atol':1e-5,'gradient_norm_before_clip':float(norm),
       'families':sorted(families),'changed_probes':{k:v[0] for k,v in probes.items()},'frozen_base_no_grad':True,'visual_forward_calls':0,
       'update_seconds':update_seconds,'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved(),
       'diagnostic_learning_rate':1e-4,'optimizer_dtype':sorted({str(p.dtype) for p in params}), 'not_formal_first_step':True})
  del output,native,batch,optim,probes;model.zero_grad(set_to_none=True);model.eval();torch.cuda.empty_cache()
  stage='native_two_step_cache';_,prefix=interface.render_prefix(r,tok);ids=torch.tensor([prefix],device='cuda:0');cache_start=time.perf_counter()
  generation_calls=[]
  def observe_generation(module,args,kwargs):
   past=kwargs.get('past_key_values');past_tokens=int(past.get_seq_length()) if past is not None and hasattr(past,'get_seq_length') else None
   actual=kwargs.get('input_ids');generation_calls.append({'input_tokens':int(actual.shape[-1]) if actual is not None else None,'past_tokens':past_tokens,'cache_type':type(past).__name__ if past is not None else None})
  observer=base.register_forward_pre_hook(observe_generation,with_kwargs=True)
  with torch.inference_mode():generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=interface.greedy_config(transformers,2))
  observer.remove()
  suffix=generated[0,len(prefix):].tolist();torch.cuda.synchronize()
  if not 1<=len(suffix)<=2:raise ValueError('Native cache generation returned no suffix')
  cache_decode_observed=any(c['past_tokens'] is not None and c['past_tokens']>0 for c in generation_calls)
  if len(suffix)==2 and not cache_decode_observed:raise ValueError('Two-token native generation did not reuse populated cache')
  save('cache-two-token.json',{'generated_tokens':len(suffix),'generated_ids':suffix,'actual_forward_calls':generation_calls,'cache_decode_observed':cache_decode_observed,'first_token_EOS_only_prefill_possible':len(suffix)==1,'seconds':time.perf_counter()-cache_start,'no_quality_or_JSON_claim':True,'greedy_config':interface.greedy_config(transformers,2).to_dict()})
  save('complete.json',{'status':'engineering_probe_complete','source_sha256':helpers.sha(__file__),'elapsed_seconds':time.time()-start,'load_seconds':load_seconds,
       'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'training_rows':124,'longest_full_tokens':len(e['input_ids']),
       'no_model_or_optimizer_checkpoint_saved':True,'formal_must_fresh_start':True,'no_gold_no_quality_score':True})
 except Exception as error:
  save('failure.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),'elapsed_seconds':time.time()-start,'source_sha256':helpers.sha(__file__),'no_quality_conclusion':True});raise

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['train','excluded-input','data-manifest','model','model-source-manifest','cpu-reference','output-dir','gpu-uuid']:p.add_argument('--'+n,required=True)
 run(p.parse_args())
if __name__=='__main__':main()
