#!/usr/bin/env python3
"""One fresh bounded real-row native/weighted CE probe; never saves model state."""
from pathlib import Path
import argparse,json,time,os,hashlib
import slot_model as common
import loss as weighted

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--source-manifest',required=True);a=p.parse_args();b=a.directory;plan=json.loads((b/'remote-plan.json').read_text());started=time.monotonic()
 assert not (b/'run_manifest.json').exists() and os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
 for name,value in plan['source_sha256'].items():assert common.sha(b/name)==value
 import torch
 import qwen35_relation_cost as protocol
 import qwen35_smoke as helper
 from transformers import AutoTokenizer
 rows=[json.loads(l) for l in (b/'selected-rows.jsonl').read_text().splitlines()];records=json.loads((b/'selected-weights.json').read_text());assert [x['row_id'] for x in rows]==plan['row_ids']==[x['row_id'] for x in records]
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);encoded=[]
 for row,w in zip(rows,records):
  en=helper.encode(row,tok);assert len(en['input_ids'])==w['full_tokens'];assert hashlib.sha256(json.dumps(en['input_ids']).encode()).hexdigest()==w['input_ids_sha256'];assert hashlib.sha256(json.dumps(en['labels']).encode()).hexdigest()==w['labels_sha256'];assert en['labels']==[-100]*en['prefix_tokens']+w['targets'];assert w['positions']==list(range(en['prefix_tokens']-1,len(en['input_ids'])-1));assert len(w['weights'])==len(en['input_ids']);encoded.append((en,w))
 def batch(en,w,pad=0):
  ids=torch.tensor([en['input_ids']+[tok.pad_token_id]*pad],device='cuda');labels=torch.tensor([en['labels']+[-100]*pad],device='cuda');weights=torch.tensor([w['weights']+[0]*pad],device='cuda');mask=torch.ones_like(ids)
  if pad:mask[:,-pad:]=0
  return dict(input_ids=ids,labels=labels,attention_mask=mask),weights
 model,named,manifest=common.load_fresh(a.model,a.source_manifest,20261023);manifest.update(plan_sha256=common.sha(b/'remote-plan.json'),probe_source_sha256=common.sha(__file__),loss_source_sha256=common.sha(weighted.__file__))
 common.save(b/'run_manifest.json',manifest);model.eval();short,w=encoded[0];data,weights=batch(short,w);idx=torch.where(data['labels'][0]!=-100)[0];targets=data['labels'][0,idx];rowweights=weights[0,idx].float()
 full=model(**data,use_cache=False,logits_to_keep=0);reference_logits=full.logits[0,idx-1,:].float();ce=torch.nn.functional.cross_entropy(reference_logits,targets);native=full.loss
 assert torch.allclose(native,ce,rtol=1e-5,atol=1e-5)
 weighted_reference=(torch.nn.functional.cross_entropy(reference_logits,targets,reduction='none')*rowweights).sum()/rowweights.sum();weighted_reference.backward()
 native_value=float(native.detach());reference_value=float(weighted_reference.detach());ce_value=float(ce.detach());grads={n:p.grad.detach().cpu().clone() for n,p in named.items()};logits_cpu=reference_logits.detach().cpu();del full,reference_logits,native,ce,weighted_reference;model.zero_grad(set_to_none=True)
 selected=weighted.row_loss(model,data,weights,torch);assert abs(float(selected.detach())-reference_value)<=plan['tolerances']['loss_atol']+plan['tolerances']['loss_rtol']*abs(reference_value);selected.backward()
 difference=0.;denominator=0.;family={k:dict(difference_sq=0.,reference_sq=0.) for k in ['DeltaNet','full_attention','MLP']}
 for n,p in named.items():
  assert p.grad is not None and torch.isfinite(p.grad).all();g=p.grad.detach().cpu();d=float((g-grads[n]).float().square().sum());z=float(grads[n].float().square().sum());difference+=d;denominator+=z;family[protocol.family(n)]['difference_sq']+=d;family[protocol.family(n)]['reference_sq']+=z
 relative=(difference/max(denominator,1e-30))**.5;assert relative<=plan['tolerances']['gradient_relative_l2']
 selected_value=float(selected.detach());del selected,grads;model.zero_grad(set_to_none=True)
 with torch.inference_mode():
  out=model(input_ids=data['input_ids'],attention_mask=data['attention_mask'],use_cache=False,logits_to_keep=idx-1).logits[0].float().cpu();assert torch.allclose(out,logits_cpu,rtol=plan['tolerances']['logits_rtol'],atol=plan['tolerances']['logits_atol']);unweighted=torch.nn.functional.cross_entropy(out,targets.cpu());assert abs(float(unweighted)-native_value)<=plan['tolerances']['loss_atol']+plan['tolerances']['loss_rtol']*abs(native_value)
  padded,padweights=batch(short,w,8);padloss=weighted.row_loss(model,padded,padweights,torch);assert abs(float(padloss)-selected_value)<=plan['tolerances']['loss_atol']+plan['tolerances']['loss_rtol']*abs(selected_value)
 report=dict(row_id=short['row_id'],tokens=len(short['input_ids']),supervised=len(idx),native_full_CE=native_value,full_logits_hand_CE=ce_value,selected_all1_CE=float(unweighted),weighted_full_hand_CE=reference_value,weighted_selected_CE=selected_value,weighted_padded_CE=float(padloss),logits_max_abs_diff=float((out-logits_cpu).abs().max()),gradient_relative_L2=relative,family_gradient_comparison=family,padding=8,tolerances=plan['tolerances'],causal_positions='target j predicted by logit j-1',actual_elapsed_seconds=time.monotonic()-started)
 common.save(b/'short-equivalence.json',report);del data,weights,idx,targets,rowweights,logits_cpu,out,padded,padweights,padloss;torch.cuda.empty_cache()
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train();optimizer=torch.optim.AdamW(list(named.values()),lr=1e-4,weight_decay=.01);before={}
 for name,p in named.items():
  if '.lora_B.' in name and protocol.family(name) not in before:before[protocol.family(name)]=(name,p,p.detach().clone())
 assert set(before)=={'DeltaNet','full_attention','MLP'};head=model.get_base_model().lm_head;head_before=head.weight[[15,16]].detach().clone();long,w=encoded[1];data,weights=batch(long,w);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=weighted.row_loss(model,data,weights,torch)
 assert torch.isfinite(loss);loss.backward();norms={x:0. for x in before}
 for n,p in named.items():
  assert p.grad is not None and torch.isfinite(p.grad).all();norms[protocol.family(n)]+=float(p.grad.detach().float().square().sum())
 assert all(v>0 for v in norms.values());norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.);assert torch.isfinite(norm) and norm>0;optimizer.step();assert all(not torch.equal(old,p.detach()) for _,p,old in before.values());assert all(p.grad is None for p in model.parameters() if not p.requires_grad);assert head.weight.grad is None and torch.equal(head_before,head.weight[[15,16]].detach());assert all(torch.isfinite(v).all() for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v));torch.cuda.synchronize()
 result=dict(row_id=long['row_id'],input_tokens=len(long['input_ids']),supervised=long['supervised_tokens'],loss=float(loss.detach()),step_seconds=time.monotonic()-tick,gradient_norm=float(norm),family_gradient_squared_norms=norms,changed_LoRA_B={k:v[0] for k,v in before.items()},head_frozen=True,learning_rate=1e-4,optimizer_state_dtypes=sorted({str(v.dtype) for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)}),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0]);common.save(b/'longest-step.json',result)
 assert time.monotonic()-started<180;common.save(b/'complete.json',dict(status='PASS_actual_weighted_JSON_CE_probe_state_discarded',elapsed_seconds=time.monotonic()-started,manifest_sha256=common.sha(b/'run_manifest.json'),short_sha256=common.sha(b/'short-equivalence.json'),longest_sha256=common.sha(b/'longest-step.json'),completed_epoch=time.time(),no_checkpoint=True,no_quality=True));print(json.dumps(dict(complete=True,longest=result)),flush=True)
if __name__=='__main__':main()
