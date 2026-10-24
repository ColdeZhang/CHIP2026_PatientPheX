#!/usr/bin/env python3
"""Independent attempt2 same-shape real-row native/weighted CE probe; never saves model state."""
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
 common.save(b/'run_manifest.json',manifest);model.eval();short,w=encoded[0];shape_reports=[]
 def compare_gradients(reference):
  numerator=0.;denominator=0.;family={k:dict(difference_sq=0.,reference_sq=0.) for k in ['DeltaNet','full_attention','MLP']}
  for name,param in named.items():
   assert param.grad is not None and torch.isfinite(param.grad).all();actual=param.grad.detach().cpu();d=float((actual-reference[name]).float().square().sum());z=float(reference[name].float().square().sum());numerator+=d;denominator+=z;family[protocol.family(name)]['difference_sq']+=d;family[protocol.family(name)]['reference_sq']+=z
  return dict(relative_L2=(numerator/max(denominator,1e-30))**.5,families=family)
 for pad in [0,8]:
  data,weights=batch(short,w,pad);idx=torch.where(data['labels'][0]!=-100)[0];targets=data['labels'][0,idx];rowweights=weights[0,idx].float();ones=(data['labels']!=-100).float();full=model(**data,use_cache=False,logits_to_keep=0);reference_logits=full.logits[0,idx-1,:].float();hand1=torch.nn.functional.cross_entropy(reference_logits,targets);hand2=(torch.nn.functional.cross_entropy(reference_logits,targets,reduction='none')*rowweights).sum()/rowweights.sum()
  report=dict(shape_tokens=len(short['input_ids'])+pad,pad=pad,native_full_CE=float(full.loss.detach()),full_logits_hand_CE=float(hand1.detach()),weighted_full_hand_CE=float(hand2.detach()),tolerances=plan['tolerances'])
  common.save(b/f'shape{pad}-native-values.json',report)
  assert torch.allclose(full.loss,hand1,rtol=1e-5,atol=1e-5)
  if pad:
   # Same frozen full logits, same shape; poison only predictors whose pad targets are ignored.
   frozen=full.logits.detach().float();poisoned=frozen.clone();padpred=torch.arange(len(short['input_ids'])-1,poisoned.shape[1]-1,device='cuda');poisoned[:,padpred,0]+=7.;poisoned[:,padpred,1]-=11.
   def frozen_loss_and_grad(value):
    value=value.detach().requires_grad_(True);per=torch.nn.functional.cross_entropy(value[:,:-1,:].reshape(-1,value.shape[-1]),data['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(1,-1);loss=(per*weights[:,1:]).sum()/weights[:,1:].sum();gradient=torch.autograd.grad(loss,value)[0];return loss.detach(),gradient
   cleanloss,cleangrad=frozen_loss_and_grad(frozen);dirtyloss,dirtygrad=frozen_loss_and_grad(poisoned);poison=dict(padding_targets=pad,predictor_positions=padpred.tolist(),loss_clean=float(cleanloss),loss_poisoned=float(dirtyloss),loss_exact=torch.equal(cleanloss,dirtyloss),gradient_exact=torch.equal(cleangrad,dirtygrad),pad_gradient_zero=bool((dirtygrad[:,padpred,:]==0).all()))
   common.save(b/'frozen-logits-pad-poison.json',poison);assert poison['loss_exact'] and poison['gradient_exact'] and poison['pad_gradient_zero'];del frozen,poisoned,cleanloss,cleangrad,dirtyloss,dirtygrad
  full.loss.backward(retain_graph=True);grad1={n:p.grad.detach().cpu().clone() for n,p in named.items()};model.zero_grad(set_to_none=True);hand2.backward();grad2={n:p.grad.detach().cpu().clone() for n,p in named.items()};logits_cpu=reference_logits.detach().cpu();del full,reference_logits,hand1,hand2;model.zero_grad(set_to_none=True)
  value1=weighted.row_loss(model,data,ones,torch);report['selected_all1_CE']=float(value1.detach());value1.backward();report['gradient_w1']=compare_gradients(grad1);del grad1,value1;model.zero_grad(set_to_none=True)
  common.save(b/f'shape{pad}-w1-values-gradients.json',report)
  assert abs(report['selected_all1_CE']-report['native_full_CE'])<=plan['tolerances']['loss_atol']+plan['tolerances']['loss_rtol']*abs(report['native_full_CE']);assert report['gradient_w1']['relative_L2']<=plan['tolerances']['gradient_relative_l2']
  value2=weighted.row_loss(model,data,weights,torch);report['weighted_selected_CE']=float(value2.detach());value2.backward();report['gradient_w2']=compare_gradients(grad2);del grad2,value2;model.zero_grad(set_to_none=True)
  common.save(b/f'shape{pad}-w2-values-gradients.json',report)
  assert abs(report['weighted_selected_CE']-report['weighted_full_hand_CE'])<=plan['tolerances']['loss_atol']+plan['tolerances']['loss_rtol']*abs(report['weighted_full_hand_CE']);assert report['gradient_w2']['relative_L2']<=plan['tolerances']['gradient_relative_l2']
  with torch.inference_mode():
   selected=model(input_ids=data['input_ids'],attention_mask=data['attention_mask'],use_cache=False,logits_to_keep=idx-1).logits[0].float().cpu();report['logits_max_abs_diff']=float((selected-logits_cpu).abs().max());report['logits_allclose']=torch.allclose(selected,logits_cpu,rtol=plan['tolerances']['logits_rtol'],atol=plan['tolerances']['logits_atol'])
  common.save(b/f'shape{pad}-complete.json',report);assert report['logits_allclose'];shape_reports.append(report);del data,weights,idx,targets,rowweights,ones,logits_cpu,selected;torch.cuda.empty_cache()
 comparison=dict(shapes=shape_reports,original_failed_attempt_preserved=True,native_full_pad_minus_unpad=shape_reports[1]['native_full_CE']-shape_reports[0]['native_full_CE'],weighted_full_pad_minus_unpad=shape_reports[1]['weighted_full_hand_CE']-shape_reports[0]['weighted_full_hand_CE'],cross_shape_difference_reported_not_an_equivalence_assertion=True,elapsed_seconds=time.monotonic()-started)
 common.save(b/'short-equivalence.json',comparison)
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
 assert time.monotonic()-started<180;common.save(b/'complete.json',dict(status='PASS_actual_weighted_JSON_CE_same_shape_attempt2_state_discarded',elapsed_seconds=time.monotonic()-started,manifest_sha256=common.sha(b/'run_manifest.json'),short_sha256=common.sha(b/'short-equivalence.json'),longest_sha256=common.sha(b/'longest-step.json'),completed_epoch=time.time(),no_checkpoint=True,no_quality=True));print(json.dumps(dict(complete=True,longest=result)),flush=True)
if __name__=='__main__':main()
