"""One full document per step; whole-document gradients and fixed final epoch."""
from pathlib import Path
import argparse,gc,hashlib,json,random,time,os
import numpy as np
import torch
from transformers import AutoModel,get_linear_schedule_with_warmup
from common import ref,save,unpack,packed
from model import PatientSpanModel,optimizer_groups


def seed(value):
 random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def gradient_summary(model):
 groups={}
 for name,p in model.named_parameters():
  if p.grad is not None:
   assert torch.isfinite(p.grad).all(),name
   group='encoder' if name.startswith('encoder.') else name.split('.')[0]
   groups[group]=groups.get(group,0.)+float(p.grad.float().square().sum())
 assert groups['encoder']>0 and groups['condition_head']>0 and groups['patient_head']>0
 return groups

def snapshot(model):
 names=['encoder.embeddings.word_embeddings.weight','encoder.encoder.layer.0.attention.self.query.weight','ner_head.weight','condition_head.weight','patient_head.weight','patient_bias.weight'];params=dict(model.named_parameters());return {n:params[n].detach().cpu().clone() for n in names}

def changes(model,before):
 params=dict(model.named_parameters());r={n:not torch.equal(v,params[n].detach().cpu()) for n,v in before.items()};assert all(r.values()),r;return r

def one_update(model,row,opt):
 opt.zero_grad(set_to_none=True);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 with torch.autocast('cuda',dtype=torch.bfloat16):info=model.loss(row)
 assert torch.isfinite(info['loss']);info['loss'].backward();grads=gradient_summary(model);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm);opt.step();torch.cuda.synchronize()
 return {'pmc_id':row['pmc_id'],'seconds':time.monotonic()-start,'windows':len(row['windows']),'tokens':row['token_length'],'patients':len(row['patients']),'candidates':len(row['candidates']),'bags':len(row['bags']),'patient_loss_groups':info['patient_loss_groups'],'positive_bags':sum(b['label'] for b in row['bags']),'loss':float(info['loss'].detach()),'NER_loss':float(info['NER_loss'].detach()),'bag_loss':float(info['bag_loss'].detach()),'grad_squared_sum':grads,'unclipped_grad_norm':norm,'optimizer_lr':[g['lr'] for g in opt.param_groups],'geometry_table_min':float(model.geometry_table.detach().min()),'geometry_table_max':float(model.geometry_table.detach().max()),'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['probe','formal'],required=True);ap.add_argument('--model',required=True);ap.add_argument('--encoded',required=True);ap.add_argument('--output',required=True);ap.add_argument('--plan',required=True);a=ap.parse_args();O=Path(a.output);O.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seed(20261024);plan=json.loads(Path(a.plan).read_text());pre=json.loads(Path(plan['effective_preparation']).read_text());assert pre['configuration']['epochs']==plan['training']['epochs']==24;train=unpack(Path(a.encoded)/'train.json.gz');held=unpack(Path(a.encoded)/'held.json.gz');assert len(train)==32 and len(held)==16
 assets={n:ref(Path(a.model)/n) for n in ['pytorch_model.bin','config.json','vocab.txt','tokenizer_config.json']}
 assert {n:v['sha256'] for n,v in assets.items()}==plan['model_files']
 encoder,loading=AutoModel.from_pretrained(a.model,local_files_only=True,attn_implementation='eager',add_pooling_layer=False,output_loading_info=True)
 save(O/'loading.json',loading);assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'];assert all(k.startswith('cls.') or k in ['pooler.dense.weight','pooler.dense.bias'] for k in loading['unexpected_keys'])
 model=PatientSpanModel(encoder).cuda();model.train();num=sum(p.numel() for p in model.parameters());assert num<=10000000000;opt=torch.optim.AdamW(optimizer_groups(model));assert num==109187763;assert sum(p.numel() for p in opt.param_groups[1]['params'])==50
 save(O/'initialized.json',{'mode':a.mode,'train_ids':[r['pmc_id'] for r in train],'held_ids':[r['pmc_id'] for r in held],'parameters':num,'encoder_parameters':sum(p.numel() for p in encoder.parameters()),'unused_pretraining_keys':loading['unexpected_keys'],'probe_state_used':False,'effective_preparation':ref(plan['effective_preparation']),'torch':torch.__version__,'assets':assets,'plan':ref(a.plan),'source':ref(__file__),'model_source':ref(Path(__file__).with_name('model.py')),'base_model_source':ref(Path(__file__).parent/'vendor/base_model.py'),'optimizer_groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params'])} for g in opt.param_groups],'config':pre['configuration']})
 if a.mode=='probe':
  maximum_windows=max(train,key=lambda r:(len(r['windows']),len(r['candidates'])));maximum_patient_candidates=max(train,key=lambda r:len(r['patients'])*len(r['candidates']));rows={r['pmc_id']:r for r in [maximum_windows,maximum_patient_candidates]};assert set(rows)==set(plan['probe']['documents']);records=[]
  for row in rows.values():
   before=snapshot(model);r=one_update(model,row,opt);r['actual_parameter_changes']=changes(model,before);records.append(r);save(O/'updates.json',records);print(json.dumps(r),flush=True)
  seconds_per_window=max(r['seconds']/r['windows'] for r in records);estimate=seconds_per_window*(300*8+156*.5)+60+120
  save(O/'complete.json',{'status':'PASS_REAL_FULL_DOCUMENT_ENGINEERING_ONLY','updates':records,'seconds':time.monotonic()-started,'training_state_discarded':True,'formal_estimate_seconds_conservative':estimate,'estimate_rule':'max measured seconds/window * (2400 training windows + half 156 infer windows) + 60 setup/save +120 normalization allowance','original_engineering_budget_seconds':120,'parameters':num});return
 total=24*len(train);scheduler=get_linear_schedule_with_warmup(opt,num_warmup_steps=int(total*.1),num_training_steps=total);steps=0;epoch_reports=[]
 for epoch in range(1,25):
  order=list(range(len(train)));random.Random(20261024+epoch).shuffle(order);rows=[]
  for index in order:
   r=one_update(model,train[index],opt);scheduler.step();steps+=1;rows.append(r)
   save(O/'progress.json',{'epoch':epoch,'global_step':steps,'total_steps':total,'last':r,'elapsed':time.monotonic()-started})
  report={'epoch':epoch,'steps':len(rows),'seconds':sum(r['seconds'] for r in rows),'loss_mean':float(np.mean([r['loss'] for r in rows])),'NER_loss_mean':float(np.mean([r['NER_loss'] for r in rows])),'bag_loss_mean':float(np.mean([r['bag_loss'] for r in rows])),'documents':[r['pmc_id'] for r in rows],'first_lr':rows[0]['optimizer_lr'],'last_lr':rows[-1]['optimizer_lr'],'geometry_table':model.geometry_table.detach().cpu().tolist(),'max_allocated_bytes':max(r['max_allocated_bytes'] for r in rows)};epoch_reports.append(report);save(O/'epochs.json',epoch_reports);print(json.dumps(report),flush=True)
 assert steps==768;checkpoint=O/'checkpoint.pt';torch.save({'model':model.state_dict(),'epoch':24,'global_step':steps,'configuration':pre['configuration'],'train_ids':[r['pmc_id'] for r in train]},checkpoint);save(O/'checkpoint.json',ref(checkpoint))
 with checkpoint.open('rb') as handle:os.fsync(handle.fileno())
 for directory in [O,O.parent]:
  fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 reloaded=torch.load(checkpoint,map_location='cpu',weights_only=False);live=model.state_dict();loaded=reloaded['model'];assert set(live)==set(loaded);assert reloaded['epoch']==24 and reloaded['global_step']==768 and reloaded['configuration']==pre['configuration'];validated=[]
 for name,value in live.items():
  target=loaded[name];current=value.detach().cpu();assert current.shape==target.shape and current.dtype==target.dtype and torch.equal(current,target),name;validated.append({'key':name,'shape':list(target.shape),'dtype':str(target.dtype),'numel':target.numel(),'exact_value_equal':True,'sha256':hashlib.sha256(target.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})
 save(O/'checkpoint-reload.json',{'status':'FINAL_DEPLOYMENT_STATE_FSYNC_RELOAD_ALL_TENSORS_EXACT','checkpoint':ref(checkpoint),'file_fsynced':True,'directory_fsync_paths':[str(O),str(O.parent)],'keys_shapes_dtypes_values_all_equal':True,'tensors':validated,'optimizer_or_rng_saved':False,'extra_forward':False,'comparison':'CPU torch.load versus live model.state_dict immediately after final save'})
 del reloaded,loaded,live,current,target;del opt,scheduler;gc.collect();torch.cuda.empty_cache();model.eval();preds=[]
 for row in held:
  start=time.monotonic()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):records=model.infer(row,floor=-2.)
  payload={'pmc_id':row['pmc_id'],'patient_ids':[p['patient_id'] for p in row['patients']],'records':records,'all_candidate_count':len(row['candidates']),'global_logit_floor':-2.,'patient_logit_floor':None,'token_ids_sha256':row['token_ids_sha256']};r=packed(O/('raw-'+row['pmc_id']+'.json.gz'),payload);r.update(pmc_id=row['pmc_id'],records=len(records),seconds=time.monotonic()-start);preds.append(r);save(O/'inference-progress.json',preds)
 assert len(preds)==16;save(O/'complete.json',{'status':'COMPLETE_FRESH24EPOCH_FULL_F1_RAW','epochs':epoch_reports,'global_step':steps,'seconds':time.monotonic()-started,'predictions':preds,'checkpoint':ref(checkpoint),'all_patients':sum(len(r['patients']) for r in held),'all_documents':len(held),'no_quality':True,'raw_floor':-2.,'joint_decoder':{'globalNER_gt':0.,'patient_logit_gt':0.}})
if __name__=='__main__':main()
