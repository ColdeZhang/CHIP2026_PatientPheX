"""Unchanged public-parent GP training, strict ancestor jobs, full candidate scores."""
import gc,gzip,json,math,os,random,time
from pathlib import Path
from common import ref,sha,save,tensor_compare,fsync_dir

def main():
 import numpy as np
 import torch,transformers
 from transformers import get_linear_schedule_with_warmup
 from model import load_model
 root=Path(__file__).resolve().parent;cfg=json.load(open(root/'formal-plan.json'));out=root/'output';out.mkdir();fsync_dir(root)
 start=time.monotonic();cpu=json.load(open(root/'cpu-complete.json'));assert cpu['status']=='PASS_NATIVE64_AND_TARGET_RUNTIME_INTERFACE_NO_CUDA'
 assert ref(root/'data.json.gz')['sha256']==cpu['data']['sha256']==cfg['data']['sha256']
 for n,r in cpu['assets'].items():
  st=(Path(cfg['model'])/n).stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns']
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()));assert len(data['jobs'])==4
 torch.set_num_threads(1);assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();torch.cuda.reset_peak_memory_stats();reports=[]
 for job in data['jobs']:
  name=job['name'];dest=out/name;dest.mkdir();fsync_dir(out);jstart=time.monotonic();train=job['train'];held=job['held'];n=len(train);total=8*n
  assert n in (32,48) and set(job['train_ids']).isdisjoint(job['input_ids']) and set(job['train_ids']).isdisjoint(cfg['F1_ids'])
  assert all('positive' not in b for r in held for b in r['blocks'])
  random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
  model=load_model(cfg['model'],dest/'loading.json');optimizer=None;scheduler=None;logs=[];steps=0;visits=[]
  if job['reuse']:
   assert name=='inner0';cp=Path(cfg['GP23_checkpoint']['path']);assert ref(cp)==cfg['GP23_checkpoint']
   from safetensors.torch import load_file
   state=json.load(open(cfg['GP23_model_manifest']['path']));assert ref(cfg['GP23_model_manifest']['path'])==cfg['GP23_model_manifest'];assert state['epochs']==8 and state['actual_adam_updates']==256
   assert set(state['visits'])=={r['row_id'] for r in train} and len(state['visits'])==256
   model.load_state_dict(load_file(str(cp)),strict=True);del state
   save(dest/'reuse.json',dict(status='READ_ONLY_GP23_MODEL_STATE_REUSE_NO_OPTIMIZER_FIT',checkpoint=ref(cp),train_ids=job['train_ids']))
  else:
   assert name in ('inner2','inner3','final48');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
   scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total),total)
   for epoch in range(8):
    model.train();order=list(range(n));random.shuffle(order);loss_sum=0.;begin=time.monotonic()
    for i in order:
     optimizer.zero_grad(set_to_none=True);row=train[i]
     with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(row)
     assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert torch.isfinite(norm)
     optimizer.step();scheduler.step();steps+=1;visits.append(row['row_id']);loss_sum+=float(loss.detach())
    assert set(order)==set(range(n));item=dict(epoch=epoch+1,actual_adam_updates=steps,documents=n,order=order,mean_document_loss=loss_sum/n,seconds=time.monotonic()-begin);logs.append(item)
    save(dest/'progress.json',item);save(out/'progress.json',dict(job=name,**item));print(json.dumps(dict(job=name,**item)),flush=True)
   assert steps==total and len(optimizer.state)==136 and all(int(s['step'].item())==total for s in optimizer.state.values())
   optimizer.zero_grad(set_to_none=True);begin=time.monotonic();nr=np.random.get_state()
   state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':{'python':random.getstate(),'numpy':(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'epochs':8,'actual_adam_updates':total,'gradients_cleared':True,'plan':ref(root/'formal-plan.json'),'visits':visits}
   cp=dest/'final-epoch8-full-state.pt'
   with cp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
   fsync_dir(dest);fsync_dir(out);saved=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,saved)
   model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
   random.setstate(saved['rng']['python']);ns=saved['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]));torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
   tensor_compare(model.state_dict(),saved['model']);tensor_compare(optimizer.state_dict(),saved['optimizer']);tensor_compare(scheduler.state_dict(),saved['scheduler']);tensor_compare(random.getstate(),saved['rng']['python']);tensor_compare(torch.get_rng_state(),saved['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),saved['rng']['cuda']);nr=np.random.get_state();tensor_compare((nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),saved['rng']['numpy'])
   save(dest/'final-state-ACK.json',dict(status='FULL_STATE_FSYNC_REOPEN_RESTORE_PASS',checkpoint=ref(cp),matched_tensors=matched,actual_adam_updates=total,every_optimizer_step=total,seconds=time.monotonic()-begin,resume_equivalence_claimed=False));del state,saved,optimizer,scheduler;optimizer=scheduler=None
  model.eval();values=[[] for _ in job['targets']];seen=[];begin=time.monotonic()
  with torch.inference_mode():
   for row in held:
    with torch.autocast('cuda',dtype=torch.bfloat16):scores=model.scores(row)
    for bi,score in enumerate(scores):
     assert torch.isfinite(score).all();targets=[t for t in job['window_targets'][row['row_id']] if t['block_index']==bi]
     if not targets:continue
     ii=torch.tensor([t['i'] for t in targets],device=score.device);jj=torch.tensor([t['j'] for t in targets],device=score.device);actual=score[ii,jj].float().cpu().tolist()
     for t,v in zip(targets,actual):
      assert math.isfinite(v);values[t['target_index']].append(dict(row_id=row['row_id'],block_index=bi,i=t['i'],j=t['j'],score=v))
    seen.append(row['row_id'])
  assert seen==[r['row_id'] for r in held];scorefile=dest/'target-scores.jsonl'
  with scorefile.open('w') as f:
   for t,vs in zip(job['targets'],values):
    assert t['lcgp_span_valid']==bool(vs);record=dict(**t,lcgp_span_logit=float(max(v['score'] for v in vs)) if vs else 0.,window_scores=vs);f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
   f.flush();os.fsync(f.fileno())
  fsync_dir(dest);report=dict(status='COMPLETE_ALL_TARGET_BOUNDARIES',name=name,output=ref(scorefile),target_spans=len(values),held_windows=len(seen),held_ids=job['input_ids'],train_ids=job['train_ids'],model_checkpoint=ref(cp),state_ACK=ref(dest/'final-state-ACK.json') if not job['reuse'] else None,reuse=job['reuse'],epochs=logs,actual_adam_updates=steps,no_score_floor=True,held_gold_read=False,seconds=time.monotonic()-jstart,inference_seconds=time.monotonic()-begin)
  save(dest/'complete.json',report);reports.append(ref(dest/'complete.json'));del model;gc.collect();torch.cuda.empty_cache()
 save(out/'complete.json',dict(status='COMPLETE_THREE_FITS_FOUR_FULL_SCORE_PASSES',jobs=reports,model_fits=3,score_passes=4,actual_adam_updates=896,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch=torch.__version__,transformers=transformers.__version__,quality_runs=0,F1_gold_reads=0))
if __name__=='__main__':main()
