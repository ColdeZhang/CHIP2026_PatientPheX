"""Discarded engineering: full candidate losses, resident Adam, length/pair Pareto."""
import gzip,json,os,random,time
from pathlib import Path
from common import save,ref,tensor_compare,fsync_dir

def main():
 import torch,numpy as np,transformers
 from model import load_model
 root=Path(__file__).resolve().parent;out=root/'probe-output';out.mkdir()
 cfg=json.loads((root/'execution-plan.json').read_text());cpu=json.loads((root/'cpu-complete.json').read_text())
 assert cpu['status']=='PASS_FIXED_NATIVE_IDS_ALL_PAIR_POOLING_NO_CUDA'
 assert cpu['plan']==ref(root/'execution-plan.json') and cpu['data']==ref(root/'data.json.gz')
 for n,r in cpu['assets'].items():
  st=(Path(cfg['model'])/n).stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns']
 torch.set_num_threads(1);assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
 random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()));start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 model=load_model(cfg['model'],out/'loading.json');model.train()
 backbone=list(model.encoder.parameters());heads=[p for n,p in model.named_parameters()if not n.startswith('encoder.')]
 optimizer=torch.optim.AdamW([dict(params=backbone,lr=2e-5),dict(params=heads,lr=1e-4)],weight_decay=.01)
 def schedule(step):return (step+1)/24 if step<24 else max(0.,(240-step)/216)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
 longest=data['train'][data['pareto']['longest']];mostpairs=data['train'][data['pareto']['most_pairs']]
 save(out/'actual-shapes.json',dict(selected={k:dict(pmc_id=data['train'][i]['pmc_id'],**data['train'][i]['shape'])for k,i in data['pareto'].items()},train=[x['shape']for x in data['train']],held=[x['shape']for x in data['held']]))
 before={n:p.detach().cpu().clone()for n,p in model.named_parameters()};times=[]
 def run(doc):
  begin=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits=model(doc);labels=torch.tensor(doc['labels'],device='cuda',dtype=torch.float32)
   assert logits.shape==labels.shape and bool(torch.isfinite(logits).all())
   loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels)
  assert bool(torch.isfinite(loss));(loss/4).backward();torch.cuda.synchronize()
  for n,p in model.named_parameters():assert p.grad is not None and bool(torch.isfinite(p.grad).all()),n
  return dict(pmc_id=doc['pmc_id'],shape=doc['shape'],all_pairs=len(logits),loss=float(loss.detach()),forward_backward_seconds=time.monotonic()-begin)
 for doc in (longest,mostpairs,longest,mostpairs):times.append(run(doc))
 gradients={n:dict(norm=float(p.grad.float().norm()),nonzero=int(p.grad.count_nonzero()))for n,p in model.named_parameters()}
 assert all(v['nonzero']>0 for v in gradients.values())
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 lr=[g['lr']for g in optimizer.param_groups];assert all(x>0 for x in lr)
 begin=time.monotonic();optimizer.step();scheduler.step();torch.cuda.synchronize();adam_seconds=time.monotonic()-begin
 changed=[n for n,p in model.named_parameters()if not torch.equal(before[n],p.detach().cpu())]
 assert len(changed)==len(list(model.parameters())) and len(optimizer.state)==len(changed)
 del before;optimizer.zero_grad(set_to_none=True)
 # Real resident optimizer plus accumulated gradients, both maximum shapes.
 times.append(run(mostpairs));times.append(run(longest));peak=torch.cuda.max_memory_allocated();optimizer.zero_grad(set_to_none=True)
 cpstart=time.monotonic();nr=np.random.get_state();rng=dict(python=random.getstate(),numpy=(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
 state=dict(model={k:v.detach().cpu()for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng,actual_Adam_updates=1,actual_forward_backwards=6,discard_probe_state=True)
 cp=out/'discarded-full-state.pt'
 with cp.open('wb')as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 fsync_dir(out);restored=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,restored)
 model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
 random.setstate(restored['rng']['python']);ns=restored['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]));torch.set_rng_state(restored['rng']['torch']);torch.cuda.set_rng_state_all(restored['rng']['cuda'])
 for name,obj in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict())]:tensor_compare(obj,restored[name])
 tensor_compare(random.getstate(),restored['rng']['python']);tensor_compare(torch.get_rng_state(),restored['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),restored['rng']['cuda'])
 nr=np.random.get_state();tensor_compare((nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),restored['rng']['numpy'])
 saved=time.monotonic()-cpstart;worst=max(t['forward_backward_seconds']for t in times)
 length=longest['shape']['max_encoder_tokens'];linear_ratio=sum(x['shape']['max_encoder_tokens']/length for x in data['train'])
 save(out/'gradients.json',gradients)
 save(out/'complete.json',dict(status='PASS_FULL_CONTEXT_RELATION_ENGINEERING_DISCARD',parameters=sum(p.numel()for p in model.parameters()),actual_forward_backwards=6,actual_adam_updates=1,lr_at_update=lr,
  changed_parameter_tensors=len(changed),gradient_tensors=len(gradients),optimizer_states=len(optimizer.state),saved_tensors_compared=matched,checkpoint=ref(cp),state_save_reopen_seconds=saved,times=times,adam_seconds=adam_seconds,
  elapsed_seconds=time.monotonic()-start,peak_training_allocated_bytes=peak,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  formal30_cost=dict(worst_all960_FB_seconds=1.25*960*worst+240*adam_seconds+saved+90,linear_token_scaled_seconds=1.5*30*linear_ratio*worst+240*adam_seconds+saved+90,estimates_not_runtime_authorization=True),
  held_inference_not_run=True,held_gold_reads=0,quality_calls=0,probe_state_reuse_forbidden=True,torch=torch.__version__,transformers=transformers.__version__))
if __name__=='__main__':main()
