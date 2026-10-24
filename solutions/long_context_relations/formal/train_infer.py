"""One fresh30ep/240Adam from public MLM; all fixed F1 pairs, no gold."""
import gzip,json,math,os,random,time
from pathlib import Path
from common import ref,sha,save,tensor_compare,fsync_dir

def main():
 import torch,numpy as np,transformers
 from model import load_model
 root=Path(__file__).resolve().parent;cfg=json.loads((root/'formal-plan.json').read_text());out=root/'output';out.mkdir();start=time.monotonic()
 cpu=json.loads(Path(cfg['cpu_receipt']).read_text());assert cpu['status']=='PASS_FIXED_NATIVE_IDS_ALL_PAIR_POOLING_NO_CUDA'
 assert sha(root/'data.json.gz')==cfg['data']['sha256']==cpu['data']['sha256']
 for n,r in cpu['assets'].items():
  st=(Path(cfg['model'])/n).stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns']
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()));assert len(data['train'])==32 and len(data['held'])==16
 assert all('labels'in x for x in data['train'])and all('labels'not in x for x in data['held'])
 assert sum(len(x['pairs'])for x in data['train'])==4495 and sum(sum(x['labels'])for x in data['train'])==727
 torch.set_num_threads(1);random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
 assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();torch.cuda.reset_peak_memory_stats()
 model=load_model(cfg['model'],out/'loading.json')
 optimizer=torch.optim.AdamW([dict(params=list(model.encoder.parameters()),lr=2e-5),dict(params=[p for n,p in model.named_parameters()if not n.startswith('encoder.')],lr=1e-4)],weight_decay=.01)
 def schedule(step):return(step+1)/24 if step<24 else max(0.,(240-step)/216)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule);steps=0;visits=[];epochs=[]
 save(out/'inputs.json',dict(plan=ref(root/'formal-plan.json'),data=ref(root/'data.json.gz'),cpu_receipt=ref(cfg['cpu_receipt']),parent='PUBLIC_C3648AA8_FRESH_NO_TASK1_OR_PROBE_REUSE',parameters=sum(p.numel()for p in model.parameters()),assets=cpu['assets'],loss='unweighted BCE mean per document then actual4doc group mean'))
 for epoch in range(30):
  model.train();order=list(range(32));random.Random(20261024+epoch).shuffle(order);losses=[];begin=time.monotonic();pairs_seen=0
  for first in range(0,32,4):
   group=order[first:first+4];optimizer.zero_grad(set_to_none=True)
   for index in group:
    doc=data['train'][index]
    with torch.autocast('cuda',dtype=torch.bfloat16):
     logits=model(doc);labels=torch.tensor(doc['labels'],device='cuda',dtype=torch.float32)
     assert logits.shape==labels.shape and bool(torch.isfinite(logits).all())
     loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels)
    assert bool(torch.isfinite(loss));(loss/len(group)).backward();losses.append(float(loss.detach()));visits.append(doc['pmc_id']);pairs_seen+=len(doc['pairs'])
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert bool(torch.isfinite(norm));optimizer.step();scheduler.step();steps+=1
  assert set(order)==set(range(32))and pairs_seen==4495
  item=dict(epoch=epoch+1,actual_adam_updates=steps,documents=32,pairs_seen=pairs_seen,order=order,mean_document_loss=sum(losses)/32,seconds=time.monotonic()-begin)
  epochs.append(item);save(out/'progress.json',item);print(json.dumps(item),flush=True)
 assert steps==240 and len(visits)==960 and len(optimizer.state)==140 and all(int(x['step'].item())==240 for x in optimizer.state.values())
 optimizer.zero_grad(set_to_none=True);cpstart=time.monotonic();nr=np.random.get_state()
 state=dict(model={k:v.detach().cpu()for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=dict(python=random.getstate(),numpy=(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),epochs=30,actual_adam_updates=240,gradients_cleared=True,visits=visits,plan=ref(root/'formal-plan.json'))
 cp=out/'final-epoch30-full-state.pt'
 with cp.open('wb')as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 fsync_dir(out);restored=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,restored)
 model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
 random.setstate(restored['rng']['python']);ns=restored['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]));torch.set_rng_state(restored['rng']['torch']);torch.cuda.set_rng_state_all(restored['rng']['cuda'])
 for name,obj in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict())]:tensor_compare(obj,restored[name])
 tensor_compare(random.getstate(),restored['rng']['python']);tensor_compare(torch.get_rng_state(),restored['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),restored['rng']['cuda']);nr=np.random.get_state();tensor_compare((nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),restored['rng']['numpy'])
 cpref=ref(cp);save(out/'final-state-ACK.json',dict(status='FULL_STATE_FSYNC_REOPEN_RESTORE_PASS',checkpoint=cpref,matched_tensors=matched,actual_adam_updates=240,every_optimizer_step=240,seconds=time.monotonic()-cpstart,resume_equivalence_claimed=False))
 del state,restored,optimizer,scheduler;model.eval();seen=[];rows=[];inferstart=time.monotonic();torch.cuda.empty_cache()
 with torch.inference_mode():
  for doc in data['held']:
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(doc).float().cpu().tolist()
   assert len(logits)==len(doc['keys'])and all(math.isfinite(x)for x in logits)
   rows.extend(dict(key=key,logit=value,probability=1/(1+math.exp(-max(-80.,min(80.,value)))))for key,value in zip(doc['keys'],logits))
   seen.append(doc['pmc_id']);save(out/'inference-progress.json',dict(documents_complete=len(seen),pairs_complete=len(rows),total_documents=16,total_pairs=2874))
 assert len(rows)==2874 and len({tuple(x['key'])for x in rows})==2874 and seen==[x['pmc_id']for x in data['held']]
 scorepath=out/'held-scores.jsonl'
 with scorepath.open('x')as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 fsync_dir(out)
 save(out/'complete.json',dict(status='COMPLETE_FRESH30EP240_AND_ALL16_HELD_RELATIONS',epochs=epochs,actual_adam_updates=240,train_visits=960,train_pairs_per_epoch=4495,positive_train_pairs=727,
  held_docs=16,held_patients=39,held_windows=17,held_pairs=2874,held_doc_ids=seen,held_scores=ref(scorepath),held_gold_reads=0,quality_runs=0,checkpoint=cpref,state_ACK=ref(out/'final-state-ACK.json'),
  seconds=time.monotonic()-start,inference_seconds=time.monotonic()-inferstart,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch=torch.__version__,transformers=transformers.__version__))
if __name__=='__main__':main()
