"""One fixed original epoch30 head -> all F0 pairs, no fit or quality."""
import gzip,json,math,time
from pathlib import Path
from common import save,ref,sha,fsync_dir

def main():
 import torch,transformers
 from model import load_model
 start=time.monotonic();root=Path(__file__).resolve().parent;cfg=json.loads((root/'infer-plan.json').read_text());out=root/'output';out.mkdir()
 cpu=json.loads((root/'cpu-complete.json').read_text());assert cpu['status']=='PASS_ALL_F0_NATIVE_IDS_OWNERS_NO_CUDA'
 assert ref(root/'data.json.gz')['sha256']==cpu['data']['sha256']==cfg['data']['sha256']
 assert ref(cfg['checkpoint']['path'])==cfg['checkpoint']
 assert ref(cfg['parent_ACK']['path'])==cfg['parent_ACK'];ack=json.loads(Path(cfg['parent_ACK']['path']).read_text())
 assert ack['checkpoint']==cfg['checkpoint'] and ack['actual_adam_updates']==240
 for n,v in cpu['assets'].items():
  st=(Path(cfg['model'])/n).stat();assert st.st_size==v['bytes'] and st.st_ino==v['guard']['inode'] and st.st_mtime_ns==v['guard']['mtime_ns']
 data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))['held'];assert len(data)==16 and all('labels'not in d for d in data)
 torch.set_num_threads(1);assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();torch.cuda.reset_peak_memory_stats()
 model=load_model(cfg['model'],out/'loading.json')
 state=torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=False)
 assert state['epochs']==30 and state['actual_adam_updates']==240 and len(state['visits'])==960
 model.load_state_dict(state['model'],strict=True)
 for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state['model'][k]),k
 loaded_tensors=len(state['model']);del state;model.eval();rows=[];seen=[];infer_start=time.monotonic()
 with torch.inference_mode():
  for doc in data:
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(doc).float().cpu().tolist()
   assert len(logits)==len(doc['keys'])and all(math.isfinite(x)for x in logits)
   rows.extend(dict(key=k,logit=v,probability=1/(1+math.exp(-max(-80.,min(80.,v)))))for k,v in zip(doc['keys'],logits));seen.append(doc['pmc_id'])
 assert len(rows)==len({tuple(x['key'])for x in rows})==2195
 dest=out/'held-scores.jsonl'
 with dest.open('x')as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
  f.flush();__import__('os').fsync(f.fileno())
 fsync_dir(out)
 save(out/'complete.json',dict(status='COMPLETE_SAME_EPOCH30_ALL_F0_PAIRS_NO_FIT',actual_adam_updates=0,parent_adam_updates=240,
  held_docs=16,held_patients=47,held_windows=16,held_pairs=2195,held_doc_ids=seen,held_scores=ref(dest),
  checkpoint=cfg['checkpoint'],parent_ACK=cfg['parent_ACK'],loaded_model_tensors=loaded_tensors,loaded_values_exact=True,
  cpu=ref(root/'cpu-complete.json'),plan=ref(root/'infer-plan.json'),new_fit=0,held_gold_reads=0,quality_runs=0,
  seconds=time.monotonic()-start,inference_seconds=time.monotonic()-infer_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  torch=torch.__version__,transformers=transformers.__version__))

if __name__=='__main__':main()
