"""One discarded real shared-Sap engineering probe. Never starts formal training."""
import gzip
import json
import math
import os
from pathlib import Path
import random
import time
from common import ref,save,sha,tensor_compare,fsync_dir


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
    root=Path(__file__).resolve().parent;out=root/'probe-output';out.mkdir()
    cfg=json.loads((root/'execution-plan.json').read_text());cpu=json.loads((root/'cpu-complete.json').read_text())
    assert cpu['status']=='PASS_ALL_NATIVE_IDS_NO_MODEL_OR_CUDA'
    assert cpu['data']==ref(root/'data.json.gz') and cpu['plan']==ref(root/'execution-plan.json')
    for name,record in cpu['assets'].items():
        st=(Path(cfg['model'])/name).stat()
        assert st.st_ino==record['guard']['inode'] and st.st_mtime_ns==record['guard']['mtime_ns'] and st.st_size==record['bytes']
    torch.set_num_threads(1);random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
    assert torch.cuda.device_count()==1
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True)
    model,loading=AutoModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
    save(out/'loading.json',loading)
    assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
    assert all(k.startswith('cls.') or k=='embeddings.position_ids' for k in loading['unexpected_keys'])
    assert sum(p.numel() for p in model.parameters())==109482240
    model.cuda();model.eval()
    params=[p for name,p in model.named_parameters() if not name.startswith('pooler.')]
    def inputs(rows):
        return {k:v.cuda() for k,v in tok.pad(dict(input_ids=rows),padding=True,return_tensors='pt').items()}
    def encode(rows,batch=128):
        result=[]
        for start in range(0,len(rows),batch):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                v=model(**inputs(rows[start:start+batch])).last_hidden_state[:,0]
            result.append(F.normalize(v.float(),p=2,dim=1).cpu())
        return torch.cat(result)
    banktime=time.monotonic();bank=encode([x['ids'] for x in data['bank']]);torch.cuda.synchronize();bank_seconds=time.monotonic()-banktime
    assert bank.shape==(41099,768) and torch.isfinite(bank).all()
    querytime=time.monotonic();queries=encode([x['ids'] for x in data['train']]);torch.cuda.synchronize();query_seconds=time.monotonic()-querytime
    bankgpu=bank.cuda();retrieved=[]
    for start in range(0,len(queries),64):
        sim=queries[start:start+64].cuda()@bankgpu.T
        assert torch.isfinite(sim).all()
        retrieved.extend(sim.topk(20,dim=1).indices.cpu().tolist())
    del bankgpu,sim
    candidates=[sorted(set(top)|set(row['positive_public_rows'])) for row,top in zip(data['train'],retrieved)]
    assert len(candidates)==1160
    with open(out/'initial-bank.npy','wb') as f:np.save(f,bank.numpy());f.flush();os.fsync(f.fileno())
    save(out/'hard20-and-candidates.json',dict(top20=retrieved,candidates=candidates))
    # Actual candidate count and token length define the Pareto stress batches.
    stats=[]
    for i,(row,cs) in enumerate(zip(data['train'],candidates)):
        lengths=[len(row['ids'])]+[len(data['bank'][j]['ids']) for j in cs]
        stats.append(dict(i=i,rows=len(lengths),maximum=max(lengths),padded_tokens=len(lengths)*max(lengths)))
    selected={}
    for name,key in [('largest_candidates',lambda x:(x['rows'],x['maximum'],x['i'])),
                     ('longest',lambda x:(x['maximum'],x['padded_tokens'],x['i']))]:
        selected[name]=[x['i'] for x in sorted(stats,key=key,reverse=True)[:16]]
    save(out/'actual-shapes.json',dict(train_queries=1160,bank_rows=41099,query_stats=stats,batches=selected,
        maximum_candidate_count=max(map(len,candidates)),bank_seconds=bank_seconds,query_seconds=query_seconds))
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=11,num_training_steps=219)
    scaler=torch.amp.GradScaler('cuda')
    model.train();times=[];snap={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('pooler.')}
    def run_batch(indices):
        rows=[];layout=[]
        for i in indices:
            start=len(rows);rows.append(data['train'][i]['ids']);rows.extend(data['bank'][j]['ids'] for j in candidates[i]);layout.append((i,start,len(rows)))
        x=inputs(rows);t=time.monotonic()
        with torch.autocast('cuda',dtype=torch.float16):
            values=model(**x).last_hidden_state[:,0]
            values=F.normalize(values.float(),p=2,dim=1)
            losses=[]
            for i,start,end in layout:
                row=data['train'][i]
                with torch.autocast('cuda',enabled=False):
                    score=(values[start:start+1]@values[start+1:end].T).flatten()/.05
                assert torch.isfinite(score).all()
                known=set(row['known_nonnegative_ids']);ids=[set(data['bank'][j]['identifiers']) for j in candidates[i]]
                pertarget=[]
                for target in row['train_positive_ids']:
                    pos=torch.tensor([target in s for s in ids],device='cuda',dtype=torch.bool)
                    neg=torch.tensor([not(s&known) for s in ids],device='cuda',dtype=torch.bool)
                    assert pos.any() and not(pos&neg).any()
                    pertarget.append(torch.logsumexp(score[pos|neg],0)-torch.logsumexp(score[pos],0))
                losses.append(torch.stack(pertarget).mean())
            loss=torch.stack(losses).mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();torch.cuda.synchronize()
        return dict(loss=float(loss.detach()),seconds=time.monotonic()-t,sequences=len(rows),padded_length=int(x['input_ids'].shape[1]),query_indices=indices)
    # Preserve the actual fresh warmup schedule: its first Adam has zero LR,
    # the next Adam uses the first nonzero warmup LR and must change weights.
    first=run_batch(selected['largest_candidates']);scaler.unscale_(optimizer)
    gradients={}
    for n,p in model.named_parameters():
        if n.startswith('pooler.'):
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
            gradients[n]=dict(nonzero=int(torch.count_nonzero(p.grad)),norm=float(p.grad.float().norm()))
            assert gradients[n]['nonzero']>0,n
    torch.nn.utils.clip_grad_norm_(params,1.)
    step_start=time.monotonic();scaler.step(optimizer);scaler.update();scheduler.step();torch.cuda.synchronize()
    first['adam_seconds']=time.monotonic()-step_start;times.append(first)
    optimizer.zero_grad(set_to_none=True)
    second=run_batch(selected['largest_candidates']);scaler.unscale_(optimizer)
    for p in params:assert p.grad is not None and torch.isfinite(p.grad).all()
    torch.nn.utils.clip_grad_norm_(params,1.)
    step_start=time.monotonic();scaler.step(optimizer);scaler.update();scheduler.step();torch.cuda.synchronize()
    second['adam_seconds']=time.monotonic()-step_start;times.append(second)
    changed=[]
    for n,p in model.named_parameters():
        if n in snap and not torch.equal(snap[n],p.detach().cpu()):changed.append(n)
    assert changed and len(optimizer.state)==len(params)
    optimizer.zero_grad(set_to_none=True);del snap
    # Resident optimizer and real accumulated gradients at both observed
    # Pareto batch shapes after the two exact original warmup Adam steps.
    times.append(run_batch(selected['largest_candidates']))
    times.append(run_batch(selected['longest']))
    for p in params:assert p.grad is not None and torch.isfinite(p.grad).all()
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
    optimizer.zero_grad(set_to_none=True)
    checkpoint_start=time.monotonic()
    state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
        actual_adam_updates=2,additional_backwards_after_adam=2,gradients_cleared_before_checkpoint=True,training_authorized=False,probe_state_must_be_discarded=True)
    # Numpy RNG is stored in a deterministic primitive/tensor form for full comparison.
    ns=state['rng']['numpy'];state['rng']['numpy']=(ns[0],torch.from_numpy(ns[1].astype('int64')),ns[2],ns[3],ns[4])
    cp=out/'discarded-probe-state.pt'
    with open(cp,'wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    fsync_dir(out)
    restored=torch.load(cp,map_location='cpu',weights_only=False)
    matched=tensor_compare(state,restored)
    model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler']);scaler.load_state_dict(restored['scaler'])
    random.setstate(restored['rng']['python']);ns=restored['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]));torch.set_rng_state(restored['rng']['torch']);torch.cuda.set_rng_state_all(restored['rng']['cuda'])
    tensor_compare(optimizer.state_dict(),restored['optimizer']);tensor_compare(model.state_dict(),restored['model'])
    tensor_compare(scheduler.state_dict(),restored['scheduler']);tensor_compare(scaler.state_dict(),restored['scaler'])
    tensor_compare(random.getstate(),restored['rng']['python']);tensor_compare(torch.get_rng_state(),restored['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),restored['rng']['cuda'])
    ns=np.random.get_state();tensor_compare((ns[0],torch.from_numpy(ns[1].astype('int64')),ns[2],ns[3],ns[4]),restored['rng']['numpy'])
    save(out/'complete.json',dict(status='PASS_REAL_ENGINEERING_PROBE_DISCARD_STATE',elapsed_seconds=time.monotonic()-begin,
        bank_seconds=bank_seconds,query_seconds=query_seconds,batches=times,real_adam_updates=2,backwards=4,
        scheduler='Original fresh linear warmup schedule; first lr0 then first nonzero lr; final probe state discarded.',
        nonpooler_gradient_tensors=len(gradients),changed_parameter_tensors=len(changed),optimizer_parameter_states=len(optimizer.state),
        full_state_compared_tensors=matched,state_save_reload_seconds=time.monotonic()-checkpoint_start,checkpoint=ref(cp),
        training_peak_allocated_bytes=peak,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in model.parameters()),
        estimated_formal_seconds=1.5*(219*(max(x['seconds'] for x in times)+first['adam_seconds'])+4*bank_seconds+3*query_seconds)+120,
        estimates_only=True,probe_to_formal_reuse_forbidden=True,torch=torch.__version__,transformers=transformers.__version__))
    save(out/'gradients.json',gradients)


if __name__=='__main__':main()
