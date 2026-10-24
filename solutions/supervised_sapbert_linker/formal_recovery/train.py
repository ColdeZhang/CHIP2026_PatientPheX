"""Fresh fixed3epoch/219step shared encoder; no quality, threshold or checkpoint selection."""
import gzip,json,os,random,time
from pathlib import Path
from common import ref,save,fsync_dir,tensor_compare


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
    root=Path(__file__).resolve().parent;out=root/'formal-output';out.mkdir()
    cfg=json.loads((root/'formal-plan.json').read_text());cpu=json.loads((root/'cpu-complete.json').read_text())
    assert cpu['status']=='PASS_ALL_NATIVE_IDS_NO_MODEL_OR_CUDA' and cpu['plan']['sha256']==cfg['native_plan_sha256']
    assert cpu['data']['sha256']==ref(root/'data.json.gz')['sha256']==cfg['data_sha256']
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
    model.cuda();params=[p for n,p in model.named_parameters() if not n.startswith('pooler.')]
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=11,num_training_steps=219)
    scaler=torch.amp.GradScaler('cuda',init_scale=4096.)
    candidates=[]
    def inputs(rows):
            return {k:v.cuda() for k,v in tok.pad(dict(input_ids=rows),padding=True,return_tensors='pt').items()}
    def encode(rows,batch=128):
            result=[]
            for start in range(0,len(rows),batch):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    v=model(**inputs(rows[start:start+batch])).last_hidden_state[:,0]
                result.append(F.normalize(v.float(),p=2,dim=1).cpu())
            return torch.cat(result)
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

    epochs=[];updates=0;bank_records=[]
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    for epoch in range(1,4):
        model.eval();bt=time.monotonic()
        bank=encode([x['ids'] for x in data['bank']]);queries=encode([x['ids'] for x in data['train']]);torch.cuda.synchronize()
        bankgpu=bank.cuda();retrieved=[]
        for start in range(0,len(queries),64):
            sim=queries[start:start+64].cuda()@bankgpu.T
            assert torch.isfinite(sim).all()
            retrieved.extend(sim.topk(20,dim=1).indices.cpu().tolist())
        del bankgpu,sim
        candidates=[sorted(set(top)|set(row['positive_public_rows'])) for row,top in zip(data['train'],retrieved)]
        assert len(candidates)==1160 and max(map(len,candidates))<=46
        save(out/('epoch-%d-candidates.json'%epoch),dict(top20=retrieved,candidates=candidates))
        bank_records.append(dict(epoch=epoch,bank_rows=41099,queries=1160,seconds=time.monotonic()-bt,max_candidates=max(map(len,candidates))))
        order=list(range(1160));random.shuffle(order);model.train();et=time.monotonic();records=[]
        for offset in range(0,1160,16):
            indices=order[offset:offset+16];optimizer.zero_grad(set_to_none=True)
            batch_identity=dict(epoch=epoch,offset=offset,completed_updates=updates,query_indices=indices,scale=scaler.get_scale())
            save(out/'current-batch.json',batch_identity)
            row=None
            try:
                row=run_batch(indices);scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in params)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            except Exception as error:
                detail={n:dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False) for n,p in model.named_parameters() if not n.startswith('pooler.')}
                save(out/'numerical-failure.json',dict(batch=batch_identity,forward=row,exception_type=type(error).__name__,message=str(error),gradients=detail))
                raise
            before_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            assert scaler.get_scale()>=before_scale,'No silent skipped update permitted'
            scheduler.step();updates+=1
            row.update(update=updates,gradient_norm=float(norm),lr=scheduler.get_last_lr()[0]);records.append(row)
        assert len(records)==73 and len(records[-1]['query_indices'])==8
        assert sorted(j for x in records for j in x['query_indices'])==list(range(1160))
        epoch_record=dict(epoch=epoch,updates=73,rows=1160,total_updates=updates,seconds=time.monotonic()-et,
            average_query_loss=sum(x['loss']*len(x['query_indices']) for x in records)/1160,
            max_sequences=max(x['sequences'] for x in records),max_padded_length=max(x['padded_length'] for x in records))
        epochs.append(epoch_record);save(out/('epoch-%d.json'%epoch),dict(summary=epoch_record,steps=records))
        save(out/'progress.json',dict(epoch=epoch,total_updates=updates,rows_seen=epoch*1160))
    assert updates==219 and len(optimizer.state)==197
    assert {int(x['step']) for x in optimizer.state.values()}=={219}
    optimizer.zero_grad(set_to_none=True)
    changed=[]
    for n,p in model.named_parameters():
        assert torch.isfinite(p).all()
        different=not torch.equal(initial[n],p.detach().cpu())
        if n.startswith('pooler.'):assert not different
        elif different:changed.append(n)
    assert len(changed)==197
    del initial
    checkpoint_start=time.monotonic()
    state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
        actual_adam_updates=219,additional_backwards_after_adam=0,gradients_cleared_before_checkpoint=True,training_authorized=True,probe_state_loaded=False)
    # Numpy RNG is stored in a deterministic primitive/tensor form for full comparison.
    ns=state['rng']['numpy'];state['rng']['numpy']=(ns[0],torch.from_numpy(ns[1].astype('int64')),ns[2],ns[3],ns[4])
    cp=out/'final-epoch3-state.pt'
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

    checkpoint_seconds=time.monotonic()-checkpoint_start
    checkpoint_ref=ref(cp)
    save(out/'final-state-ACK.json',dict(status='FINAL_EPOCH3_FULL_STATE_SAVED_REOPENED',updates=219,epochs=3,
        checkpoint=checkpoint_ref,compared_tensors=matched,seconds=checkpoint_seconds,
        model_optimizer_scheduler_scaler_rng_exact=True,resume_trajectory_equivalence_claimed=False))
    model.eval();bt=time.monotonic();bank=encode([x['ids'] for x in data['bank']]);torch.cuda.synchronize()
    bank_records.append(dict(epoch='final',bank_rows=41099,seconds=time.monotonic()-bt))
    held=encode([x['ids'] for x in data['held']]);assert held.shape==(303,768) and torch.isfinite(held).all()
    bankgpu=bank.cuda();outputs=[]
    for start in range(0,303,64):
        scores=held[start:start+64].cuda()@bankgpu.T;assert torch.isfinite(scores).all()
        values,index=scores.max(1);top_scores,top_rows=scores.topk(20,dim=1)
        for j,(v,i) in enumerate(zip(values.cpu().tolist(),index.cpu().tolist())):
            outputs.append(dict(text=data['held'][start+j]['text'],score=v,public_row=i,identifiers=data['bank'][i]['identifiers'],
                top20_rows=top_rows[j].cpu().tolist(),top20_scores=top_scores[j].cpu().tolist()))
    assert len(outputs)==303 and len({x['text'] for x in outputs})==303
    for name,array in [('final-public-bank.npy',bank.numpy()),('held-query-vectors.npy',held.numpy())]:
        with open(out/name,'wb') as f:np.save(f,array);f.flush();os.fsync(f.fileno())
    save(out/'held303-results.json',outputs)
    save(out/'complete.json',dict(status='COMPLETE_FRESH3EPOCH219_AND_ALL303_NO_QUALITY',elapsed_seconds=time.monotonic()-begin,
        epochs=epochs,updates=219,train_query_visits=3480,bank_refreshes=bank_records,bank_refresh_count=4,held_rows=303,
        changed_nonpooler_parameters=len(changed),checkpoint=checkpoint_ref,state_ACK=ref(out/'final-state-ACK.json'),
        held_results=ref(out/'held303-results.json'),final_bank=ref(out/'final-public-bank.npy'),held_vectors=ref(out/'held-query-vectors.npy'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        fresh_base=True,probe_state_loaded=False,failed_formal_state_loaded=False,grad_scaler_initial_scale=4096.,silent_skipped_updates=0,quality_calls=0,torch=torch.__version__,transformers=transformers.__version__))


if __name__=='__main__':main()
