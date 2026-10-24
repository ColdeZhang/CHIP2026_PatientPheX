"""Fresh fixed8ep/256Adam, durable final state, all17 held windows. No gold."""
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import random
import time
from common import ref,sha,save,tensor_compare,fsync_dir


def main():
    import numpy as np
    import torch,transformers
    from transformers import get_linear_schedule_with_warmup
    from model import load_model
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'formal-plan.json').read_text())
    out=root/'output';out.mkdir();start=time.monotonic()
    cpu=json.loads(Path(cfg['cpu_receipt']).read_text())
    assert cpu['status']=='PASS_ALL_NATIVE_IDS_LABELS_AND_LOCAL_GP_CONTRACT_NO_CUDA'
    assert sha(root/'data.json.gz')==cpu['data']['sha256']==cfg['data']['sha256']
    for n,r in cpu['assets'].items():
        st=(Path(cfg['model'])/n).stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns']
    data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()))
    assert len(data['train'])==32 and len(data['held'])==17
    assert all('positive' not in b for r in data['held'] for b in r['blocks'])
    torch.set_num_threads(1);random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();torch.cuda.reset_peak_memory_stats()
    # Only the immutable public parent is loaded. Never probe/full-state resume.
    model=load_model(cfg['model'],out/'loading.json');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,25,256);log=[];steps=0;visits=[]
    save(out/'inputs.json',{'plan':ref(root/'formal-plan.json'),'data':ref(root/'data.json.gz'),'cpu_receipt':ref(cfg['cpu_receipt']),
         'parent':'PUBLIC_C3648AA8_ONLY_FRESH_NO_PROBE_REUSE','parameters':sum(p.numel() for p in model.parameters()),'assets':cpu['assets']})
    for epoch in range(8):
        model.train();order=list(range(32));random.shuffle(order);loss_sum=0.;begin=time.monotonic()
        for i in order:
            optimizer.zero_grad(set_to_none=True);row=data['train'][i]
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(row)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert torch.isfinite(norm)
            optimizer.step();scheduler.step();steps+=1;visits.append(row['row_id']);loss_sum+=float(loss.detach())
        assert set(order)==set(range(32))
        item={'epoch':epoch+1,'actual_adam_updates':steps,'documents':32,'order':order,'mean_document_loss':loss_sum/32,'seconds':time.monotonic()-begin}
        log.append(item);save(out/'progress.json',item);print(json.dumps(item),flush=True)
    assert steps==256 and len(optimizer.state)==136
    assert all(int(s['step'].item())==256 for s in optimizer.state.values())
    optimizer.zero_grad(set_to_none=True);checkpoint_start=time.monotonic();nr=np.random.get_state()
    state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
       'rng':{'python':random.getstate(),'numpy':(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
       'epochs':8,'actual_adam_updates':256,'gradients_cleared':True,'plan':ref(root/'formal-plan.json'),'visits':visits}
    cp=out/'final-epoch8-full-state.pt'
    with cp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    fsync_dir(out);saved=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,saved)
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    random.setstate(saved['rng']['python']);ns=saved['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]))
    torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    tensor_compare(model.state_dict(),saved['model']);tensor_compare(optimizer.state_dict(),saved['optimizer']);tensor_compare(scheduler.state_dict(),saved['scheduler'])
    tensor_compare(random.getstate(),saved['rng']['python']);tensor_compare(torch.get_rng_state(),saved['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),saved['rng']['cuda'])
    nr=np.random.get_state();tensor_compare((nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),saved['rng']['numpy'])
    save(out/'final-state-ACK.json',{'status':'FULL_STATE_FSYNC_REOPEN_RESTORE_PASS','checkpoint':ref(cp),'matched_tensors':matched,
         'actual_adam_updates':256,'every_optimizer_step':256,'seconds':time.monotonic()-checkpoint_start,'resume_equivalence_claimed':False})
    del state,saved;model.eval();pool={};seen=[];infer_start=time.monotonic()
    # Root prelocked independent veto supply: read existing matrices only.
    targets=json.loads((root/'baseline-targets.json').read_text())
    assert sha(root/'baseline-targets.json')==cfg['baseline_targets']['sha256']
    target_scores={x['target_id']:[] for x in targets['records']}
    with torch.inference_mode():
        for row in data['held']:
            with torch.autocast('cuda',dtype=torch.bfloat16):scores=model.scores(row)
            for t in targets['window_targets'][row['row_id']]:
                v=float(scores[t['block_index']][t['i'],t['j']].float().cpu())
                assert __import__('math').isfinite(v)
                target_scores[t['target_id']].append({'row_id':row['row_id'],'block_index':t['block_index'],'i':t['i'],'j':t['j'],'score':v})
            for b,score in zip(row['blocks'],scores):
                assert torch.isfinite(score).all()
                valid=torch.ones_like(score,dtype=torch.bool).triu();ij=torch.nonzero(valid&(score>=-2.)).cpu().tolist();values=score.float().cpu()
                for i,j in ij:
                    lo,hi=b['offsets'][i][0],b['offsets'][j][1];assert 0<=lo<hi
                    key=(row['pmc_id'],b['passage_index'],lo,hi);v=float(values[i,j])
                    if key not in pool or v>pool[key]:pool[key]=v
            seen.append(row['row_id'])
            save(out/'inference-progress.json',{'complete_windows':len(seen),'total_windows':17,'unique_spans_so_far':len(pool)})
    assert seen==[r['row_id'] for r in data['held']]
    spans=[{'pmc_id':p,'passage_index':pi,'start':lo,'end':hi,'gp_logit':v} for (p,pi,lo,hi),v in sorted(pool.items())]
    path=out/'span-pool.jsonl'
    with path.open('w') as f:
        for r in spans:f.write(json.dumps(r,separators=(',',':'))+'\n')
        f.flush();os.fsync(f.fileno())
    fsync_dir(out)
    records=[]
    for original in targets['records']:
        row={**original,'window_scores':target_scores[original['target_id']]}
        assert row['representable']==bool(row['window_scores'])
        row['max_score']=max(x['score'] for x in row['window_scores']) if row['window_scores'] else None
        records.append(row)
    scorefile=out/'baseline-span-scores.json'
    save(scorefile,{'schema':'bioclinical-baseline-span-scores-v1','status':'COMPLETE_ALL_BASELINE_TARGET_SCORES',
        'source_baseline':targets['source_baseline'],'target_map':ref(root/'baseline-targets.json'),'model_checkpoint':ref(cp),
        'records':records,'total':1313,'held_windows':17,'max_rule':'maximum over all representable windows',
        'additional_forward_calls':0,'gold_reads':0,'quality_runs':0})
    save(out/'complete.json',{'status':'COMPLETE_FRESH8EP256_AND_ALL17_HELD_RAW','epochs':log,'actual_adam_updates':256,
       'train_docs':32,'train_visits':256,'held_docs':16,'held_patients':39,'held_windows':17,'held_window_ids':seen,
       'span_pool':ref(path),'raw_candidates':len(spans),'raw_floor':-2.,'duplicate_span_merge':'max raw GP logit across all windows',
       'normalization_not_run':True,'held_gold_reads':0,'quality_runs':0,'checkpoint':ref(cp),'state_ACK':ref(out/'final-state-ACK.json'),
       'baseline_span_scores':ref(scorefile),
       'seconds':time.monotonic()-start,'inference_seconds':time.monotonic()-infer_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'torch':torch.__version__,'transformers':transformers.__version__})


if __name__=='__main__':main()
