"""One discarded native full-context GP training probe; no quality or fit arm."""
import gzip
import json
import os
from pathlib import Path
import random
import time
from common import save,ref,tensor_compare,fsync_dir


def main():
    import torch,numpy as np,transformers
    from transformers import get_linear_schedule_with_warmup
    from model import load_model
    root=Path(__file__).resolve().parent;out=root/'probe-output';out.mkdir()
    cfg=json.loads((root/'execution-plan.json').read_text());cpu=json.loads((root/'cpu-complete.json').read_text())
    assert cpu['status']=='PASS_ALL_NATIVE_IDS_LABELS_AND_LOCAL_GP_CONTRACT_NO_CUDA'
    assert cpu['plan']==ref(root/'execution-plan.json') and cpu['data']==ref(root/'data.json.gz')
    for n,r in cpu['assets'].items():
        p=Path(cfg['model'])/n;st=p.stat();assert st.st_size==r['bytes'] and st.st_ino==r['guard']['inode'] and st.st_mtime_ns==r['guard']['mtime_ns']
    torch.set_num_threads(1);assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
    data=json.loads(gzip.decompress((root/'data.json.gz').read_bytes()));start=time.monotonic()
    torch.cuda.reset_peak_memory_stats();model=load_model(cfg['model'],out/'loading.json');model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,25,256)
    shapes=data['pareto'];long=data['train'][shapes['longest']];passage=data['train'][shapes['longest_passage']]
    save(out/'actual-shapes.json',{'selected':{k:{'row_id':data['train'][i]['row_id'],**data['train'][i]['shape']} for k,i in shapes.items()},
         'all_train_shapes':[x['shape'] for x in data['train']], 'all_held_shapes':[x['shape'] for x in data['held']]})
    times=[]
    def run(row):
        begin=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(row)
        assert torch.isfinite(loss)
        loss.backward();torch.cuda.synchronize()
        for n,p in model.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),n
        return {'row_id':row['row_id'],'shape':row['shape'],'loss':float(loss.detach()),'forward_backward_seconds':time.monotonic()-begin}
    # First warmup update lr0 is retained, followed by genuine nonzero lr.
    first=run(long);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    begin=time.monotonic();optimizer.step();scheduler.step();torch.cuda.synchronize();first['adam_seconds']=time.monotonic()-begin;times.append(first)
    optimizer.zero_grad(set_to_none=True)
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    second=run(long);gradients={n:{'norm':float(p.grad.float().norm()),'nonzero':int(p.grad.count_nonzero())} for n,p in model.named_parameters()}
    assert gradients['head.weight']['nonzero']>0 and any(v['nonzero']>0 for k,v in gradients.items() if k.startswith('encoder.'))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    second['lr_before_nonzero_step']=optimizer.param_groups[0]['lr'];assert second['lr_before_nonzero_step']>0
    begin=time.monotonic();optimizer.step();scheduler.step();torch.cuda.synchronize();second['adam_seconds']=time.monotonic()-begin;times.append(second)
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p.detach().cpu())]
    assert 'head.weight' in changed and any(n.startswith('encoder.') for n in changed)
    assert len(optimizer.state)==len(list(model.parameters()))
    optimizer.zero_grad(set_to_none=True);del before
    # Already resident Adam, then truly accumulated gradients at both observed
    # training Pareto shapes. These gradients are discarded after measurement.
    times.append(run(passage));times.append(run(long));peak=torch.cuda.max_memory_allocated()
    optimizer.zero_grad(set_to_none=True);checkpoint_start=time.monotonic()
    nr=np.random.get_state();rng={'python':random.getstate(),'numpy':(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),
         'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
    state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
           'scheduler':scheduler.state_dict(),'rng':rng,'actual_Adam_updates':2,'additional_backwards':2,
           'discard_probe_state':True,'gradients_cleared_before_save':True}
    cp=out/'discarded-full-state.pt'
    with cp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    fsync_dir(out);restored=torch.load(cp,map_location='cpu',weights_only=False);matched=tensor_compare(state,restored)
    model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
    random.setstate(restored['rng']['python']);ns=restored['rng']['numpy'];np.random.set_state((ns[0],ns[1].numpy().astype('uint32'),ns[2],ns[3],ns[4]))
    torch.set_rng_state(restored['rng']['torch']);torch.cuda.set_rng_state_all(restored['rng']['cuda'])
    tensor_compare(model.state_dict(),restored['model']);tensor_compare(optimizer.state_dict(),restored['optimizer']);tensor_compare(scheduler.state_dict(),restored['scheduler'])
    tensor_compare(random.getstate(),restored['rng']['python']);tensor_compare(torch.get_rng_state(),restored['rng']['torch']);tensor_compare(torch.cuda.get_rng_state_all(),restored['rng']['cuda'])
    nr=np.random.get_state();tensor_compare((nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]),restored['rng']['numpy'])
    saved=time.monotonic()-checkpoint_start
    estimate=1.5*256*(max(x['forward_backward_seconds'] for x in times)+max(x.get('adam_seconds',0) for x in times))+saved+120
    save(out/'gradients.json',gradients)
    save(out/'complete.json',{'status':'PASS_REAL_NATIVE_LONG_CONTEXT_GP_DISCARD_STATE','parameters':sum(p.numel() for p in model.parameters()),
       'actual_forward_backwards':4,'actual_adam_updates':2,'changed_parameter_tensors':len(changed),'gradient_tensors':len(gradients),
       'optimizer_states':len(optimizer.state),'saved_tensors_compared':matched,'checkpoint':ref(cp),'state_save_reopen_seconds':saved,
       'times':times,'elapsed_seconds':time.monotonic()-start,'peak_training_allocated_bytes':peak,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
       'formal_8ep_256updates_conservative_seconds':estimate,'cost_is_estimate':True,'held_inference_not_run':True,'held_gold_reads':0,
       'torch':torch.__version__,'transformers':transformers.__version__,'precision':'FP32params/BF16encoder-autocast/FP32GP-loss',
       'model_loading':'native SDPA, reference_compileFalse, gradient_checkpointing nonreentrant; no MLM logits/hidden-all/attention outputs',
       'probe_state_reuse_forbidden':True})


if __name__=='__main__':main()
