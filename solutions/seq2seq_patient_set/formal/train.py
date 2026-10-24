"""One fresh 10x87 LongT5 run, 220 Adam calls and a complete persistent state ACK."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from common import ref, save
from runtime import bind
from state_io import save_reopen_restore
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from probe import restore_shared_input_identity


def groups(order, size=4):
    return [order[i:i+size] for i in range(0,len(order),size)]


def main(a):
    p,b,out,modeldir,rows=bind(a.plan,a.binding,'train');start=time.monotonic();context={'stage':'load'}
    try:
        import numpy as np
        import torch,transformers
        from transformers import LongT5ForConditionalGeneration,get_linear_schedule_with_warmup
        assert (torch.__version__,transformers.__version__)==(b['torch'],b['transformers'])
        assert transformers.__version__=='4.50.0'
        torch.set_num_threads(1);random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
        assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        model,loading=LongT5ForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
        save(out/'loading.json',loading);assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        restore_shared_input_identity(model,out);assert sum(x.numel() for x in model.parameters())==247587456
        model.to('cuda');model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train();torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,22,220);visits=[];epochs=[];step=0
        with (out/'updates.jsonl').open('x') as progress:
            for epoch in range(10):
                order=list(range(87));random.Random(20261024+epoch).shuffle(order);epoch_start=time.monotonic();loss_sum=0.
                for group in groups(order):
                    optimizer.zero_grad(set_to_none=True);group_losses=[];tick=time.monotonic()
                    context={'stage':'train','epoch':epoch+1,'completed_adam':step,'group':group,'group_size':len(group)}
                    for i in group:
                        row=rows[i];context['row_id']=row['row_id']
                        ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
                        if not bool(torch.isfinite(loss)):
                            save(out/'numeric-failure.json',{**context,'finite_loss':False});raise FloatingPointError('Nonfinite loss')
                        value=float(loss.detach());group_losses.append(value);loss_sum+=value
                        (loss/len(group)).backward();visits.append(row['row_id']);del ids,labels,loss
                    grads=[x.grad for x in model.parameters() if x.grad is not None]
                    finite=[bool(torch.isfinite(x).all()) for x in grads]
                    if not all(finite):
                        save(out/'numeric-failure.json',{**context,'finite_gradients':finite});raise FloatingPointError('Nonfinite gradient')
                    assert len(grads)==295
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    used_lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();step+=1
                    assert len(optimizer.state)==295 and all(int(x['step'].item())==step for x in optimizer.state.values())
                    if step in (1,220):assert all(bool(torch.isfinite(v).all()) for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v))
                    torch.cuda.synchronize();item={'epoch':epoch+1,'actual_adam_updates':step,'row_indices':group,'row_ids':[rows[i]['row_id'] for i in group],
                        'losses':group_losses,'actual_group_size':len(group),'lr_used':used_lr,'gradient_norm':float(norm),'seconds':time.monotonic()-tick}
                    progress.write(json.dumps(item)+'\n');progress.flush();os.fsync(progress.fileno())
                assert sorted(order)==list(range(87)) and step==(epoch+1)*22
                item={'epoch':epoch+1,'actual_adam_updates':step,'order':order,'rows':87,'mean_patient_loss':loss_sum/87,'seconds':time.monotonic()-epoch_start}
                epochs.append(item);save(out/'progress.json',item);print(json.dumps(item),flush=True)
        assert step==220 and len(visits)==870
        context={'stage':'final_state','actual_adam_updates':step}
        ack=save_reopen_restore(model,optimizer,scheduler,out,{'epochs':10,'actual_adam_updates':220,'visits':visits,'plan':ref(a.plan),
            'supply':p['supply'],'source':ref(__file__),'fresh_public_parent':True,'probe_state_used':False})
        assert ack['every_optimizer_step']==[220]
        save(out/'complete.json',{'status':'COMPLETE_FRESH_10EPOCH_870ROWS_220ADAM_DURABLE_STATE','epochs':epochs,'train_rows':87,'train_documents':32,
            'row_visits':870,'actual_adam_updates':220,'checkpoint':ack['checkpoint'],'state_ACK':ref(out/'final-state-ACK.json'),
            'source':ref(__file__),'plan':ref(a.plan),'binding':ref(a.binding),'seconds':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
            'peak_reserved':torch.cuda.max_memory_reserved(),'held_calls':0,'quality_calls':0,'torch':torch.__version__,'transformers':transformers.__version__})
    except BaseException:
        save(out/'failure.json',{'context':context,'seconds':time.monotonic()-start,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',required=True);q.add_argument('--binding',required=True);main(q.parse_args())
