#!/usr/bin/env python3
"""Matched whole-document training; heldout inputs contain no labels."""
import datetime,gc,gzip,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from transformers import BertConfig,BertModel
from torch_head import GlobalPairHead,endpoints,class_loss
from probe import encode

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
    q=Path(str(p)+'.pending');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def loadzip(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def state_hash(module):
    h=hashlib.sha256()
    for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    started=time.monotonic();plan=json.loads(Path('formal-plan.json').read_text())
    assert not Path('training-complete.json').exists()
    for name,ref in plan['remote_files'].items():assert sha(name)==ref['sha256']
    train=loadzip('train-documents.json.gz');held=loadzip('F1-input-documents.json.gz')
    assert len(train)==32 and len(held)==16
    trainids={d['pmc_id'] for d in train};heldids={d['pmc_id'] for d in held}
    assert not trainids&heldids and sorted(trainids)==plan['train_ids'] and sorted(heldids)==plan['heldout_ids']
    assert all('labels' in d and len(d['labels'])==len(d['pairs']) for d in train)
    assert all('labels' not in d and 'association' not in d and 'answer' not in d for d in held)
    assert sum(len(d['pairs']) for d in train)==2506 and sum(sum(d['labels']) for d in train)==594
    assert sum(len(d['pairs']) for d in held)==1557
    orders=json.loads(Path('epoch-orders.json').read_text());assert len(orders)==8
    assert all(len(x)==32 and set(x)==trainids for x in orders)
    byid={d['pmc_id']:d for d in train}
    for n,ref in plan['model_files'].items():
        f=Path(plan['model'])/n;assert f.stat().st_size==ref['bytes'] and sha(f)==ref['sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    requested_stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:requested_stop.__setitem__(0,True))
    def bounded():
        if requested_stop[0] or time.monotonic()-started>=plan['budget_seconds']:raise TimeoutError('shared formal budget')
    result=[];init_hash=None
    for mode in plan['arms']:
        bounded();armstart=time.monotonic();out=Path(mode);out.mkdir(exist_ok=False)
        torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);random.seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
        config=BertConfig.from_pretrained(plan['model'],local_files_only=True);config._attn_implementation='eager'
        encoder=BertModel(config,add_pooling_layer=False)
        raw=torch.load(Path(plan['model'])/'pytorch_model.bin',map_location='cpu',weights_only=True)
        expected=set(encoder.state_dict());prefix='bert.' if all('bert.'+k in raw for k in expected) else ''
        assert all(prefix+k in raw for k in expected)
        encoder.load_state_dict({k:raw[prefix+k] for k in expected},strict=True)
        excluded=sorted(set(raw)-{prefix+k for k in expected});del raw
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});assert encoder.is_gradient_checkpointing
        head=GlobalPairHead();assert sum(p.numel() for p in head.parameters())==344577
        ih={'encoder':state_hash(encoder),'head':state_hash(head)}
        if init_hash is None:init_hash=ih
        else:assert ih==init_hash
        encoder.to('cuda');head.to('cuda');assert all(p.requires_grad for p in encoder.parameters())
        optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':plan['LR']['encoder']},{'params':head.parameters(),'lr':plan['LR']['head']}],weight_decay=plan['weight_decay'])
        named={**{'encoder.'+k:v for k,v in encoder.named_parameters()},**{'head.'+k:v for k,v in head.named_parameters()}}
        weights=torch.tensor(plan['class_weights'],device='cuda');epochs=[];steps=0;module_updates=[]
        torch.cuda.reset_peak_memory_stats();loadseconds=time.monotonic()-armstart
        def checkpoint(filename,complete):
            path=out/filename;pending=out/(filename+'.pending')
            torch.save({'encoder':{k:v.detach().cpu() for k,v in encoder.state_dict().items()},'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'optimizer':optimizer.state_dict(),'completed_epochs':len(epochs),'steps':steps,'mode':mode,'plan_sha256':sha('formal-plan.json'),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'partial_not_for_inference':not complete},pending)
            pending.replace(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
        try:
            for epoch,order in enumerate(orders,1):
                encoder.train();head.train();losses=[];t=time.monotonic()
                for pmc in order:
                    bounded();doc=byid[pmc];optimizer.zero_grad(set_to_none=True)
                    H,valid=encode(encoder,doc);pairs=doc['pairs'];patient,concept=endpoints(H,pairs,doc['patient_mentions'],doc['concept_mentions'],valid)
                    z=head.project(H,valid);y=torch.tensor(doc['labels'],device='cuda',dtype=torch.float32);denom=weights[y.long()].sum();loss=H.new_zeros(())
                    for i in range(0,len(pairs),plan['pairchunk']):
                        logits,_,_=head.from_projected(z,patient[i:i+plan['pairchunk']],concept[i:i+plan['pairchunk']],mode)
                        loss=loss+class_loss(logits,y[i:i+plan['pairchunk']],weights,denom)
                    assert torch.isfinite(loss);loss.backward()
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in named.values())
                    checks=[]
                    if steps==0:
                        for n in plan['update_check_modules']:
                            v=named[n];idx=int(v.grad.abs().reshape(-1).argmax());gn=float(v.grad.norm());assert gn>0
                            checks.append({'name':n,'index':idx,'before':float(v.detach().reshape(-1)[idx]),'gradient_norm':gn})
                    gradnorm=torch.nn.utils.clip_grad_norm_(list(named.values()),plan['clip_norm']);assert torch.isfinite(gradnorm)
                    optimizer.step();steps+=1
                    for row in checks:
                        row['after']=float(named[row['name']].detach().reshape(-1)[row['index']]);assert row['before']!=row['after']
                    if checks:module_updates=checks
                    losses.append(float(loss.detach()));del H,valid,patient,concept,z,y,denom,loss,logits
                torch.cuda.synchronize();epochs.append({'epoch':epoch,'steps':steps,'mean_document_weighted_loss':sum(losses)/len(losses),'seconds':time.monotonic()-t,'documents':32})
                save(out/'progress.json',{'mode':mode,'epochs':epochs,'steps':steps,'elapsed_seconds':time.monotonic()-started,'plan_sha256':sha('formal-plan.json')})
                print(json.dumps({'mode':mode,**epochs[-1]}),flush=True)
            bounded();assert steps==256 and len(epochs)==8
            final=checkpoint('final-state.pt',True);bounded()
            encoder.eval();head.eval();probabilities=[];inferstart=time.monotonic()
            with torch.no_grad():
                for doc in held:
                    bounded();H,valid=encode(encoder,doc);patient,concept=endpoints(H,doc['pairs'],doc['patient_mentions'],doc['concept_mentions'],valid);z=head.project(H,valid)
                    values=[]
                    for i in range(0,len(doc['pairs']),plan['pairchunk']):
                        logits,_,_=head.from_projected(z,patient[i:i+plan['pairchunk']],concept[i:i+plan['pairchunk']],mode)
                        assert torch.isfinite(logits).all();values.extend(torch.sigmoid(logits).cpu().tolist())
                    assert len(values)==len(doc['pairs'])
                    probabilities.extend({'pmc_id':doc['pmc_id'],'patient_id':pid,'concept':cid,'probability':pr} for (pid,cid),pr in zip(doc['pairs'],values))
                    del H,valid,patient,concept,z,logits
            assert len(probabilities)==1557
            save(out/'probabilities.json',probabilities)
            record={'mode':mode,'completed_at_utc':stamp(),'seconds':time.monotonic()-armstart,'load_seconds':loadseconds,'inference_seconds':time.monotonic()-inferstart,'epochs':epochs,'steps':steps,'initial_state_sha256':ih,'final':final,'probabilities':{'path':str(out/'probabilities.json'),'bytes':(out/'probabilities.json').stat().st_size,'sha256':sha(out/'probabilities.json')},'module_updates_first_step':module_updates,'all_encoder_trainable':True,'gradient_checkpointing':True,'precision':'FP32 TF32 disabled eager','encoder_parameters':sum(v.numel() for v in encoder.parameters()),'head_parameters':sum(v.numel() for v in head.parameters()),'excluded_public_checkpoint_keys':excluded,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            save(out/'complete.json',record);result.append(record)
        except BaseException as error:
            partial=None
            if not (out/'final-state.pt').exists():
                try:partial=checkpoint('partial-not-for-inference.pt',False)
                except BaseException as failure:partial={'save_error':repr(failure)}
            save(out/'failure.json',{'at_utc':stamp(),'error':repr(error),'completed_epochs':len(epochs),'steps':steps,'partial':partial,'no_quality':True});raise
        del optimizer,encoder,head,named,weights;gc.collect();torch.cuda.empty_cache()
    bounded();save('training-complete.json',{'completed_at_utc':stamp(),'seconds':time.monotonic()-started,'plan_sha256':sha('formal-plan.json'),'arms':result,'no_labels_in_F1_inputs':True,'quality_evaluation':False,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__,'gpu_name':torch.cuda.get_device_name(0)})

if __name__=='__main__':main()
