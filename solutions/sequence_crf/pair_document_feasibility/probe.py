#!/usr/bin/env python3
"""One longest-document AdamW step per fresh arm. No saved model or quality run."""
import argparse,datetime,gc,hashlib,json,os,time
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from transformers import BertConfig,BertModel
from torch_head import GlobalPairHead,endpoints,class_loss


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def encode(encoder,doc):
    device=next(encoder.parameters()).device;n=len(doc['input_ids']);valid=torch.tensor(doc['valid_mask'],device=device,dtype=torch.bool)
    total=torch.zeros(n,768,device=device);count=torch.zeros(n,device=device);windows=doc['windows']
    for i in range(0,len(windows),4):
        part=windows[i:i+4];length=max(len(w['input_ids']) for w in part)
        ids=torch.zeros(len(part),length,device=device,dtype=torch.long);attn=torch.zeros_like(ids)
        for j,w in enumerate(part):ids[j,:len(w['input_ids'])]=torch.tensor(w['input_ids'],device=device);attn[j,:len(w['input_ids'])]=1
        hs=encoder(input_ids=ids,attention_mask=attn,return_dict=True).last_hidden_state
        for j,w in enumerate(part):
            idx=torch.arange(w['start'],w['end'],device=device);good=valid[idx];idx=idx[good]
            total=total.index_add(0,idx,hs[j,1:1+w['end']-w['start']][good])
            count=count.index_add(0,idx,torch.ones(len(idx),device=device))
    assert bool((count[valid]>0).all()) and bool((count[~valid]==0).all())
    return total/count.clamp_min(1)[:,None],valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default='probe-plan.json');a=ap.parse_args();tick=time.monotonic()
    p=json.loads(Path(a.plan).read_text());assert not Path('probe-result.json').exists()
    for name,digest in p['files'].items():assert sha(name)==digest,(name,'source or input SHA')
    assert Path('cpu-result.json').exists() and sha('cpu-result.json')==p['cpu_result_sha256']
    assert json.loads(Path('cpu-result.json').read_text())['status']=='ACTUAL_CPU_AUTOGRAD_PASS'
    doc=json.loads(Path('longest-train-document.json').read_text());assert doc['pmc_id']==p['selected_document']
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name,r in p['model_files'].items():
        f=Path(p['model'])/name;assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    records=[]
    for mode in ['uniform','pair']:
        ts=time.monotonic();torch.manual_seed(p['seed']);np.random.seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
        config=BertConfig.from_pretrained(p['model'],local_files_only=True);config._attn_implementation='eager'
        encoder=BertModel(config,add_pooling_layer=False)
        raw=torch.load(Path(p['model'])/'pytorch_model.bin',map_location='cpu',weights_only=True)
        expected=set(encoder.state_dict());prefix='bert.' if all('bert.'+key in raw for key in expected) else ''
        assert all(prefix+key in raw for key in expected)
        used={key:raw[prefix+key] for key in expected};incompatible=encoder.load_state_dict(used,strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        excluded=sorted(set(raw)-{prefix+key for key in expected});del raw,used
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assert encoder.is_gradient_checkpointing and all(layer.gradient_checkpointing for layer in [encoder.encoder])
        head=GlobalPairHead();assert sum(x.numel() for x in head.parameters())==344577
        initial_head_sha=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in head.parameters())).hexdigest()
        encoder.to('cuda').train();head.to('cuda').train();assert all(x.requires_grad for x in encoder.parameters())
        optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':2e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();loaded=time.monotonic()
        H,valid=encode(encoder,doc);pairs=[tuple(x) for x in doc['pairs']]
        patient,concept=endpoints(H,pairs,doc['patient_mentions'],doc['concept_mentions'],valid)
        z=head.project(H,valid);y=torch.tensor(doc['labels'],device='cuda',dtype=torch.float32);weights=torch.tensor(p['class_weights'],device='cuda');denom=weights[y.long()].sum()
        loss=H.new_zeros(());all_logits=[]
        for start in range(0,len(pairs),32):
            logits,_,_=head.from_projected(z,patient[start:start+32],concept[start:start+32],mode)
            loss=loss+class_loss(logits,y[start:start+32],weights,denom);all_logits.append(logits.detach())
        assert torch.isfinite(loss);torch.cuda.synchronize();forward_end=time.monotonic();loss.backward();torch.cuda.synchronize();backward_end=time.monotonic()
        named={**{'encoder.'+k:v for k,v in encoder.named_parameters()},**{'head.'+k:v for k,v in head.named_parameters()}}
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in named.values())
        targetnames=['encoder.embeddings.word_embeddings.weight','encoder.encoder.layer.0.attention.self.query.weight','encoder.encoder.layer.11.output.dense.weight','head.U','head.W','head.A','head.o']
        checks=[]
        for name in targetnames:
            v=named[name];g=v.grad;index=int(g.abs().reshape(-1).argmax());norm=float(g.norm());assert norm>0
            checks.append(dict(name=name,index=index,before=float(v.detach().reshape(-1)[index]),gradient_norm=norm))
        norm=float(torch.nn.utils.clip_grad_norm_(list(named.values()),1.));assert np.isfinite(norm) and norm>0
        optimizer.step();torch.cuda.synchronize();end=time.monotonic()
        for item in checks:
            item['after']=float(named[item['name']].detach().reshape(-1)[item['index']]);assert item['after']!=item['before'],item
        record=dict(mode=mode,completed_at_utc=stamp(),seconds=end-ts,load_seconds=loaded-ts,forward_seconds=forward_end-loaded,backward_seconds=backward_end-forward_end,update_seconds=end-backward_end,loss=float(loss.detach()),gradient_norm_before_clip=norm,module_updates=checks,initial_head_sha256=initial_head_sha,
                    encoder_parameters=sum(x.numel() for x in encoder.parameters()),head_parameters=sum(x.numel() for x in head.parameters()),public_checkpoint_excluded_keys=excluded,gradient_checkpointing=True,checkpoint_use_reentrant=False,all_encoder_parameters_trainable=True,precision='FP32, TF32 disabled, eager attention',max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),pairs=len(pairs),windows=len(doc['windows']),document=doc['pmc_id'],labels_positive=int(y.sum()),one_nonzero_LR_step=True)
        records.append(record);save('probe-partial.json',dict(plan_sha256=sha(a.plan),records=records,seconds=time.monotonic()-tick));print(json.dumps(record),flush=True)
        del optimizer,encoder,head,H,valid,patient,concept,z,y,weights,denom,loss,all_logits,logits,named,v,g
        gc.collect();torch.cuda.empty_cache()
    assert records[0]['initial_head_sha256']==records[1]['initial_head_sha256']
    result=dict(status='TWO_FRESH_ARMS_SINGLE_STEP_PASS_NO_QUALITY_OR_SAVED_STATE',completed_at_utc=stamp(),seconds=time.monotonic()-tick,plan_sha256=sha(a.plan),input_sha256=sha('longest-train-document.json'),model_files=p['model_files'],sources=p['files'],torch=torch.__version__,transformers=__import__('transformers').__version__,numpy=np.__version__,gpu_name=torch.cuda.get_device_name(0),records=records,budget_seconds=180,grace_seconds=15,formal_training_started=False,probe_state_saved=False)
    save('probe-result.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
