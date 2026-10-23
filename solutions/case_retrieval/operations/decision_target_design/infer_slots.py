#!/usr/bin/env python3
"""All80 fixed-slot forward predictions from the sole complete fresh epoch002."""
from pathlib import Path
import argparse,json,time,os,contextlib
import slot_model as common

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--source-manifest',required=True)
    a=p.parse_args();b=a.directory;plan=json.loads((b/'remote-plan.json').read_text())
    assert common.sha(__file__)==plan['infer_sha256'] and common.sha(common.__file__)==plan['common_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and not (b/'run_manifest.json').exists()
    import torch,transformers,peft
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from peft import PeftModel
    import qwen35_runner as runner
    assert common.sha(runner.__file__)=='33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669'
    assert transformers.__version__=='5.20.2' and peft.__version__=='0.21.2'
    torch.set_num_threads(1);started=time.monotonic();data=common.rows(b/'slot-meta-inputs.json.gz',plan['encoded_sha256'],False)
    terminal=Path(plan['training_directory'])
    complete=json.loads((terminal/'training_complete.json').read_text())
    assert common.sha(terminal/'training_complete.json')==plan['training_complete_sha256']
    assert complete['epochs']==2 and complete['global_step']==32
    checkpoint=terminal/'epoch-002';assert common.sha(checkpoint/'checkpoint.json')==complete['checkpoint_sha256']
    metadata=json.loads((checkpoint/'checkpoint.json').read_text());assert metadata['completed_epochs']==2 and metadata['global_step']==32
    for name in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']:
        path=checkpoint/name;assert common.sha(path)==metadata['files'][name]['sha256']==plan['adapter_files'][Path(name).name]
    actual=runner.verify_model(a.model,a.source_manifest)
    old=json.loads((terminal/'run_manifest.json').read_text())
    assert common.sha(terminal/'run_manifest.json')==complete['manifest_sha256'] and actual==old['model_files']
    config=Qwen3_5TextConfig.from_pretrained(a.model,local_files_only=True)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(a.model,config=config,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=False);model.eval()
    assert not any(x.requires_grad for x in model.parameters())
    common.save(b/'run_manifest.json',dict(plan_sha256=common.sha(b/'remote-plan.json'),infer_sha256=common.sha(__file__),
        common_sha256=common.sha(common.__file__),encoded_sha256=common.sha(b/'slot-meta-inputs.json.gz'),
        model_files=actual,adapter_files=plan['adapter_files'],training_complete_sha256=plan['training_complete_sha256'],
        loading=loading,threshold=.5,classes=[15,16],position='direct slot end; no shift',no_generation=True,no_F1_labels=True))
    torch.cuda.reset_peak_memory_stats();forward_start=time.monotonic();arms={}
    for arm in ['base','adapter']:
        count=0;arm_start=time.monotonic()
        context=model.disable_adapter() if arm=='base' else contextlib.nullcontext()
        with context,(b/f'{arm}.probabilities.jsonl').open('x') as f,(b/f'{arm}.responses.jsonl').open('x') as raw,torch.inference_mode():
            for index,row in enumerate(data):
                ids,positions=common.tensor_input(row,torch);logits=common.decision_logits(model,ids,positions,torch)
                probs=torch.softmax(logits,dim=-1)[:,1];assert torch.isfinite(probs).all()
                values=probs.tolist();mapping={cid:[] for cid,pid in row['keys']}
                for (cid,pid),value in zip(row['keys'],values):
                    if value>=.5:mapping[cid].append(pid)
                f.write(json.dumps(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],keys=row['keys'],
                    probabilities=values,logits=logits.tolist(),input_tokens=len(row['input_ids']),
                    input_ids_sha256=__import__('hashlib').sha256(json.dumps(row['input_ids']).encode()).hexdigest()))+'\n');f.flush()
                raw.write(json.dumps(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],response=json.dumps(mapping),
                    hit_token_limit=False,generated=False,input_tokens=len(row['input_ids'])))+'\n');raw.flush()
                count+=len(values)
                if (index+1)%10==0:print(json.dumps(dict(arm=arm,rows=index+1,pairs=count)),flush=True)
        torch.cuda.synchronize();assert count==1557
        arms[arm]=dict(rows=80,pairs=count,forward_seconds=time.monotonic()-arm_start,
            probabilities_sha256=common.sha(b/f'{arm}.probabilities.jsonl'),
            responses_sha256=common.sha(b/f'{arm}.responses.jsonl'),adapter_disabled=arm=='base')
    assert time.monotonic()-started<300
    common.save(b/'complete.json',dict(status='both80_fixed_slot_forward_complete',arms=arms,rows_each=80,CIDs_each=587,pairs_each=1557,
        manifest_sha256=common.sha(b/'run_manifest.json'),plan_sha256=common.sha(b/'remote-plan.json'),
        elapsed_seconds=time.monotonic()-started,forward_seconds=time.monotonic()-forward_start,
        completed_epoch=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),no_labels=True,no_generation=True,no_search=True))
    print('ALL80_FIXED_SLOT_FROZEN_NATIVE',flush=True)

if __name__=='__main__':main()
