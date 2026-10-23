#!/usr/bin/env python3
"""One frozen binary32 F0 forward: probabilities only, no policy or labels."""
from pathlib import Path
import argparse,gzip,json,time,os
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
    torch.set_num_threads(1);started=time.monotonic()
    encoded=b/'F0-slot-inputs.json.gz';assert common.sha(encoded)==plan['encoded_sha256']
    data=json.loads(gzip.decompress(encoded.read_bytes()))
    assert len(data)==58 and len({x['pmc_id'] for x in data})==16 and len({x['row_id'] for x in data})==58
    assert sum(len(x['keys']) for x in data)==1166
    assert sum(len(x['input_ids']) for x in data)==208999 and max(len(x['input_ids']) for x in data)==6872
    for row in data:
        assert not {'targets','labels','answer','response'}&set(row)
        positions=row['slot_positions'];assert positions==sorted(set(positions)) and len(positions)==len(row['keys'])
        assert positions[0]>=row['prefix_tokens'] and positions[-1]==len(row['input_ids'])-1
        assert len(row['input_ids'])<=7168 and all(x['role'] in ['system','user'] for x in row['messages'])
    terminal=Path(plan['training_directory']);complete=json.loads((terminal/'training_complete.json').read_text())
    assert common.sha(terminal/'training_complete.json')==plan['training_complete_sha256']
    assert complete['epochs']==2 and complete['global_step']==32
    checkpoint=terminal/'epoch-002';assert common.sha(checkpoint/'checkpoint.json')==complete['checkpoint_sha256']
    metadata=json.loads((checkpoint/'checkpoint.json').read_text())
    assert metadata['completed_epochs']==2 and metadata['global_step']==32 and not metadata['partial_not_for_inference']
    for name in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']:
        assert common.sha(checkpoint/name)==metadata['files'][name]['sha256']==plan['adapter_files'][Path(name).name]
    actual=runner.verify_model(a.model,a.source_manifest);old=json.loads((terminal/'run_manifest.json').read_text())
    assert common.sha(terminal/'run_manifest.json')==complete['manifest_sha256'] and actual==old['model_files']
    config=Qwen3_5TextConfig.from_pretrained(a.model,local_files_only=True)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(a.model,config=config,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=False);model.eval()
    assert not any(x.requires_grad for x in model.parameters())
    common.save(b/'run_manifest.json',dict(plan_sha256=common.sha(b/'remote-plan.json'),infer_sha256=common.sha(__file__),
        common_sha256=common.sha(common.__file__),encoded_sha256=common.sha(encoded),model_files=actual,
        adapter_files=plan['adapter_files'],training_complete_sha256=plan['training_complete_sha256'],loading=loading,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
        classes=[15,16],position='direct slot end; no shift',no_generation=True,no_F0_labels=True,no_decoding_policy=True))
    torch.cuda.reset_peak_memory_stats();forward_start=time.monotonic();count=0
    with (b/'probabilities.jsonl').open('x') as f,torch.inference_mode():
        for index,row in enumerate(data):
            ids,positions=common.tensor_input(row,torch);logits=common.decision_logits(model,ids,positions,torch)
            probs=torch.softmax(logits,dim=-1)[:,1];assert torch.isfinite(probs).all()
            f.write(json.dumps(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],keys=row['keys'],
                probabilities=probs.tolist(),logits=logits.tolist(),input_tokens=len(row['input_ids']),
                input_ids_sha256=__import__('hashlib').sha256(json.dumps(row['input_ids']).encode()).hexdigest()))+'\n');f.flush()
            count+=len(row['keys'])
            if (index+1)%10==0:print(json.dumps(dict(rows=index+1,pairs=count)),flush=True)
    torch.cuda.synchronize();assert count==1166 and time.monotonic()-started<180
    common.save(b/'complete.json',dict(status='all58_F0_binary32_probabilities_complete',rows=58,CIDs=415,pairs=1166,
        probabilities_sha256=common.sha(b/'probabilities.jsonl'),manifest_sha256=common.sha(b/'run_manifest.json'),
        plan_sha256=common.sha(b/'remote-plan.json'),elapsed_seconds=time.monotonic()-started,
        forward_seconds=time.monotonic()-forward_start,completed_epoch=time.time(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        no_labels=True,no_generation=True,no_decoding_policy=True))
    print('ALL58_F0_SCORES_COMPLETE',flush=True)
if __name__=='__main__':main()
