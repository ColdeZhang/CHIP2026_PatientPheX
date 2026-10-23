#!/usr/bin/env python3
"""Two fixed .5 arms on identical input candidates, no labels or generation."""
from pathlib import Path
import argparse, contextlib, gzip, json, os, time
import slot_model as common
import microbatch_slots as batching


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--model',required=True)
    a=ap.parse_args();b=a.directory;started=time.monotonic();plan=json.loads((b/'plan.json').read_text())
    assert common.sha(__file__)==plan['inference_sha256'] and os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
    assert all(common.sha(b/n)==v for n,v in plan['local_files'].items())
    terminal=json.loads((b/'training_complete.json').read_text())
    assert terminal['status']=='fresh_entity_637_epoch002_160_complete' and terminal['global_step']==160 and terminal['epochs']==2
    checkpoint=b/'epoch-002';assert common.sha(checkpoint/'checkpoint.json')==terminal['checkpoint_sha256']
    meta=json.loads((checkpoint/'checkpoint.json').read_text())
    assert meta['completed_epochs']==2 and meta['global_step']==160 and meta['final_fixed_epoch']
    for name,item in meta['files'].items():
        if name.startswith('adapter/'):
            assert common.sha(checkpoint/name)==item['sha256'] and (checkpoint/name).stat().st_size==item['bytes']
    data=json.loads(gzip.decompress((b/'heldout.json.gz').read_bytes()))
    assert len(data)==398 and len({r['pmc_id'] for r in data})==16
    assert sum(len(r['keys']) for r in data)==1589 and sum(len(r['input_ids']) for r in data)==295155
    assert len({r['row_id'] for r in data})==398
    for row in data:
        assert not {'targets','labels','answer','response'} & row.keys()
        assert len(row['keys'])==len(row['slot_positions']) and row['slot_positions'][-1]==len(row['input_ids'])-1
    import torch,transformers,peft
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,AutoTokenizer
    import qwen35_runner as runner
    assert transformers.__version__=='5.20.2' and peft.__version__=='0.21.2'
    torch.set_num_threads(1)
    files=runner.verify_model(a.model,b/'model-source-manifest.json')
    trained=json.loads((b/'run_manifest.json').read_text())
    assert common.sha(b/'run_manifest.json')==terminal['manifest_sha256'] and files==trained['model_files']
    config=Qwen3_5TextConfig.from_pretrained(a.model,local_files_only=True)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(a.model,config=config,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=peft.PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=False);model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True);pad=tokenizer.pad_token_id;assert pad is not None
    output=b/'inference';output.mkdir(exist_ok=False)
    common.save(output/'manifest.json',dict(source_sha256=common.sha(__file__),plan_sha256=common.sha(b/'plan.json'),
        training_complete_sha256=common.sha(b/'training_complete.json'),checkpoint_sha256=terminal['checkpoint_sha256'],
        model_files=files,adapter_files={k:v for k,v in meta['files'].items() if k.startswith('adapter/')},loading=loading,
        heldout_sha256=common.sha(b/'heldout.json.gz'),batch_source_sha256=common.sha(batching.__file__),
        cutoff=.5,existing_head_rows=[15,16],generation=False,labels=False,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__)))
    # Input order preserved; inference groups are consecutive <=8 rows and each
    # is then split by the same tested right-padding cap. No length sorting.
    groups=[list(range(i,min(i+8,len(data)))) for i in range(0,len(data),8)]
    batches=[ids for group in groups for ids in batching.microbatches(data,group)]
    arms={};torch.cuda.reset_peak_memory_stats()
    for arm in ['base','final']:
        tick=time.monotonic();count=0;total=0
        context=model.disable_adapter() if arm=='base' else contextlib.nullcontext()
        with context,torch.inference_mode(),(output/(arm+'.probabilities.jsonl')).open('x') as f:
            for ids in batches:
                rows=[data[i] for i in ids];batch=batching.collate(rows,torch,'cuda',pad)
                logits=batching.decision_logits(model,batch,torch)
                probabilities=torch.softmax(logits,dim=-1)[:,1]
                assert torch.isfinite(logits).all() and torch.isfinite(probabilities).all()
                for row,prob,logit in zip(rows,probabilities.split(batch['counts']),logits.split(batch['counts'])):
                    record=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],keys=row['keys'],
                        probabilities=prob.tolist(),logits=logit.tolist(),input_tokens=len(row['input_ids']))
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1;total+=len(row['keys'])
                f.flush()
            torch.cuda.synchronize()
        assert count==398 and total==1589
        arms[arm]=dict(rows=count,candidates=total,seconds=time.monotonic()-tick,
            probability_sha256=common.sha(output/(arm+'.probabilities.jsonl')),adapter_disabled=arm=='base')
        print(json.dumps(dict(arm=arm,**arms[arm])),flush=True)
    elapsed=time.monotonic()-started;assert elapsed<600
    common.save(output/'complete.json',dict(status='both_entity_arms_frozen_before_quality',arms=arms,seconds=elapsed,
        manifest_sha256=common.sha(output/'manifest.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        no_labels=True,no_quality=True,no_generation=True,cutoff=.5))
    print('ENTITY_TWO_ARMS_FROZEN',flush=True)


if __name__=='__main__':main()
