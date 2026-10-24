"""Actual target CPU token interfaces and inherited asset identity, no model."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from common import checked,ref,save,stamp,verify_assets


def main(plan_path,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    import torch,transformers,peft,tokenizers
    started=time.monotonic();p=json.loads(Path(plan_path).read_text());out=Path(output);out.mkdir(exist_ok=False)
    assert not torch.cuda.is_available()
    assert all(v==p['runtime'][k] for k,v in [('torch',torch.__version__),('transformers',transformers.__version__),('peft',peft.__version__),('tokenizers',tokenizers.__version__)])
    previous=json.loads(checked(p['inherited_assets']).read_text())
    assets=verify_assets(p,full=False,previous=previous)
    save(out/'verified-assets.json',assets)
    for r in p['parent_adapter_files'].values():checked(r)
    config=json.loads(Path(p['parent_adapter_dir'],'adapter_config.json').read_text())
    assert config['r']==16 and config['lora_alpha']==32 and config['lora_dropout']==.05
    assert set(config['target_modules'])==set(p['lora']['target_modules'])
    pairs=[json.loads(x) for x in gzip.decompress(checked(p['pairs']).read_bytes()).splitlines()]
    enc=[json.loads(x) for x in gzip.decompress(checked(p['encoded']).read_bytes()).splitlines()]
    assert len(pairs)==len(enc)==74 and len({x['pmc_id'] for x in pairs})==17
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True)
    raw=Tokenizer.from_file(str(Path(p['model_dir'])/'tokenizer.json'))
    audits=[]
    for row,old in zip(pairs,enc):
        assert row['row_id']==old['row_id']
        prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert prefix==raw.encode(prompt,add_special_tokens=False).ids and len(prefix)==old['prompt_length']
        for side in ['chosen','rejected']:
            messages=row['messages']+[{'role':'assistant','content':row[side]}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
            ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
            assert ids==old[side+'_input_ids']==raw.encode(text,add_special_tokens=False).ids
            assert ids[:len(prefix)]==prefix and ids[-2:]==[151645,198]
            assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text==raw.decode(ids,skip_special_tokens=False)
            assert text==prompt+row[side]+'<|im_end|>\n'
            audits.append(dict(row_id=row['row_id'],side=side,tokens=len(ids),supervised=len(ids)-len(prefix),ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    save(out/'row-audit.json',audits)
    result=dict(status='PASS_ACTUAL_TARGET_HF_NATIVE_ALL148_AND_INHERITED_ASSET_STAT',at_utc=stamp(),seconds=time.monotonic()-started,
                sequence_count=len(audits),tokens=sum(x['tokens'] for x in audits),max_tokens=max(x['tokens'] for x in audits),
                supervised=sum(x['supervised'] for x in audits),all_complete_ids_prefix_EOS_roundtrip=True,
                assets=ref(out/'verified-assets.json'),parent_adapter_files=p['parent_adapter_files'],
                row_audit=ref(out/'row-audit.json'),plan=ref(plan_path),source=ref(__file__),
                no_model_loaded=True,GPU_calls=0,held_gold_reads=0,quality_calls=0)
    assert result['tokens']==519954 and result['supervised']==9344 and result['max_tokens']==5534
    save(out/'complete.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.plan,a.output)
