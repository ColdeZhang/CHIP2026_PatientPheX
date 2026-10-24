"""Prepared CPU-only census. Never train/load a model or read held-out gold."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
def sha(data): return hashlib.sha256(data).hexdigest()
def jsha(data): return sha(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--base',required=True)
    args=parser.parse_args();p=Path(args.plan);root=p.parent;plan=json.loads(p.read_text());base=Path(args.base)
    assert plan['status']=='PREPARED_CPU_ONLY_NOT_EXECUTED_NO_TRAINING_AUTHORIZATION'
    assert not (root/'native-result.json').exists() and not (root/'native-row-audit.json').exists()
    for name,record in plan['files'].items():
        b=(root/name).read_bytes();assert len(b)==record['bytes'] and sha(b)==record['sha256']
    save(root/'native-CPU-intent.json',dict(plan_sha256=sha(p.read_bytes()),
         source_sha256=sha(Path(__file__).read_bytes()),pid=os.getpid(),
         operation='one CPU tokenizer census only; unknown/failed status is not permission to repeat'))
    asset_actual={}
    for name,record in plan['tokenizer_files'].items():
        b=(base/name).read_bytes();assert len(b)==record['bytes'] and sha(b)==record['sha256']
        asset_actual[name]=dict(bytes=len(b),sha256=sha(b))
    import torch,transformers,tokenizers
    assert torch.__version__==plan['runtime']['torch'] and transformers.__version__==plan['runtime']['transformers']
    assert tokenizers.__version__==plan['runtime']['tokenizers'];torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    auto=transformers.AutoTokenizer.from_pretrained(base,local_files_only=True)
    native=tokenizers.Tokenizer.from_file(str(base/'tokenizer.json'))
    spec=importlib.util.spec_from_file_location('qwen35_smoke',root/'qwen35_smoke.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    groups=[]
    for name,refkey in [('original124.jsonl.gz','original_train'),('expanded124.jsonl.gz','expanded_train')]:
        b=gzip.decompress((root/name).read_bytes());record=plan[refkey]
        assert len(b)==record['bytes'] and sha(b)==record['sha256']
        groups.append([json.loads(line) for line in b.decode().splitlines()])
    original,expanded=groups;assert len(original)==len(expanded)==124
    old_encoded=[];new_encoded=[];audits=[]
    for index,(old,new) in enumerate(zip(original,expanded)):
        assert old['answer'].encode()==new['answer'].encode()
        assert {k:v for k,v in old.items() if k!='messages'}=={k:v for k,v in new.items() if k!='messages'}
        assert [m['role'] for m in old['messages']]==[m['role'] for m in new['messages']]==['system','user']
        assert new['messages'][:-1]==old['messages'][:-1] and new['messages'][-1]['content'].startswith(old['messages'][-1]['content']+plan['append_prefix'])
        item=dict(index=index,row_id=old['row_id'],pmc_id=str(old['pmc_id']),answer_utf8_sha256=sha(old['answer'].encode()))
        for key,row,collection in [('original',old,old_encoded),('expanded',new,new_encoded)]:
            encoded=helper.encode(row,auto);collection.append(encoded)
            prefix=row['messages'];full=prefix+[dict(role='assistant',content=row['answer'])]
            for label,messages,gen in [('prefix',prefix,True),('full',full,False)]:
                rendered=auto.apply_chat_template(messages,tokenize=False,add_generation_prompt=gen,enable_thinking=False)
                ids=auto.encode(rendered,add_special_tokens=False)
                templated=auto.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=gen,enable_thinking=False)
                assert ids==templated==native.encode(rendered,add_special_tokens=False).ids
                assert auto.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(ids,skip_special_tokens=False)==rendered
                if label=='full':assert ids==encoded['input_ids']
                else:assert ids==encoded['input_ids'][:encoded['prefix_tokens']]
            item[key]=dict(full_tokens=len(encoded['input_ids']),prefix_tokens=encoded['prefix_tokens'],
                           supervised_tokens=encoded['supervised_tokens'],input_ids_sha256=jsha(encoded['input_ids']),
                           labels_sha256=jsha(encoded['labels']),rendered_full_sha256=encoded['rendered_full_sha256'])
        a,b=old_encoded[-1],new_encoded[-1]
        assert a['input_ids'][a['prefix_tokens']:]==b['input_ids'][b['prefix_tokens']:]
        item['assistant_supervised_IDs_exact']=True;item['native_Auto_template_and_decode_exact']=True;audits.append(item)
    assert jsha(old_encoded)==plan['original_encoded_sha256']
    assert sum(len(e['input_ids']) for e in old_encoded)==402594 and max(len(e['input_ids']) for e in old_encoded)==5523
    assert sum(e['supervised_tokens'] for e in old_encoded)==sum(e['supervised_tokens'] for e in new_encoded)==7170
    def stats(encoded):
        values=[len(e['input_ids']) for e in encoded]
        return dict(total=sum(values),minimum=min(values),median=statistics.median(values),maximum=max(values),
                    longest_indices=[i for i,v in enumerate(values) if v==max(values)],
                    over_caps={str(cap):sum(v>cap for v in values) for cap in [6144,8192,12288,16384,32768]},
                    sum_squared_lengths=sum(v*v for v in values),supervised_tokens=sum(e['supervised_tokens'] for e in encoded))
    oldstats,newstats=stats(old_encoded),stats(new_encoded)
    cap=((newstats['maximum']+511)//512)*512
    context=json.loads((base/'config.json').read_text())['text_config']['max_position_embeddings']
    ratio=newstats['total']/oldstats['total']
    result=dict(status='PASS_NATIVE_CPU_TRAIN124_FULL_REPORT_COST_AND_MASK_ONLY_NO_TRAINING',
        rows=124,training_documents=32,original=oldstats,expanded=newstats,token_ratio=ratio,
        squared_length_ratio=newstats['sum_squared_lengths']/oldstats['sum_squared_lengths'],
        proposed_rejection_cap=cap,model_context=context,max_within32768=newstats['maximum']<=32768,
        cap_within_context=cap<=context,training_authorized=False,training_performed=False,
        all_assistant_supervision_IDs_unchanged=True,all124_published_native_Auto_template_decode_exact=True,
        truncation=False,model_loaded=False,validation_gold_read=False,cuda_initialized=torch.cuda.is_initialized(),
        actual_tokenizer_assets=asset_actual,runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
          tokenizers=tokenizers.__version__,tokenizer_class=type(auto).__name__),
        original_encoded_sha256=jsha(old_encoded),expanded_encoded_sha256=jsha(new_encoded),
        old_actual_training_seconds=1149.460371017456,two_epochs_tokens=2*newstats['total'],
        linear_two_epoch_seconds_PROXY_NOT_MEASURED=1149.460371017456*ratio,
        proxy_limits='Same A800 and old training implementation; selected CE/long attention/full-head/backward/checkpoint costs unmeasured; not latency or memory guarantee.',
        plan_sha256=sha(p.read_bytes()),source_sha256=sha(Path(__file__).read_bytes()),elapsed_seconds=time.monotonic()-started)
    save(root/'native-row-audit.json',audits);save(root/'native-result.json',result)
    print(json.dumps(result))
if __name__=='__main__':main()
