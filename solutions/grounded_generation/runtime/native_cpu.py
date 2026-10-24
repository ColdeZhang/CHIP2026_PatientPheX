"""Replay every new input in the allocated native runtime, with no model weights."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def default(value):
    if isinstance(value,(set,frozenset)):return sorted(value,key=repr)
    raise TypeError('Unsupported metadata container: '+type(value).__name__)


def sync(parent):
    parent=Path(parent).resolve()
    while True:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if parent.parent==parent:return
        parent=parent.parent


def save(path,value):
    path=Path(path)
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=default);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    sync(path.parent)


def observe(path,value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,default=default)+'\n');handle.flush();os.fsync(handle.fileno())


def ref(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();started=time.monotonic();plan=json.loads(Path(args.plan).read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for name,item in plan['files'].items():assert ref(name)['bytes']==item['bytes'] and sha(name)==item['sha256']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);sync(out.parent)
    import torch,transformers,peft,tokenizers
    from transformers import AutoTokenizer
    from transformers.loss.loss_utils import fixed_cross_entropy
    from tokenizers import Tokenizer
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from native_tokenizer import encode
    import indexed_head
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),cuda_available=torch.cuda.is_available(),
        modeling=ref(modeling.__file__),indexed_head=ref(indexed_head.__file__),no_model_weights=True)
    save(out/'runtime.json',runtime)
    assert runtime['torch']=='2.6.0+cu124' and runtime['transformers']=='5.20.2'
    assert runtime['peft']=='0.21.2' and runtime['tokenizers']=='0.23.1'
    assert not runtime['cuda_initialized'] and not runtime['cuda_available'] and importlib.util.find_spec('fla') is None
    assert runtime['modeling']['sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
    assert runtime['indexed_head']['sha256']=='0d28201832721c800696321c6afa1ef7831d63b697620a02290d65317a4428bb'
    model=Path(plan['model']);source=json.loads(Path(plan['model_source_manifest']).read_text())
    assert sha(plan['model_source_manifest'])=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
    assets={item['name']:item for item in source['files']}
    checked={}
    for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','generation_config.json']:
        item=assets[name];actual=ref(model/name);assert actual['bytes']==item['size'] and actual['sha256']==item['sha256'];checked[name]=actual
    save(out/'tokenizer-assets.json',checked)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    native=Tokenizer.from_file(str(model/'tokenizer.json'))
    train=[json.loads(line) for line in Path(plan['train']).read_text().splitlines()]
    held=[json.loads(line) for line in Path(plan['held_input']).read_text().splitlines()]
    expected_train={r['row_id']:r for r in json.loads(Path(plan['train_audit']).read_text())}
    expected_held={r['row_id']:r for r in json.loads(Path(plan['held_audit']).read_text())}
    assert len(train)==87 and len(held)==39 and set(expected_train)=={r['row_id'] for r in train} and set(expected_held)=={r['row_id'] for r in held}
    train_encoded=[];held_encoded=[]
    for phase,rows,expected in [('train',train,expected_train),('held',held,expected_held)]:
        for row in rows:
            wanted=expected[row['row_id']]
            prefix=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(prefix,add_special_tokens=False);independent=native.encode(prefix,add_special_tokens=False).ids
            checks=dict(native_prefix_IDs_equal=ids==independent,
                native_prefix_decode_equal=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(independent,skip_special_tokens=False),
                rendered_prefix_SHA_equal=hashlib.sha256(prefix.encode()).hexdigest()==wanted['rendered_prefix_sha256'])
            if phase=='train':
                value=encode(row,tokenizer);full_text=tokenizer.apply_chat_template(row['messages']+[{'role':'assistant','content':row['answer']}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
                full_native=native.encode(full_text,add_special_tokens=False).ids
                checks.update(full_IDs_equal=value['input_ids']==full_native,
                    all_local_IDs_SHA_equal=digest(value['input_ids'])==wanted['input_ids_sha256'],
                    all_local_labels_SHA_equal=digest(value['labels'])==wanted['labels_sha256'],
                    local_full_text_SHA_equal=value['rendered_full_sha256']==wanted['rendered_full_sha256'],
                    prefix_count_equal=value['prefix_tokens']==wanted['prefix_tokens'],
                    supervised_count_equal=value['supervised_tokens']==wanted['supervised_tokens'],
                    full_decode_equal=tokenizer.decode(value['input_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(full_native,skip_special_tokens=False))
                train_encoded.append(value)
            else:
                assert 'answer' not in row
                checks.update(all_local_IDs_SHA_equal=digest(ids)==wanted['input_ids_sha256'],input_count_equal=len(ids)==wanted['input_tokens'])
                held_encoded.append(dict(row_id=row['row_id'],input_ids=ids))
            observation=dict(phase=phase,row_id=row['row_id'],checks=checks,passes=all(checks.values()))
            observe(out/'observations.jsonl',observation)
            assert observation['passes'],(phase,row['row_id'])
    # Incremental numeric contract uses actual new label positions but synthetic
    # small vocab/hidden; old broad contract and actual old GPU proof remain reused.
    torch.set_num_threads(1);torch.manual_seed(20261024)
    numerical=[]
    for index in plan['frontier_indices']:
        original=train_encoded[index];labels=torch.tensor([original['labels']],dtype=torch.long)
        selected,targets=indexed_head.selection(labels)
        expected=[i-1 for i,y in enumerate(original['labels']) if y!=-100]
        assert selected.tolist()==expected and targets.numel()==original['supervised_tokens']
        simulated=labels.clone();simulated[simulated!=-100]%=37
        for dtype in [torch.float32,torch.bfloat16]:
            hidden=torch.randn(1,len(original['input_ids']),8,dtype=dtype)
            head=torch.nn.Linear(8,37,bias=False,dtype=dtype);head.requires_grad_(False)
            def native_loss(logits,gold,vocab):
                shifted=torch.nn.functional.pad(gold,(0,1),value=-100)[...,1:].contiguous()
                return fixed_cross_entropy(logits.reshape(-1,vocab).float(),shifted.reshape(-1),num_items_in_batch=None,ignore_index=-100)
            value=indexed_head.frozen_head_observations(hidden,simulated,head,native_loss,fixed_cross_entropy)
            value.update(original_row_index=index,row_id=original['row_id'],actual_supervised_positions_exact=True,
                         synthetic_vocabulary=37,synthetic_hidden=8,not_actual_model_logits=True)
            name=f'numeric-{index}-{str(dtype).split(".")[-1]}.json';save(out/name,value)
            assert value['passes'],name;numerical.append(ref(out/name))
    packed=gzip.compress((json.dumps(dict(train=train_encoded,held=held_encoded),separators=(',',':'))+'\n').encode(),mtime=0)
    with (out/'encoded.json.gz').open('xb') as handle:handle.write(packed);handle.flush();os.fsync(handle.fileno())
    sync(out)
    result=dict(status='PASS_NATIVE_FULL_INPUT_AND_INDEXED_POSITION_CPU',plan=ref(args.plan),source=ref(__file__),runtime=ref(out/'runtime.json'),
        tokenizer_assets=ref(out/'tokenizer-assets.json'),observations=ref(out/'observations.jsonl'),encoded=ref(out/'encoded.json.gz'),
        train_rows=87,held_rows=39,all_local_native_actual_IDs_labels_rendering_exact=True,
        complete_inputs_and_answers=True,numerical=numerical,old_broad_numerical_contract_reused=plan['old_indexed_review'],
        no_weights_no_GPU_no_fit_no_quality=True,seconds=time.monotonic()-started)
    save(out/'result.json',result);print(json.dumps(ref(out/'result.json')),flush=True)


if __name__=='__main__':main()
