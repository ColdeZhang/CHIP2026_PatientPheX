"""One complete local native-tokenizer census; no weights, fit, or labels from held data."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

from prepare import ref, put


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def stats(values):
    return dict(sum=sum(values),minimum=min(values),maximum=max(values),median=statistics.median(values),mean=statistics.mean(values))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();started=time.monotonic();plan=json.loads(args.plan.read_text())
    assert plan['cpu_budget_seconds']==120
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU token census120s')));signal.alarm(120)
    for mapping in [plan['references'],plan['sources']]:
        for item in mapping.values():assert ref(item['path'])==item
    assert plan['sources']['token_census.py']==ref(__file__)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,plan['lightweight_libraries'])
    from native_tokenizer import FrozenTokenizer, encode
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    put(out/'intent.json',dict(plan=ref(args.plan),no_model_no_GPU_no_quality=True))
    try:
        refs=plan['references'];manifest=json.loads(Path(refs['target_manifest']['path']).read_text())
        assert manifest['status']=='COMPLETE_CPU_WEAK_TARGETS_NO_MODEL_NO_QUALITY'
        for name in ['train','held_input']:
            assert manifest['artifacts'][name]==refs[name]
        train=[json.loads(line) for line in Path(refs['train']['path']).read_text().splitlines()]
        held=[json.loads(line) for line in Path(refs['held_input']['path']).read_text().splitlines()]
        assert len(train)==87 and len(held)==39
        tokenizer=FrozenTokenizer(refs['tokenizer']['path'],refs['template']['path'])
        audits=[];encoded=[]
        for index,row in enumerate(train):
            value=encode(row,tokenizer);prefix=value['prefix_tokens'];tokens=value['input_ids'];labels=value['labels']
            assert tokens==tokenizer.backend.encode(tokenizer.apply_chat_template(row['messages']+[{'role':'assistant','content':row['answer']}],tokenize=False,add_generation_prompt=False,enable_thinking=False),add_special_tokens=False).ids
            assert labels[:prefix]==[-100]*prefix and labels[prefix:]==tokens[prefix:]
            positions=[i-1 for i,label in enumerate(labels) if label!=-100]
            assert min(positions)>=0 and len(positions)==value['supervised_tokens']
            audits.append(dict(index=index,row_id=row['row_id'],pmc_id=row['pmc_id'],target_patient_id=row['target_patient_id'],
                full_tokens=len(tokens),prefix_tokens=prefix,supervised_tokens=value['supervised_tokens'],
                predecessor_positions=len(positions),input_ids_sha256=digest(tokens),labels_sha256=digest(labels),
                predecessor_positions_sha256=digest(positions),rendered_full_sha256=value['rendered_full_sha256'],
                rendered_prefix_sha256=value['rendered_prefix_sha256'],eos_token_id=value['eos_token_id'],
                eos_supervised_position=value['eos_supervised_position']))
            encoded.append(value)
        held_audits=[];held_encoded=[]
        for row in held:
            assert 'answer' not in row
            rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            tokens=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert tokens==tokenizer.backend.encode(rendered,add_special_tokens=False).ids
            held_audits.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],input_tokens=len(tokens),
                input_ids_sha256=digest(tokens),rendered_prefix_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
            held_encoded.append(dict(row_id=row['row_id'],input_ids=tokens))
        payload=(json.dumps(dict(train=encoded,held=held_encoded),ensure_ascii=False,separators=(',',':'))+'\n').encode()
        binary=out/'encoded.json.gz'
        with binary.open('xb') as handle:
            handle.write(gzip.compress(payload,mtime=0));handle.flush();os.fsync(handle.fileno())
        fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        frontier=[row for row in audits if not any(other['full_tokens']>=row['full_tokens'] and other['supervised_tokens']>=row['supervised_tokens'] and (other['full_tokens']>row['full_tokens'] or other['supervised_tokens']>row['supervised_tokens']) for other in audits)]
        artifacts=dict(encoded=ref(binary),train_audit=put(out/'train-audit.json',audits),held_audit=put(out/'F1-input-audit.json',held_audits))
        result=dict(status='COMPLETE_LOCAL_NATIVE_TOKENIZER_CENSUS',plan=ref(args.plan),sources=plan['sources'],references=refs,
            artifacts=artifacts,train_rows=87,train_documents=32,held_rows=39,held_documents=16,
            full_training_tokens=stats([r['full_tokens'] for r in audits]),prefix_training_tokens=stats([r['prefix_tokens'] for r in audits]),
            supervised_tokens=stats([r['supervised_tokens'] for r in audits]),held_input_tokens=stats([r['input_tokens'] for r in held_audits]),
            max_length_row=max(audits,key=lambda r:r['full_tokens']),max_supervision_row=max(audits,key=lambda r:r['supervised_tokens']),
            joint_length_supervision_frontier=frontier,no_input_or_answer_truncation=True,all_labels_mask_prefix_and_supervise_EOS=True,
            all_causal_positions_q_minus_one_exact=True,published_tokenizer_all_actual_ids=True,
            native_Transformers_runtime_replay_still_required=True,no_model_no_GPU_no_fit_no_quality=True,seconds=time.monotonic()-started)
        print(json.dumps(put(out/'manifest.json',result)))
    except BaseException:
        put(out/'failure.json',dict(traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
