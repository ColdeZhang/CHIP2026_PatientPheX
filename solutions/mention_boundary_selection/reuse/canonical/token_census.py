"""Full-template native token/mask census for one frozen tokenizer; no model."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from common import ref,write,load_jsonl,validate_refs,digest
from reuse.native_tokenizer import encode

class NativeTokenizer:
    """Same frozen text renderer, parameterized EOS for Qwen3 vs Qwen3.5 only."""
    def __init__(self,tokenizer_file,template_text,eos_id,tokenizers_version='0.23.1'):
        import tokenizers,jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        assert tokenizers.__version__==tokenizers_version and jinja2.__version__=='3.1.6'
        self.backend=tokenizers.Tokenizer.from_file(str(tokenizer_file))
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def fail(message):raise ValueError(message)
        env.globals['raise_exception']=fail
        self.template=env.from_string(template_text)
        self.eos_token='<|im_end|>';self.eos_token_id=self.backend.token_to_id(self.eos_token)
        assert self.eos_token_id==eos_id
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking,return_dict=False):
        assert enable_thinking is False and return_dict is False
        text=self.template.render(messages=messages,tools=None,add_generation_prompt=add_generation_prompt,enable_thinking=False)
        return self.backend.encode(text,add_special_tokens=False).ids if tokenize else text

def stats(values):return dict(sum=sum(values),minimum=min(values),maximum=max(values),median=statistics.median(values))

def census(plan,out):
    start=time.monotonic();validate_refs(plan['inputs']);r=plan['inputs']
    train=load_jsonl(r['train']['path']);held=load_jsonl(r['held_input']['path']);assert len(train)==87 and len(held)==39
    assert all('answer' not in x for x in held)
    template=(Path(r['template']['path']).read_text() if plan['template_kind']=='file'
              else json.loads(Path(r['template']['path']).read_text())['chat_template'])
    tok=NativeTokenizer(r['tokenizer']['path'],template,plan['eos_token_id'],plan.get('tokenizers_version','0.23.1'))
    encoded=[];audits=[];held_encoded=[];held_audits=[]
    # No trie rebuilding/retesting here: this increment is the complete real chat boundary (H38).
    for i,row in enumerate(train):
        v=encode(row,tok);pre=v['prefix_tokens'];ids=v['input_ids'];labs=v['labels']
        prefix=tok.apply_chat_template(row['messages'],False,True,False)
        full=tok.apply_chat_template(row['messages']+[dict(role='assistant',content=row['answer'])],False,False,False)
        suffix=tok.backend.encode(row['answer']+tok.eos_token+'\n',add_special_tokens=False).ids
        assert tok.backend.decode(ids,skip_special_tokens=False)==full
        assert full==prefix+row['answer']+tok.eos_token+'\n' and ids[pre:]==suffix
        positions=[j-1 for j,value in enumerate(labs) if value!=-100]
        assert min(positions)>=0 and labs==[-100]*pre+ids[pre:]
        audits.append(dict(index=i,row_id=row['row_id'],pmc_id=row['pmc_id'],target_patient_id=row['target_patient_id'],
            full_tokens=len(ids),prefix_tokens=pre,supervised_tokens=v['supervised_tokens'],predecessor_positions=len(positions),
            input_ids_sha256=digest(ids),labels_sha256=digest(labs),predecessor_positions_sha256=digest(positions),
            rendered_prefix_sha256=v['rendered_prefix_sha256'],rendered_full_sha256=v['rendered_full_sha256'],
            eos_token_id=v['eos_token_id'],eos_supervised_position=v['eos_supervised_position'],
            full_vs_prefix_plus_suffix_native_exact=True))
        encoded.append(v)
    for row in held:
        prefix=tok.apply_chat_template(row['messages'],False,True,False);ids=tok.apply_chat_template(row['messages'],True,True,False)
        assert tok.backend.decode(ids,skip_special_tokens=False)==prefix
        held_audits.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],input_tokens=len(ids),input_ids_sha256=digest(ids),rendered_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest()))
        held_encoded.append(dict(row_id=row['row_id'],input_ids=ids))
    frontier=[r for r in audits if not any(q['full_tokens']>=r['full_tokens'] and q['supervised_tokens']>=r['supervised_tokens'] and (q['full_tokens']>r['full_tokens'] or q['supervised_tokens']>r['supervised_tokens']) for q in audits)]
    first=list(range(87));random.Random(20261023).shuffle(first)
    groups=[first[i:i+8] for i in range(0,87,8)];assert list(map(len,groups))==[8]*10+[7]
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    packed=gzip.compress((json.dumps(dict(train=encoded,held=held_encoded),separators=(',',':'))+'\n').encode(),mtime=0)
    with (out/'encoded.json.gz').open('xb') as h:h.write(packed);h.flush();os.fsync(h.fileno())
    arts=dict(encoded=ref(out/'encoded.json.gz'),train_audit=write(out/'train-audit.json',audits),held_audit=write(out/'F1-input-audit.json',held_audits))
    maxfull=max(x['full_tokens'] for x in audits);maxheld=max(x['input_tokens'] for x in held_audits)
    result=dict(status='COMPLETE_LOCAL_NATIVE_CANONICAL_TEMPLATES',model=plan['model'],source=ref(__file__),inputs=r,
        artifacts=arts,train_rows=87,held_rows=39,train_documents=32,held_documents=16,
        full_training_tokens=stats([r['full_tokens'] for r in audits]),prefix_training_tokens=stats([r['prefix_tokens'] for r in audits]),
        supervised_tokens=stats([r['supervised_tokens'] for r in audits]),held_input_tokens=stats([r['input_tokens'] for r in held_audits]),
        shortest_row=min(audits,key=lambda r:r['full_tokens']),longest_row=max(audits,key=lambda r:r['full_tokens']),
        max_supervision_row=max(audits,key=lambda r:r['supervised_tokens']),joint_length_supervision_frontier=frontier,
        proposed_training_cap=math.ceil(maxfull/256)*256,proposed_inference_total_cap=math.ceil((maxheld+512)/256)*256,
        actual_epoch1_groups=groups,probe_first_real_group=groups[0],probe_accumulate_next_three=groups[1][:3],
        configuration=dict(rows=87,documents=32,epochs=2,steps_per_epoch=11,optimizer_calls=22,warmup_steps=2,
            batch_size=1,gradient_accumulation=8,tail_group_size=7,seed=20261023,learning_rate=1e-4,
            weight_decay=.01,max_grad_norm=1.,rank=16,lora_alpha=32,dropout=.05,dtype='bfloat16',attention='sdpa',
            gradient_checkpointing={'use_reentrant':False},packing=False,quantization=False,
            loss='standard per-row mean causal CE; divide each row by actual group size',
            full_training_tokens=sum(len(v['input_ids']) for v in encoded),supervised_tokens=sum(v['supervised_tokens'] for v in encoded),maximum_full_tokens=maxfull),
        all87_full_prefix_suffix_IDs_exact=True,all87_full_labels_EOS_newline_causal_positions_exact=True,
        complete_all126_no_truncation=True,no_model_no_GPU_no_quality=True,actual_HF_runtime_replay_pending=True,seconds=time.monotonic()-start)
    write(out/'manifest.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());result=census(plan,a.output);print(json.dumps(dict(result=ref(Path(a.output)/'manifest.json'),full=result['full_training_tokens'],supervised=result['supervised_tokens'],held=result['held_input_tokens'])))
