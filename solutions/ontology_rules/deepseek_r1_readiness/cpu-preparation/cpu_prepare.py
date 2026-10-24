"""Exact R1 tokenizer/template input-only audit and deterministic3 selection. No model load."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,os,platform,time,traceback
from strict_final import tests

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha_bytes(p.read_bytes()))
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def encoded_sha(ids):return sha_bytes(json.dumps(ids,separators=(',',':')).encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();planpath=Path(a.plan).resolve();base=planpath.parent;plan=json.loads(planpath.read_text());start=time.monotonic();out=base/'actual';out.mkdir(exist_ok=False)
    result=dict(status='running_CPU_input_only_no_model',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),plan=ref(planpath),no_gold_model_GPU_or_quality=True)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getuid()==plan['expected_uid']
        for name,expected in plan['sources'].items():assert sha_bytes((base/name).read_bytes())==expected
        model=Path(plan['model_dir']);assets={}
        for e in plan['required_small_assets']:
            assert not e['name'].endswith('.safetensors')
            p=model/e['name'];assert p.stat().st_size==e['bytes'] and sha_bytes(p.read_bytes())==e['sha256'];assets[e['name']]=ref(p)
        raw=gzip.decompress((base/'prompts.jsonl.gz').read_bytes());assert sha_bytes(raw)==plan['original_prompts_sha256'];rows=[json.loads(x) for x in raw.decode().splitlines()]
        assert len(rows)==80 and len({r['row_id'] for r in rows})==80 and len({r['pmc_id'] for r in rows})==16
        assert all(set(r)=={'row_id','pmc_id','code_map','patient_ids','messages'} and [m['role'] for m in r['messages']]==['system','user'] and all(type(m['content']) is str for m in r['messages']) for r in rows)
        frozen_rows=[json.dumps(r,ensure_ascii=False,separators=(',',':')).encode() for r in rows]
        import torch,transformers,tokenizers
        from transformers import AutoTokenizer
        assert not torch.cuda.is_initialized()
        tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        old_template=(base/'old-chat-template.jinja').read_text();config=json.loads((model/'config.json').read_text());tconfig=json.loads((model/'tokenizer_config.json').read_text())
        assert tok.is_fast and config['model_type']=='qwen3' and config['tie_word_embeddings'] is False and 'enable_thinking' not in tok.chat_template
        special={s:dict(ids=tok.encode(s,add_special_tokens=False),token_id=tok.convert_tokens_to_ids(s)) for s in [str(tok.bos_token),str(tok.eos_token),'<｜User｜>','<｜Assistant｜>','<think>','</think>']}
        assert tok.eos_token_id==config['eos_token_id'] and tok.bos_token_id==config['bos_token_id']
        assert all(v['ids']==[v['token_id']] and 0<=v['token_id']<config['vocab_size'] for v in special.values())
        audit=[];encoded=[];wrapping=[]
        for index,row in enumerate(rows):
            rendered=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
            assert all(m['content'] in rendered for m in row['messages'])
            expected=str(tok.bos_token)+row['messages'][0]['content']+'<｜User｜>'+row['messages'][1]['content']+'<｜Assistant｜>'
            assert rendered==expected
            assert rendered==tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)==tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            ids=tok.encode(rendered,add_special_tokens=False);native=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True)
            assert ids==native and ids[0]==tok.bos_token_id and ids[-1]==special['<｜Assistant｜>']['token_id'] and all(0<=x<config['vocab_size'] for x in ids)
            old=tok.apply_chat_template(row['messages'],chat_template=old_template,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert all(m['content'] in old for m in row['messages']) and old.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
            assert json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()==frozen_rows[index]
            a=dict(index=index,row_id=row['row_id'],pmc_id=row['pmc_id'],candidate_count=len(row['code_map']),patient_count=len(row['patient_ids']),original_row_sha256=sha_bytes(frozen_rows[index]),messages_sha256=sha_bytes(json.dumps(row['messages'],ensure_ascii=False,separators=(',',':')).encode()),native_prompt_sha256=sha_bytes(rendered.encode()),native_ids_sha256=encoded_sha(ids),native_input_tokens=len(ids),old_template_rendered_sha256=sha_bytes(old.encode()),old_native_token_count_not_measured=True,original_message_candidate_patient_values_unchanged=True)
            audit.append(a);encoded.append(ids);wrapping.append(rendered)
        ordered=sorted(range(80),key=lambda i:(len(encoded[i]),i));chosen=[0,ordered[len(ordered)//2],ordered[-1]];assert len(set(chosen))==3
        with (out/'selected-prompts.jsonl').open('x') as f:
            for i in chosen:f.write(json.dumps(rows[i],ensure_ascii=False)+'\n')
        save(out/'selected-encoded.json',dict(policy='original first, median(native length/order), longest(native length/order); no gold',rows=[dict(audit[i],selection_role=role,input_ids=encoded[i],rendered_prompt=wrapping[i]) for role,i in zip(['original_first','median_native_length','longest_native_length'],chosen)]))
        save(out/'all80-input-audit.json',audit)
        result.update(status='PASS_actual_native_R1_tokenizer_80_input_only_and_fixed3_prepared',documents=16,rows=80,candidate_occurrences=sum(len(r['code_map']) for r in rows),unique_document_HPO=sum(len(set(v for r in rows if r['pmc_id']==d for v in r['code_map'].values())) for d in {r['pmc_id'] for r in rows}),native_token_count=dict(minimum=min(map(len,encoded)),median=len(encoded[ordered[40]]),maximum=max(map(len,encoded)),sum=sum(map(len,encoded))),selected=[dict(role=role,**audit[i],max_new_tokens=1024,max_total_tokens=len(encoded[i])+1024) for role,i in zip(['original_first','median_native_length','longest_native_length'],chosen)],special_tokens=special,tokenizer_class=type(tok).__name__,tokenizer_config_class=tconfig['tokenizer_class'],native_enable_thinking_ignored_exact_all80=True,native_generation_prefix='native <｜Assistant｜>, no forced think prefix or closing boundary',old_template_comparison='Exact frozen Qwen3.5 old nonthinking template string rendered only; not its native tokenizer IDs. Same message contents/candidates/patients retained.',parser_tests=tests(),assets=assets,runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__),cuda_initialized=torch.cuda.is_initialized(),outputs={n:ref(out/n) for n in ['all80-input-audit.json','selected-prompts.jsonl','selected-encoded.json']},weight_tensor_values_read=False,model_loaded=False,quality_generated=False)
        assert result['cuda_initialized'] is False
    except BaseException as e:result.update(status='FAILED_CPU_PREPARATION_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result.update(seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(out/'result.json',result);print(json.dumps({k:result.get(k) for k in ['status','native_token_count','seconds','error']}),flush=True)
if __name__=='__main__':main()
