"""Only new F0 inputs: same native tokenizer/schema, no repeated train contract."""
import argparse, gzip, hashlib, json, os, sys, time, traceback
from pathlib import Path

P=Path(__file__).parent
sys.path.insert(0,str(P/'parent'))
from helpers import ref, write, validate_refs, digest
from schema import CandidateCodec
sys.path.insert(0,str(P/'parent/reuse/canonical'))
from token_census import NativeTokenizer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='1'
        import torch,transformers,tokenizers
        from transformers import AutoTokenizer
        torch.set_num_threads(1)
        assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        assert transformers.__version__=='4.51.3' and tokenizers.__version__=='0.21.2'
        validate_refs(plan['files']);model=Path(plan['model_dir'])
        for name,x in plan['model_assets'].items():
            r=ref(model/name);assert r['sha256']==x['sha256'] and r['bytes']==x['bytes']
        runtime={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,
                 'CUDA_initialized':torch.cuda.is_initialized(),'no_model_or_weights':True}
        write(out/'runtime.json',runtime)
        raw=gzip.decompress(Path(plan['held_gzip']['path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==plan['held_gzip']['raw_sha256']
        held=[json.loads(s) for s in raw.splitlines()];assert len(held)==1085
        tk=AutoTokenizer.from_pretrained(model,local_files_only=True)
        native=NativeTokenizer(model/'tokenizer.json',json.loads((model/'tokenizer_config.json').read_text())['chat_template'],151645,'0.21.2')
        codec=CandidateCodec(native.backend,151645,8)
        assert len(codec.outputs)==514
        encoded=[];audit=[];ids_seen=set();spans_seen=set()
        for index,row in enumerate(held):
            assert 'answer' not in row and 'target_status' not in row
            key=(row['pmc_id'],row['offset'],row['length'],row['quote'])
            assert key not in spans_seen and row['row_id'] not in ids_seen
            spans_seen.add(key);ids_seen.add(row['row_id'])
            assert len(row['quote'])==row['length']>0
            assert len(row['candidates'])==8==len({c['identifier'] for c in row['candidates']})
            assert [c['code'] for c in row['candidates']]==[f'C{i:02d}' for i in range(1,9)]
            prompt=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tk.encode(prompt,add_special_tokens=False)
            assert ids==native.backend.encode(prompt,add_special_tokens=False).ids
            assert native.apply_chat_template(row['messages'],False,True,False)==prompt
            assert native.apply_chat_template(row['messages'],True,True,False)==ids
            assert tk.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.backend.decode(ids,skip_special_tokens=False)==prompt
            assert len(ids)+128<=2176
            value={k:row[k] for k in ['row_id','pmc_id','offset','length','quote','candidates']};value['input_ids']=ids
            encoded.append(value);audit.append({'index':index,'row_id':row['row_id'],'pmc_id':row['pmc_id'],
                    'tokens':len(ids),'input_ids_sha256':digest(ids),'native_HF_IDs_rendering_exact':True})
        assert len({r['pmc_id'] for r in held})==16
        p=out/'dev.encoded.jsonl.gz';p.write_bytes(gzip.compress((''.join(json.dumps(x,separators=(',',':'))+'\n' for x in encoded)).encode(),mtime=0))
        ar=write(out/'dev.audit.json',audit)
        counts={'dev':{'rows':1085,'tokens':sum(x['tokens'] for x in audit),'max_tokens':max(x['tokens'] for x in audit),
                       'encoded':ref(p),'audit':ar}}
        result={'status':'PASS_ALL1085_NATIVE_F0_SAME_HEAD_INPUTS','plan':ref(a.plan),'source':ref(__file__),
                'counts':counts,'held_rows':1085,'documents':16,'patients':47,'all_native_HF_IDs_rendering_exact':True,
                'old514_grammar_reused_without_retesting':plan['inherited_grammar_contract'],
                'inference_total_cap':2176,'max_new_tokens':128,'eos_token_id':151645,
                'no_gold_model_GPU_fit_quality':True,'seconds':time.monotonic()-tick}
        write(out/'complete.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        write(out/'failure.json',{'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-tick});raise

if __name__=='__main__':main()
