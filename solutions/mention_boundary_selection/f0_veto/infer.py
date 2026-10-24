"""Same final head and decoder, complete strong-F0 spans; no training or quality."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE/'parent'))
from gpu_entry import decode_one,native
from common import checked,save,sha,update,stamp,verify_assets,gpu_gate,load_model,memory
from schema import CandidateCodec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch=None;stage='source_and_inputs'
    try:
        for name,digest in plan['source_files'].items():assert sha(HERE/name)==digest,name
        cpu=json.loads(checked(plan['cpu_result']).read_text())
        assert cpu['status']=='PASS_ALL1085_NATIVE_F0_SAME_HEAD_INPUTS' and cpu['held_rows']==1085
        assert cpu['documents']==16 and cpu['patients']==47 and cpu['all_native_HF_IDs_rendering_exact']
        rows=native.read_encoded(cpu,'dev');assert len(rows)==1085
        assert len({r['row_id'] for r in rows})==1085
        assert len({(r['pmc_id'],r['offset'],r['length'],r['quote']) for r in rows})==1085
        assert len({r['pmc_id'] for r in rows})==16
        gpu_gate(plan,out)
        previous=json.loads(checked(plan['prior_verified_assets']).read_text())
        save(out/'verified-assets.json',verify_assets(plan,full=False,previous=previous))
        save(out/'asset-verification-scope.json',{'mode':'all13 current size/mtime_ns/inode exact plus original full-SHA certificate','prior':plan['prior_verified_assets'],'new_full_weight_hash':False})
        checked(plan['adapter_model']);checked(plan['adapter_config'])
        for item in plan['training_ACKs']:checked(item)
        assert time.time()<=float(os.environ['LAUNCH_EXPIRES_UNIX']),'Expired before first model load'
        stage='same_final_adapter_load'
        torch,np,model,named,manifest=load_model(plan,out,str(Path(plan['adapter_model']['path']).parent))
        assert manifest['loaded_adapter_tensors_exact']==504 and not named
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
        tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True)
        backend=Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json'))
        codec=CandidateCodec(backend,tk.eos_token_id,8);codec.native_backend=backend
        assert tk.eos_token_id==151645 and len(codec.outputs)==514
        raw=out/'generation';raw.mkdir();results=[];stage='complete1085_constrained_generate'
        for index,row in enumerate(rows):
            update(out/'progress.json',{'at_utc':stamp(),'completed':index,'row_id':row['row_id'],'stage':stage})
            results.append(decode_one(torch,model,tk,codec,row,raw,index,128))
        save(out/'results.json',results)
        counts={k:0 for k in ['HPO','NIL','NO','BOUNDARY','HPO_missing','HPO_empty_OTHER','rejected','cap']}
        for result in results:
            counts['cap']+=result['capped'];parsed=result['parsed']
            if parsed is None:counts['rejected']+=1;continue
            counts[parsed['status']]+=1
            if parsed['status']=='HPO':
                counts['HPO_missing']+=parsed['missing'];counts['HPO_empty_OTHER']+=not parsed['selected']
        save(out/'complete.json',{'status':'COMPLETE_FIXED_MENTION_CANDIDATE_INFERENCE',
            'documents':16,'patients':47,'processed_rows':1085,'counts':counts,
            'results_sha256':sha(out/'results.json'),'training_epoch':2,
            'at_utc':stamp(),'elapsed_seconds':time.monotonic()-start,'memory':memory(torch),
            'same_final_head':plan['adapter_model'],'no_quality_gold_read':True,'new_training_calls':0})
    except BaseException:
        save(out/'failure.json',{'stage':stage,'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-start,'memory':memory(torch) if torch is not None else None});raise

if __name__=='__main__':main()
