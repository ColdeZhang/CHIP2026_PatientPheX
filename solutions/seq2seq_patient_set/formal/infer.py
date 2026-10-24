"""Complete39 greedy/cap1600 from the sole final state; raw IDs before decoding."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from common import ref,save,fsync_dir
from runtime import bind,same
from transport import decode_generated
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import quote_codec


def main(a):
    p,b,out,modeldir,rows=bind(a.plan,a.binding,'infer');start=time.monotonic();stage='checkpoint_load'
    try:
        import torch,transformers,sentencepiece as spm
        from transformers import LongT5Config,LongT5ForConditionalGeneration,GenerationConfig,LogitsProcessor,LogitsProcessorList
        assert (torch.__version__,transformers.__version__)==(b['torch'],b['transformers'])
        assert transformers.__version__=='4.50.0';torch.set_num_threads(1)
        for name in ('checkpoint','state_ACK','training_complete'):same(b[name]['path'],b[name])
        ack=json.loads(Path(b['state_ACK']['path']).read_text());training=json.loads(Path(b['training_complete']['path']).read_text())
        assert ack['status']=='FULL_STATE_FSYNC_REOPEN_RESTORE_PASS' and ack['every_optimizer_step']==[220]
        assert ack['checkpoint']==b['checkpoint']==training['checkpoint'] and training['state_ACK']==b['state_ACK']
        assert training['status']=='COMPLETE_FRESH_10EPOCH_870ROWS_220ADAM_DURABLE_STATE'
        same(a.plan,training['plan']);state=torch.load(b['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert state['epochs']==10 and state['actual_adam_updates']==220 and len(state['visits'])==870 and not state['probe_state_used']
        model=LongT5ForConditionalGeneration(LongT5Config.from_pretrained(modeldir,local_files_only=True))
        loading=model.load_state_dict(state['model'],strict=True);save(out/'loading.json',{'missing_keys':loading.missing_keys,'unexpected_keys':loading.unexpected_keys})
        assert not loading.missing_keys and not loading.unexpected_keys
        assert model.shared.weight is model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight
        assert model.lm_head.weight is not model.shared.weight and sum(x.numel() for x in model.parameters())==247587456
        del state;gc.collect();assert time.time()<b['admit_until']
        assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();model.to('cuda');model.eval();model.config.use_cache=True
        sp=spm.SentencePieceProcessor(model_file=str(modeldir/'spiece.model'))
        save(out/'prior-generation-config.json',model.generation_config.to_dict())
        model.generation_config=GenerationConfig(pad_token_id=0,eos_token_id=1,decoder_start_token_id=0)
        config=GenerationConfig(pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,do_sample=False,num_beams=1,max_new_tokens=1600,use_cache=True)
        prepared,unused=model._prepare_generation_config(config)
        save(out/'generation-config.json',{'prepared':prepared.to_dict(),'unused':unused})
        assert not unused and not prepared.do_sample and prepared.num_beams==1 and prepared.max_new_tokens==1600
        assert prepared.min_p is None and prepared.typical_p==1.
        class Finite(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite unmasked generation logits')
                return scores
        gens=[];raws=[];seen=[];decode_errors=0;stage='complete39_generation'
        (out/'generation').mkdir()
        with (out/'responses.jsonl').open('x') as response_stream:
            for index,row in enumerate(rows):
                assert sp.encode(row['source_text'],out_type=int)+[1]==row['input_ids']
                ids=torch.tensor([row['input_ids']],device='cuda');torch.cuda.synchronize();tick=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    value=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=config,logits_processor=LogitsProcessorList([Finite()]))
                torch.cuda.synchronize();allids=value[0].tolist();assert allids[0]==0;suffix=allids[1:]
                gen={'index':index,'row_id':row['row_id'],'pmc_id':row['pmc_id'],'target_patient_id':row['patient_id'],
                    'source_sha256':row['source_sha256'],'input_token_count':len(row['input_ids']),
                    'all_generated_ids':allids,'generated_token_ids':suffix,'decoder_start_removed':0,'output_tokens':len(suffix),
                    'actual_last_token_eos':bool(suffix and suffix[-1]==1),'hit_token_limit':len(suffix)>=1600,
                    'seconds':time.monotonic()-tick,'decoded_or_parsed':False}
                gp=out/'generation'/f'{index:04d}.ids.json';save(gp,gen);gens.append(ref(gp))
                # Retain native diagnostic text even when strict transport cannot decode.
                # Native diagnostic excludes only IDs outside the native SP inventory;
                # those IDs remain verbatim above and are not silently accepted.
                native=sp.decode([x for x in suffix if 0<=x<sp.get_piece_size()]);response=None;error=None
                try:response=decode_generated(sp,suffix)
                except ValueError as exc:error=str(exc);decode_errors+=1
                raw={**gen,'generation':ref(gp),'native_sp_diagnostic':native,'response':response,'transport_error':error,
                     'raw_before_parse':True,'parsed':False,'codec':p['sources']['formal/transport.py'],'SP_model':p['model']['spiece.model']}
                rp=out/'generation'/f'{index:04d}.raw.json';save(rp,raw);raws.append(ref(rp));seen.append(row['row_id'])
                response_stream.write(json.dumps(raw,ensure_ascii=False)+'\n');response_stream.flush();os.fsync(response_stream.fileno())
                save(out/'progress.json',{'rows':len(seen),'total':39,'latest_row':row['row_id'],'seconds':time.monotonic()-start})
                del ids,value
        assert seen==[x['row_id'] for x in rows];fsync_dir(out/'generation');fsync_dir(out)
        save(out/'complete.json',{'status':'COMPLETE_LONGT539_FINAL_EPOCH10_GREEDY1600_RAW','rows':39,'documents':16,
            'row_ids':seen,'generation_refs':gens,'raw_refs':raws,'responses':ref(out/'responses.jsonl'),'source':ref(__file__),
            'plan':ref(a.plan),'binding':ref(a.binding),'supply':p['supply'],'checkpoint':b['checkpoint'],'state_ACK':b['state_ACK'],
            'transport_error_rows':decode_errors,'seconds':time.monotonic()-start,'no_regeneration_or_answer_repair':True,
            'quality_calls':0,'parser_calls':0,'torch':torch.__version__,'transformers':transformers.__version__})
    except BaseException:
        save(out/'failure.json',{'stage':stage,'seconds':time.monotonic()-start,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',required=True);q.add_argument('--binding',required=True);main(q.parse_args())
