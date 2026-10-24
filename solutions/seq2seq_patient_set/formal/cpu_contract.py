"""Native base identity/126 SP IDs; tiny native full-state save/reopen/restore. CPU only."""
import argparse
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import ref,save,tensor_compare
from state_io import save_reopen_restore
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from probe import restore_shared_input_identity
import quote_codec


def main(a):
    import numpy as np
    import torch,transformers,tokenizers,sentencepiece as spm
    from transformers import LongT5Config,LongT5ForConditionalGeneration,get_linear_schedule_with_warmup
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1)
    save(out/'runtime.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,
        'sentencepiece':spm.__version__,'CUDA_initialized':torch.cuda.is_initialized()})
    assert transformers.__version__=='4.50.0' and not torch.cuda.is_initialized()
    model,loading=LongT5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
    save(out/'loading.json',loading);assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    restore_shared_input_identity(model,out);assert sum(x.numel() for x in model.parameters())==247587456
    del model
    sp=spm.SentencePieceProcessor(model_file=str(Path(a.model)/'spiece.model'));counts={}
    for split,n in [('train',87),('held_input',39)]:
        rows=[json.loads(x) for x in gzip.decompress((Path(a.supply)/(split+'.jsonl.gz')).read_bytes()).splitlines()];assert len(rows)==n
        for r in rows:
            assert sp.encode(r['source_text'],out_type=int)+[1]==r['input_ids']
            if split=='train':assert quote_codec.encode(sp,r['target_text'])[0]==r['labels'] and quote_codec.decode(sp,r['labels'])==r['target_text']
        counts[split]=len(rows)
    # Native tiny architecture exercises the same tied-input/untied-head state APIs.
    random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024)
    config=LongT5Config(vocab_size=64,d_model=16,d_kv=4,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=4,
        local_radius=3,global_block_size=4,encoder_attention_type='transient-global',tie_word_embeddings=False,
        pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,dropout_rate=0.)
    model=LongT5ForConditionalGeneration(config);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);scheduler=get_linear_schedule_with_warmup(opt,1,2)
    group_sizes=[]
    for indices in (range(4),range(4,7)):
        opt.zero_grad(set_to_none=True);group_sizes.append(len(indices))
        for index in indices:
            loss=model(input_ids=torch.tensor([[3,4,5,6,7,1]]),labels=torch.tensor([[8+index,1]])).loss
            (loss/len(indices)).backward()
        assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
        opt.step();scheduler.step()
    ack=save_reopen_restore(model,opt,scheduler,out,{'actual_adam_updates':2,'epochs':1,'scope':'tiny_CPU_native_contract_not_formal'})
    state=torch.load(out/'final-full-state.pt',map_location='cpu',weights_only=False)
    restored=LongT5ForConditionalGeneration(config);restored.load_state_dict(state['model'],strict=True)
    tensor_compare(restored.state_dict(),state['model']);assert restored.shared.weight is restored.encoder.embed_tokens.weight is restored.decoder.embed_tokens.weight
    assert restored.lm_head.weight is not restored.shared.weight and ack['every_optimizer_step']==[2]
    # Different group-size scaling and a corrupt tensor must be distinguishable.
    actual=sum((i+1)/3 for i in range(3));assert actual==2 and actual!=sum((i+1)/4 for i in range(3))
    try:tensor_compare(torch.tensor([1.]),torch.tensor([2.]));raise RuntimeError('Corruption guard missed')
    except AssertionError:pass
    assert not torch.cuda.is_initialized()
    save(out/'complete.json',{'status':'PASS_TARGET_NATIVE_BASE_SP126_AND_TINY_FULL_STATE_CPU','counts':counts,'native_base_parameters':247587456,
        'group_sizes':group_sizes,'state_ACK':ref(out/'final-state-ACK.json'),'fresh_model_reload_exact':True,
        'all87_target_byte_roundtrip':True,'actual_GPU_calls':0,'formal_training_calls':0,'seconds':time.monotonic()-start,
        'runtime':ref(out/'runtime.json'),'source':ref(__file__),
        'assets':{n:ref(Path(a.model)/n) for n in ('config.json','pytorch_model.bin','spiece.model','tokenizer.json')},
        'supply':{n:ref(Path(a.supply)/n) for n in ('manifest.json','train.jsonl.gz','held_input.jsonl.gz')}})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('model','supply','output'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
