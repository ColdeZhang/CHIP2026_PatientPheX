"""Prepared but NOT RUN: fixed 4 shape FB / 2 Adam plus tiny train-only decode."""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback


def ref(p):
    p=Path(p).resolve();data=p.read_bytes()
    return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def put(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stage='admission'
    put(out/'intent.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),argv=os.sys.argv))
    try:
        assert time.time()<a.admit_until,'Expired launch request'
        assert a.authorized_probe_only=='four-shapes-two-adam-no-formal'
        props=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        selected=[x.split(',') for x in props.splitlines() if x.split(',')[0].strip()==a.gpu_uuid]
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        put(out/'gpu-admission.json',dict(gpus=props,apps=apps,expected_uuid=a.gpu_uuid))
        assert len(selected)==1 and int(selected[0][1].strip())>=20000 and a.gpu_uuid not in apps
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
        plan=json.loads(Path(a.plan).read_text());supply=json.loads(Path(a.supply).read_text())
        assert all(ref(a.supply)[k]==plan['supply'][k] for k in ('bytes','sha256'))
        assert all(ref(__file__)[k]==plan['probe_source'][k] for k in ('bytes','sha256'))
        assert plan['selected']=='google/long-t5-tglobal-base'
        assert plan['target_mode']=='original-G87-quotes-existing-sentinel'
        import quote_codec
        assert all(ref(quote_codec.__file__)[k]==plan['codec_source'][k] for k in ('bytes','sha256'))
        modeldir=Path(a.model_dir)
        for name,item in plan['assets'].items():
            observed=ref(modeldir/name);assert all(observed[k]==item[k] for k in ('bytes','sha256')),name
        trainpath=Path(a.supply).parent/Path(supply['artifacts']['train']['path']).name
        assert all(ref(trainpath)[k]==supply['artifacts']['train'][k] for k in ('bytes','sha256'))
        train=[json.loads(x) for x in gzip.decompress(trainpath.read_bytes()).decode().splitlines()];assert len(train)==87
        assert time.time()<a.admit_until,'Expired before model load'
        stage='load'
        import torch,transformers
        from transformers import LongT5ForConditionalGeneration,GenerationConfig
        assert transformers.__version__=='4.50.0',transformers.__version__
        torch.set_num_threads(1);torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
        assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        model,loading=LongT5ForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,
            torch_dtype=torch.float32,output_loading_info=True)
        put(out/'loading.json',loading)
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        assert sum(p.numel() for p in model.parameters())==247587456
        assert model.config.is_encoder_decoder and not model.config.tie_word_embeddings
        assert model.config.pad_token_id==0 and model.config.eos_token_id==1 and model.config.decoder_start_token_id==0
        model.to('cuda');model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
        records=[];stage='four_shapes'
        for index,case in enumerate(supply['probe_cases']):
            row=train[case['index']];n=case['pad_to'] or len(row['input_ids']);assert len(row['input_ids'])<=n
            ids=torch.tensor([row['input_ids']+[0]*(n-len(row['input_ids']))],device='cuda')
            mask=torch.tensor([[1]*len(row['input_ids'])+[0]*(n-len(row['input_ids']))],device='cuda')
            labels=torch.tensor([row['labels']],device='cuda');optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss
            finite_loss=bool(torch.isfinite(loss));put(out/f'{index}-forward.json',dict(case=case,loss=float(loss.detach()) if finite_loss else None,finite=finite_loss))
            assert finite_loss;loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            finite=all(bool(torch.isfinite(g).all()) for g in grads)
            put(out/f'{index}-gradient.json',dict(tensors=len(grads),all_finite=finite));assert finite
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            torch.cuda.synchronize();fb=time.monotonic()-tick;adam_seconds=None
            if index<2:
                before=model.shared.weight.detach().clone();tick=time.monotonic();optimizer.step();torch.cuda.synchronize();adam_seconds=time.monotonic()-tick
                assert bool((before!=model.shared.weight).any());del before
                assert all(bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
            records.append(dict(case=case,padded_encoder=n,decoder=len(row['labels']),FB_seconds=fb,Adam_seconds=adam_seconds,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
            put(out/f'{index}-complete.json',records[-1]);del ids,mask,labels,loss
        stage='tiny_train_decode';optimizer.zero_grad(set_to_none=True);model.gradient_checkpointing_disable();model.eval()
        # Isolate all decoder defaults. This is engineering output, not patient quality.
        config=GenerationConfig(pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,do_sample=False,num_beams=1,max_new_tokens=16)
        model.generation_config=GenerationConfig(pad_token_id=0,eos_token_id=1,decoder_start_token_id=0)
        row=train[supply['probe_cases'][0]['index']];ids=torch.tensor([row['input_ids']],device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):generated=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=config)
        put(out/'tiny-train-raw.json',dict(row_id=row['row_id'],generated_ids=generated.tolist(),quality_not_computed=True,max_new_tokens=16))
        put(out/'complete.json',dict(status='COMPLETE_FOUR_SHAPE_TWO_ADAM_SEQ2SEQ_ENGINEERING',records=records,
            seconds=time.monotonic()-start,parameters=247587456,source=ref(__file__),plan=ref(a.plan),supply=ref(a.supply),
            torch=torch.__version__,transformers=transformers.__version__,probe_state_discarded=True,
            checkpoint_saved=False,formal_training=False,held_model_calls=0,quality_calls=0))
    except BaseException:
        put(out/'failure.json',dict(stage=stage,seconds=time.monotonic()-start,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','supply','model-dir','output','gpu-uuid','authorized-probe-only'):p.add_argument('--'+n,required=True)
    p.add_argument('--admit-until',type=float,required=True);main(p.parse_args())
