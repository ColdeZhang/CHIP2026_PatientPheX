"""Actual5.20.2 loading-info types on a tiny random CPU checkpoint,0forward."""
import argparse,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import atomic,ref,json_default

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
    import torch,transformers
    from transformers import GPT2Config,GPT2LMHeadModel
    torch.set_num_threads(1);torch.manual_seed(1)
    assert transformers.__version__=='5.20.2'
    model=GPT2LMHeadModel(GPT2Config(vocab_size=17,n_positions=16,n_embd=8,n_layer=1,n_head=2,bos_token_id=1,eos_token_id=2))
    model.save_pretrained(out/'tiny-random');del model
    model,loading=GPT2LMHeadModel.from_pretrained(out/'tiny-random',local_files_only=True,dtype=torch.float32,output_loading_info=True)
    observed_types={k:type(v).__name__ for k,v in loading.items()}
    assert any(isinstance(v,(set,frozenset)) for v in loading.values()),observed_types
    atomic(out/'actual-loading.json',loading);assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    parsed=json.loads((out/'actual-loading.json').read_text());assert set(parsed)==set(loading)
    for k,v in loading.items():assert parsed[k]==(sorted(v,key=repr) if isinstance(v,(set,frozenset)) else v)
    bad=dict(missing_keys={'z.key','a.key'},unexpected_keys=set(),mismatched_keys=set(),error_msgs=[])
    atomic(out/'nonempty-fields.json',bad);assert json.loads((out/'nonempty-fields.json').read_text())['missing_keys']==['a.key','z.key']
    assert not all(not bad.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    unknown_rejected=False;nan_rejected=False
    try:atomic(out/'unknown.json',dict(value=object()))
    except TypeError:unknown_rejected=True
    try:atomic(out/'nan.json',dict(value=float('nan')))
    except ValueError:nan_rejected=True
    assert unknown_rejected and nan_rejected and not (out/'unknown.json').exists() and not (out/'nan.json').exists()
    assert not torch.cuda.is_initialized()
    atomic(out/'result.json',dict(status='PASS_ACTUAL_LOADING_METADATA_ONLY',source=ref(__file__),common=ref(Path(__file__).parent/'common.py'),runtime=dict(torch=torch.__version__,transformers=transformers.__version__),
        actual_loading_types=observed_types,actual_loading=ref(out/'actual-loading.json'),all_original_fields_preserved=True,original_missing_mismatch_gate_retained=True,
        nonempty_fields_still_rejected=True,unknown_types_rejected=True,nonfinite_JSON_rejected=True,forward_calls=0,CUDA_initialized=False,
        random_tiny_parameters=sum(p.numel() for p in model.parameters()),pretrained_weights_downloaded=False,real9B_model_not_loaded=True))
if __name__=='__main__':main()
