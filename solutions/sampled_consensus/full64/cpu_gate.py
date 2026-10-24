"""Read-only original input/runtime/assets + real CPU sampling-interface contracts."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import probe_runner as probe
from sampling import SEEDS,GENERATION,row_seed
save=probe.save;legacy=probe.legacy

def stat_ref(p):
    p=Path(p);s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    for name,r in plan['input_refs'].items():assert Path(r['remote_path']).stat().st_size==r['bytes'] and legacy.digest(r['remote_path'])==r['sha256'],name
    rows=legacy.read_prompts(plan['input_refs']['prompts']['remote_path']);audit=json.loads(Path(plan['input_refs']['audit']['remote_path']).read_text())
    assert len(rows)==len(audit)==64;assert sum(len(r['code_map']) for r in rows)==466
    seeds=[]
    for i,(r,t) in enumerate(zip(rows,audit)):
        assert t['index']==i and t['row_id']==r['row_id'] and t['expanded_row_sha256']==probe.jsha(r)
        assert t['expanded']['tokens']<=14400
        for seed in SEEDS:seeds.append({'seed':seed,'row_id':r['row_id'],'row_seed':row_seed(seed,r['row_id'])})
    assert len({s['row_seed'] for s in seeds})==128
    assert sum(t['expanded']['tokens'] for t in audit)==512091 and max(t['expanded']['tokens'] for t in audit)==13230
    prior_path=plan['prior_base_SHA_certificate']['path'];assert legacy.digest(prior_path)==plan['prior_base_SHA_certificate']['sha256']
    prior=json.loads(Path(prior_path).read_text());base=prior['base_files']
    for name in base:
        path=str(Path(plan['model'])/name);assert stat_ref(path)==prior['asset_stats'][path],path
    # The same immutable base was fully SHA-read in the completed F0 CPU gate;
    # exact inode/mtime/size identity is rechecked, new full64 adapter SHA below.
    adapter={}
    for name,digest in plan['adapter_files'].items():
        p=Path(plan['adapter'])/name;assert legacy.digest(p)==digest;adapter[name]={'bytes':p.stat().st_size,'sha256':digest}
    # Real runtime imports and logit processor construction, no model or CUDA load.
    import torch,transformers,peft,tokenizers
    from transformers import GenerationConfig,LogitsProcessorList,Qwen3_5TextConfig
    from transformers.generation.utils import GenerationMixin
    assert not torch.cuda.is_initialized()
    actual={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'tokenizers':tokenizers.__version__}
    assert actual==plan['versions']
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.generation import utils as generation_utils
    runtime_sha={"qwen_modeling":legacy.digest(modeling_qwen3_5.__file__),"generation_utils":legacy.digest(generation_utils.__file__)}
    assert runtime_sha==plan['runtime_sha256']
    backend={'fla_installed':importlib.util.find_spec('fla') is not None,'causal_conv1d_installed':importlib.util.find_spec('causal_conv1d') is not None}
    assert not any(backend.values()),'Original native G requires no FLA or causal_conv1d backend'
    class Interface(GenerationMixin):
        config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True)
        device=torch.device('cpu')
    d=Interface();config=GenerationConfig(**GENERATION,eos_token_id=plan['eos_ids'],pad_token_id=plan['pad_token_id']);config.validate()
    d._prepare_special_tokens(config,device=torch.device('cpu'))
    processors=d._get_logits_processor(generation_config=config,input_ids_seq_length=7,encoder_input_ids=None,prefix_allowed_tokens_fn=None,logits_processor=LogitsProcessorList(),device=torch.device('cpu'),model_kwargs={})
    names=[type(p).__name__ for p in processors];assert names==['TemperatureLogitsWarper','TopKLogitsWarper','TopPLogitsWarper','MinPLogitsWarper'],names
    # Same row seed gives same actual CPU multinomial draws regardless of order.
    logits=torch.arange(32,dtype=torch.float32)[None,:]/8;input_ids=torch.tensor([[2,3,4,5,6,7,8]])
    processed=processors(input_ids,logits.clone());probs=torch.softmax(processed,dim=-1);assert torch.isfinite(probs).all() and torch.count_nonzero(probs)<=20
    samples={}
    for seed in SEEDS:
        rs=row_seed(seed,rows[0]['row_id']);torch.manual_seed(rs);one=torch.multinomial(probs,32,replacement=True).tolist();torch.manual_seed(rs);assert one==torch.multinomial(probs,32,replacement=True).tolist();samples[str(seed)]=one
    assets={str(Path(plan['model'])/n):stat_ref(Path(plan['model'])/n) for n in base};assets.update({str(Path(plan['adapter'])/n):stat_ref(Path(plan['adapter'])/n) for n in adapter})
    tokenizer_json=json.loads((Path(plan['model'])/'tokenizer.json').read_text());think={t['content']:t['id'] for t in tokenizer_json['added_tokens'] if t['content'] in ['<think>','</think>']};assert len(think)==2
    assert not torch.cuda.is_initialized()
    result={'status':'PASS_NATIVE_ASSETS_ORIGINAL64_CPU_SAMPLING_INTERFACE','plan_sha256':legacy.digest(a.plan),'input_tokens':512091,'max_input_tokens':13230,'rows':64,'no_G_endpoint_or_cache':True,'no_new_tokenization':True,'no_GPU_model_fit_or_quality':True,'base_files':base,'reused_base_SHA_certificate':plan['prior_base_SHA_certificate'],'base_identity_stats_unchanged':True,'adapter_files':adapter,'asset_stats':assets,'versions':actual,'runtime_sha256':runtime_sha,'backend':backend,'generation_kwargs':GENERATION,'presence_policy':'zero: no added presence processor','processors':names,'generation_interface_signature':str(inspect.signature(d._get_logits_processor)),'cpu_seed_draws':samples,'row_seeds':seeds,'think_token_ids':think,'elapsed_seconds':time.monotonic()-started}
    save(out/'complete.json',result);print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'think_token_ids':think,'processors':names}),flush=True)
if __name__=='__main__':main()
