"""GPU-visible import-only gate, prepared but never launched by CPU readiness."""
import argparse,json,os,pathlib,subprocess,time
from cpu_contract import graph,sha
import qwen35_smoke as H
p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--run-authorized',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
assert a.run_authorized and os.getenv('CUDA_VISIBLE_DEVICES')==a.gpu_uuid,'explicit allocation required'
r={'scope':'GPU-visible imports and dispatch introspection only, no forward/backward/JIT benchmark','started':time.time(),'passed':False}
try:
    procs=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    assert not any(x.split(',')[0].strip()==a.gpu_uuid for x in procs.splitlines()),'allocated GPU occupied'
    import torch
    r['cuda_initialized_before']=torch.cuda.is_initialized()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    r['bindings']={n:graph(getattr(q,n)) for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update']}
    for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:assert any(x['module'].startswith('fla.ops.gated_delta_rule') for x in r['bindings'][n]),n
    for n in ['causal_conv1d_fn','causal_conv1d_update']:assert not any(x['module'].startswith('causal_conv1d') for x in r['bindings'][n]),n
    r['modeling_sha256']=sha(q.__file__);assert r['modeling_sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
    r['cuda_initialized_after']=torch.cuda.is_initialized();r['passed']=True
except Exception as e:
    import traceback;r['error']=str(e);r['traceback']=traceback.format_exc()
r['elapsed_seconds']=time.time()-r['started'];H.write(a.output,r);print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
