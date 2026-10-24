"""Discriminating synthetic algebra/schema contract; no real weight payload."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import time
import numpy as np
from delta_mean import validate_configs, concatenate_pair, compose_state


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config1', type=Path, required=True)
    parser.add_argument('--config2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    c1, c2 = json.loads(args.config1.read_text()), json.loads(args.config2.read_text())
    checks=[]
    def check(label, value):
        if not value: raise AssertionError(label)
        checks.append(label)
    def reject(label, operation):
        try: operation()
        except ValueError: checks.append(label); return
        raise AssertionError('Invalid fixture accepted: ' + label)
    new = validate_configs(c1, c2)
    check('actualconfig1and2 identical bytes',args.config1.read_bytes()==args.config2.read_bytes())
    check('new rank32/alpha32 scaling1 only',new['r']==new['lora_alpha']==32 and
          {k:v for k,v in new.items() if k not in ['r','lora_alpha']}==
          {k:v for k,v in c1.items() if k not in ['r','lora_alpha']})
    rng=np.random.default_rng(20261024)
    a1,a2=(rng.normal(size=(16,7)) for _ in range(2))
    b1,b2=(rng.normal(size=(5,16)) for _ in range(2))
    a,b=concatenate_pair(a1,b1,a2,b2)
    expected=(2*b1@a1+2*b2@a2)/2
    actual=b@a; error=float(np.max(np.abs(actual-expected)))
    check('effective delta mean FP64',np.allclose(actual,expected,rtol=1e-13,atol=1e-13))
    x=rng.normal(size=(3,7)); base=rng.normal(size=(5,7))
    check('single linear layer same base plus mean delta',np.allclose(x@(base+actual).T,
        (x@(base+2*b1@a1).T+x@(base+2*b2@a2).T)/2,rtol=1e-13,atol=1e-13))
    naive=2*((b1+b2)/2)@((a1+a2)/2)
    check('naive factor average rejected as wrong delta',not np.allclose(naive,expected))
    samea,sameb=concatenate_pair(a1,b1,a1,b1)
    check('identical checkpoints recover same effective delta',np.allclose(sameb@samea,2*b1@a1))
    zeroa,zerob=concatenate_pair(a1,b1,a2,np.zeros_like(b2))
    check('zero second delta halves first',np.allclose(zerob@zeroa,b1@a1))
    # LoRA gauge invariance: invertible changes of factor basis leave delta fixed.
    gauge=np.diag(np.linspace(.5,2,16)); inverse=np.diag(1/np.diag(gauge))
    ga,gb=concatenate_pair(gauge@a1,b1@inverse,a2,b2)
    check('effective mean invariant to factor gauge',np.allclose(gb@ga,expected,rtol=1e-13,atol=1e-13))
    af,bf=concatenate_pair(a1.astype('float32'),b1.astype('float32'),a2.astype('float32'),b2.astype('float32'))
    check('FP32 factors concat bitexact',np.array_equal(af[:16],a1.astype('float32')) and np.array_equal(bf[:,16:],b2.astype('float32')))
    float_error=float(np.max(np.abs(bf@af-expected)))
    check('FP32 numeric tolerance not bitwise guarantee',np.allclose(bf@af,expected,rtol=2e-5,atol=2e-5))
    for key,value in [('use_dora',True),('use_rslora',True),('r',8),('lora_alpha',16),
                      ('rank_pattern',{'q_proj':8}),('alpha_pattern',{'q_proj':16}),
                      ('modules_to_save',['lm_head']),('bias','all'),('lora_bias',True),
                      ('use_qalora',True),('fan_in_fan_out',True),('target_parameters',['w']),
                      ('alora_invocation_tokens',[1]),('kasa_config',{}),('layer_replication',[[0,1]])]:
        bad=deepcopy(c1);bad[key]=value
        reject('reject variant '+key,lambda bad=bad:validate_configs(bad,bad))
    bad=deepcopy(c2);bad['revision']='different'
    reject('reject differing config/base/revision',lambda:validate_configs(c1,bad))
    prefix='base_model.model.model.layers.0.self_attn.q_proj'; keya=prefix+'.lora_A.weight';keyb=prefix+'.lora_B.weight'
    s1={keya:a1,keyb:b1};s2={keya:a2,keyb:b2}
    output,conf=compose_state(s1,s2,c1,c2,[prefix])
    check('savedPEFT-like factor keys preserved',set(output)==set(s1) and output[keya].shape==(32,7) and output[keyb].shape==(5,32))
    reject('reject missing factor',lambda:compose_state({keya:a1},s2,c1,c2,[prefix]))
    reject('reject extra base weight',lambda:compose_state({**s1,'base.weight':base},s2,c1,c2,[prefix]))
    reject('reject rank mismatch',lambda:concatenate_pair(a1[:8],b1,a2,b2))
    nan=a1.copy();nan[0,0]=np.nan
    reject('reject nonfinite',lambda:concatenate_pair(nan,b1,a2,b2))
    # Saving/reopening the synthetic arrays tests shape/key preservation, not PEFT loading.
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez(args.output/'synthetic-r32.npz',**output)
    with np.load(args.output/'synthetic-r32.npz',allow_pickle=False) as restored:
        check('synthetic save/reopen all factor values exact',set(restored.files)==set(output) and all(np.array_equal(restored[k],output[k]) for k in output))
    result=dict(status='PASS_NUMPY_EFFECTIVE_DELTA_AND_SCHEMA_SYNTHETIC_NOT_REAL_PEFT_LOAD',
        checks=checks,checks_count=len(checks),seconds=time.monotonic()-start,
        numpy_version=np.__version__,fp64_max_absolute_error=error,fp32_max_absolute_error=float_error,
        naive_factor_average_max_absolute_error=float(np.max(np.abs(naive-expected))),
        actual_configs=[ref(args.config1),ref(args.config2)],source=ref(__file__),helper=ref(Path(__file__).with_name('delta_mean.py')),
        actual_checkpoint_tensors_loaded=False,actual_PeftModel_loaded=False,
        Torch_CUDA_model_gold_quality_calls=0,actual_GPU_bitwise_equivalence_claimed=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks_count','seconds','fp64_max_absolute_error','fp32_max_absolute_error']}))


if __name__=='__main__':main()
