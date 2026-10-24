"""CPU contract against actual TF5.20.2 DynamicCache and supported mixed containers."""
import copy,json,os,pathlib,types
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from cache_snapshot_attempt2 import capture,compare
import gpu_probe as frozen
import qwen35_smoke as H
cfg=Qwen3_5TextConfig.from_pretrained('/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/models/Qwen3.5-9B',local_files_only=True)
cache=DynamicCache(config=cfg)
for i in [0,1]:
    cache.layers[i].conv_states[0]=torch.full((1,8192,4),i+1,dtype=torch.bfloat16)
    cache.layers[i].recurrent_states[0]=torch.full((1,32,128,128),i+1,dtype=torch.float32)
a=capture(cache,torch);assert len(a)==4 and len(cache.layers)==32
assert set(a)=={'0/conv_states/int:0','0/recurrent_states/int:0','1/conv_states/int:0','1/recurrent_states/int:0'}
b=capture(copy.deepcopy(cache),torch);th=json.load(open('probe-plan.json'))['thresholds'];result=compare(a,b,torch,th,frozen.compare_states)
assert len([x for x in a.layout if x[0]=='layer'])==32
assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
cache.layers[0].conv_states[0].zero_();assert a['0/conv_states/int:0'].count_nonzero()==8192*4
mixed=types.SimpleNamespace(layers=[types.SimpleNamespace(conv_states={4:torch.ones(2),0:None},recurrent_states=[torch.ones(3),None]),types.SimpleNamespace(conv_states=torch.ones(4),recurrent_states=(None,torch.ones(5)))])
m=capture(mixed,torch);assert len(m)==4 and any(x[1]=='none' for x in m.layout if len(x)>1)
reordered=copy.deepcopy(mixed);reordered.layers[0].conv_states={0:None,4:torch.ones(2)};assert m.layout==capture(reordered,torch).layout
rejected={}
for label,change in [('unknown_scalar',lambda x:setattr(x.layers[0],'conv_states',{0:123})),('missing_key',lambda x:x.layers[0].conv_states.pop(0)),('dtype_change',lambda x:x.layers[0].conv_states.update({4:torch.ones(2,dtype=torch.float64)})),('missing_layer',lambda x:x.layers.pop())]:
    value=copy.deepcopy(mixed);change(value)
    try:other=capture(value,torch);compare(m,other,torch,th,frozen.compare_states)
    except (TypeError,ValueError,AssertionError):rejected[label]=True
    else:raise AssertionError(label+' was not rejected')
try:capture(types.SimpleNamespace(layers=[types.SimpleNamespace(conv_states={0:None},recurrent_states={0:None})]),torch)
except ValueError:rejected['empty_no_false_pass']=True
else:raise AssertionError('Empty snapshot incorrectly accepted')
assert not torch.cuda.is_initialized()
H.write('attempt2-cache-cpu-contract.json',{'passed':True,'actual_dynamic_cache_layers':32,'actual_initialized_tensors':4,'actual_field_container':'dict[int,Tensor|None]','mixed_dict_list_tuple_tensor_none':True,'mapping_order_stable':True,'snapshot_detached_copy':True,'all_layer_keys_shape_dtype_preserved':True,'fail_closed_cases':rejected,'cuda_initialized':False,'structure_sha256':result['structure_sha256']})
print(pathlib.Path('attempt2-cache-cpu-contract.json').read_text())
