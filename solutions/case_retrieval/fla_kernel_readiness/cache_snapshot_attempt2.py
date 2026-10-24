"""Instrumentation-only cache snapshot repair; no kernel/model/threshold changes."""
import hashlib,json
from collections.abc import Mapping

class StateSnapshot(dict):
    def __init__(self):
        super().__init__(); self.layout=[]

def capture(cache,torch):
    result=StateSnapshot()
    def visit(value,path):
        if value is None:
            result.layout.append((path,'none'));return
        if torch.is_tensor(value):
            result.layout.append((path,'tensor',tuple(value.shape),str(value.dtype)))
            if path in result:raise ValueError('Duplicate state path: '+path)
            result[path]=value.detach().cpu().clone();return
        if isinstance(value,Mapping):
            keys=list(value)
            if any(type(k) not in (int,str) for k in keys):raise TypeError('Unsupported state mapping key: '+path)
            keys.sort(key=lambda k:(type(k).__name__,str(k)))
            result.layout.append((path,'mapping',tuple((type(k).__name__,k) for k in keys)))
            for k in keys:visit(value[k],path+'/'+type(k).__name__+':'+str(k))
            return
        if isinstance(value,(list,tuple)):
            result.layout.append((path,type(value).__name__,len(value)))
            for i,v in enumerate(value):visit(v,path+'/index:'+str(i))
            return
        raise TypeError('Unsupported cache value '+type(value).__name__+' at '+path)
    if not isinstance(cache.layers,(list,tuple)):raise TypeError('Unsupported cache.layers container')
    result.layout.append(('layers','count',len(cache.layers)))
    for i,layer in enumerate(cache.layers):
        result.layout.append(('layer',i,type(layer).__name__))
        for name in ['conv_states','recurrent_states']:
            path=f'{i}/{name}'
            if hasattr(layer,name):visit(getattr(layer,name),path)
            else:result.layout.append((path,'absent'))
    if not result:raise ValueError('No initialized conv/recurrent tensors; cannot claim state equivalence')
    return result

def compare(a,b,torch,thresholds,original_compare):
    if not isinstance(a,StateSnapshot) or not isinstance(b,StateSnapshot):raise TypeError('StateSnapshot required')
    if a.layout!=b.layout:raise AssertionError('Cache structure/layer/key/shape/dtype differs')
    metrics=original_compare(a,b,torch,thresholds)
    return {'structure_sha256':hashlib.sha256(json.dumps(a.layout,sort_keys=True).encode()).hexdigest(),'state_tensor_count':len(a),'tensor_metrics':metrics}
