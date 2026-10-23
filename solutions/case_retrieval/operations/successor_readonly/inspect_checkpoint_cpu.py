import os,json,hashlib,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
root=Path('relation-training/formal2ep/epoch-002')
p=root/'training_state.pt';start=time.time()
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()=='d2dcc78070f53d36984fea99198ea721edad6e8918479a7600318e9fbf51344e'
s=torch.load(p,map_location='cpu',weights_only=False)
opt=s['optimizer'];states=list(opt['state'].values());steps=sorted({float(x['step']) for x in states})
assert steps==[52.] and s['completed_epochs']==2 and s['global_step']==52
assert all(x['exp_avg'].shape==x['exp_avg_sq'].shape for x in states)
assert not torch.cuda.is_initialized()
print(json.dumps({'scope':'CPU read-only state inspection, no model load or training','path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'state_keys':sorted(s),'completed_epochs':s['completed_epochs'],'global_step':s['global_step'],'optimizer_state_entries':len(states),'optimizer_steps':steps,'optimizer_learning_rates':[g['lr'] for g in opt['param_groups']],'optimizer_initial_lrs':[g.get('initial_lr') for g in opt['param_groups']],'adam_moment_elements':sum(x['exp_avg'].numel() for x in states),'moments_finite':all(bool(torch.isfinite(x[k]).all()) for x in states for k in ['exp_avg','exp_avg_sq']),'scheduler':s['scheduler'],'rng_fields_present':all(k in s for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']),'cuda_rng_states':len(s['cuda_rng']),'cuda_initialized':torch.cuda.is_initialized(),'torch_version':torch.__version__,'elapsed_seconds':time.time()-start},default=str))
