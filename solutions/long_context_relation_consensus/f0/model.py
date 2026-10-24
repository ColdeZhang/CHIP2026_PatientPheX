"""Native full-context entity pooling and shared binary block-bilinear head.

The binary target and pooling are established old DocRE mechanisms; the new
signal is token interaction inside a long-context encoder. No attention matrices.
"""
import torch
from torch import nn

class FullContextRelations(nn.Module):
 def __init__(self,encoder):
  super().__init__();self.encoder=encoder
  h=encoder.config.hidden_size
  self.head=nn.Linear(h,256);self.tail=nn.Linear(h,256)
  self.classifier=nn.Linear(8192,1);self.dropout=nn.Dropout(.1)
 def entity_states(self,doc):
  device=self.head.weight.device;physical=[[] for e in doc['entities']]
  for wi,window in enumerate(doc['windows']):
   ids=torch.tensor([window['ids']],device=device,dtype=torch.long)
   hidden=self.encoder(input_ids=ids,attention_mask=torch.ones_like(ids),output_hidden_states=False,output_attentions=False).last_hidden_state[0]
   lo,hi=window['body_range']
   for ei,e in enumerate(doc['entities']):
    for i,j,owner,pi in e['spans']:
     if owner!=wi:continue
     assert lo<=i<j<=hi
     physical[ei].append(hidden[i-lo+1:j-lo+1].float().mean(0))
  assert all(len(values)==len(e['spans']) for values,e in zip(physical,doc['entities']))
  return torch.stack([torch.logsumexp(torch.stack(values),dim=0) for values in physical])
 def forward(self,doc):
  states=self.entity_states(doc);pairs=torch.tensor(doc['pairs'],device=states.device,dtype=torch.long)
  with torch.autocast(device_type=states.device.type,enabled=False):
   h=torch.tanh(self.head(states[pairs[:,0]].float())).reshape(-1,8,32)
   t=torch.tanh(self.tail(states[pairs[:,1]].float())).reshape(-1,8,32)
   blocks=(h.unsqueeze(-1)*t.unsqueeze(-2)).flatten(1)
   return self.classifier(self.dropout(blocks)).squeeze(-1)

def load_model(path,loading_path):
 from transformers import AutoModel
 from common import save
 enc,info=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,
      attn_implementation='sdpa',reference_compile=False,output_loading_info=True)
 save(loading_path,info)
 assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
 assert all(k.startswith('head.') or k.startswith('decoder.') for k in info.get('unexpected_keys',[]))
 assert sum(p.numel() for p in enc.parameters())==149014272
 enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model=FullContextRelations(enc);assert sum(p.numel() for p in model.parameters())==149416193
 return model.cuda()
