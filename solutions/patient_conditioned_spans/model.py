"""Shared complete-document encoder and patient-conditioned span scores.

Encoder checkpoints preserve RNG and keep every document window in the graph.
Patient anchors are gathered after overlap averaging; never detached. Relation
max-MIL is across the entire concept bag, not a per-window approximation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from vendor.gp import multilabel_loss


def rotate_positions(x,positions):
 d=x.shape[-1];assert d%2==0
 inv=10000.0**(-torch.arange(d//2,device=x.device,dtype=torch.float32)/(d//2))
 phase=positions.float()[:,None]*inv[None];c,s=phase.cos(),phase.sin()
 a,b=x[...,0::2],x[...,1::2]
 return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)


class PatientSpanModel(nn.Module):
 def __init__(self,encoder,head_size=64,encoder_batch=4,score_chunk=8192):
  super().__init__();self.encoder=encoder;h=encoder.config.hidden_size;self.ner_head=nn.Linear(h,2*head_size);self.condition_head=nn.Linear(h,2*head_size);self.patient_head=nn.Linear(h,2*head_size);self.patient_bias=nn.Linear(h,1);self.head_size=head_size;self.encoder_batch=encoder_batch;self.score_chunk=score_chunk
  for layer in [self.ner_head,self.condition_head,self.patient_head,self.patient_bias]:nn.init.normal_(layer.weight,std=.02);nn.init.zeros_(layer.bias)
 def _encode(self,ids,mask):return self.encoder(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state
 def encode_document(self,row):
  dev=next(self.parameters()).device;values=[];indices=[];ws=row['windows']
  for begin in range(0,len(ws),self.encoder_batch):
   batch=ws[begin:begin+self.encoder_batch];n=max(len(w['ids']) for w in batch);ids=torch.zeros((len(batch),n),dtype=torch.long,device=dev);mask=torch.zeros_like(ids)
   for i,w in enumerate(batch):ids[i,:len(w['ids'])]=torch.tensor(w['ids'],device=dev);mask[i,:len(w['ids'])]=1
   hidden=checkpoint(self._encode,ids,mask,use_reentrant=False,preserve_rng_state=True) if self.training else self._encode(ids,mask)
   for i,w in enumerate(batch):values.append(hidden[i,1:len(w['ids'])-1].float());indices.append(torch.arange(w['start'],w['end'],device=dev))
  index=torch.cat(indices);value=torch.cat(values);full=value.new_zeros((row['token_length'],value.shape[-1])).index_add(0,index,value);count=value.new_zeros(row['token_length']).index_add(0,index,value.new_ones(index.shape));assert bool((count>0).all());full=full/count[:,None]
  patients=torch.stack([torch.stack([full[a:b].mean(0) for a,b in p['spans']]).mean(0) for p in row['patients']]);return full,patients
 def _ner_scores(self,q,k,pairs):return (q[pairs[:,0]]*k[pairs[:,1]]).sum(-1)/math.sqrt(self.head_size)
 def global_scores(self,hidden,pairs):
  q,k=self.ner_head(hidden).chunk(2,-1);pos=torch.arange(len(hidden),device=hidden.device);q=rotate_positions(q,pos);k=rotate_positions(k,pos);parts=[]
  for indices in pairs.split(self.score_chunk):
   parts.append(checkpoint(self._ner_scores,q,k,indices,use_reentrant=False,preserve_rng_state=True) if self.training else self._ner_scores(q,k,indices))
  return torch.cat(parts)
 def patient_scores(self,hidden,patients,pairs):
  q,k=self.condition_head(hidden).chunk(2,-1);pq,pk=self.patient_head(patients).chunk(2,-1);i,j=pairs[:,0],pairs[:,1]
  a=rotate_positions(q[i][None]+pq[:,None],i);b=rotate_positions(k[j][None]+pk[:,None],j)
  return (a*b).sum(-1)/math.sqrt(self.head_size)+self.patient_bias(patients)
 def loss(self,row,retain_hidden=False):
  hidden,patients=self.encode_document(row)
  if retain_hidden:hidden.retain_grad();patients.retain_grad()
  # Explicit FP32 heads/targets/loss, independent of encoder BF16 autocast.
  with torch.autocast(device_type=hidden.device.type,enabled=False):
   pairs=torch.tensor(row['candidates'],device=hidden.device,dtype=torch.long);ner=self.global_scores(hidden.float(),pairs);positive=torch.zeros(len(pairs),device=hidden.device,dtype=torch.bool);positive[row['positive_candidates']]=True
   negative=~positive
   if row.get('ignored_NER_candidates'):negative[row['ignored_NER_candidates']]=False
   valid=positive|negative
   # One batch row is the unique physical candidate vector for the entire doc.
   nerloss=multilabel_loss(ner[None],positive[None],valid[None])
   used=sorted({i for b in row['bags'] for i in b['instances']});lookup={i:j for j,i in enumerate(used)}
   if used:
    relation=self.patient_scores(hidden.float(),patients.float(),pairs[used]);baglogits=torch.stack([relation[b['patient_index'],[lookup[i] for i in b['instances']]].max() for b in row['bags']]);labels=torch.tensor([b['label'] for b in row['bags']],device=hidden.device,dtype=torch.float32);reloss=F.binary_cross_entropy_with_logits(baglogits,labels)
   else:relation=None;baglogits=None;reloss=hidden.sum()*0
   total=nerloss+reloss
  info={'loss':total,'NER_loss':nerloss,'bag_loss':reloss,'bags':len(row['bags']),'NER_positive':int(positive.sum()),'candidate_spans':len(pairs),'patient_span_scores':relation,'bag_logits':baglogits}
  if retain_hidden:info.update({'hidden':hidden,'patients':patients})
  return info
 def infer(self,row,floor=-2.):
  hidden,patients=self.encode_document(row);dev=hidden.device;pairs=torch.tensor(row['candidates'],device=dev,dtype=torch.long);records=[]
  with torch.autocast(device_type=dev.type,enabled=False):
   for offset in range(0,len(pairs),self.score_chunk):
    pp=pairs[offset:offset+self.score_chunk];scores=self.global_scores(hidden.float(),pp);ids=torch.nonzero(scores>=floor).flatten()
    if not len(ids):continue
    selected=pp[ids];relations=self.patient_scores(hidden.float(),patients.float(),selected).T.cpu().tolist();ns=scores[ids].cpu().tolist()
    for local,nscore,rel in zip(ids.cpu().tolist(),ns,relations):records.append({'candidate_index':offset+local,'NER_logit':nscore,'patient_logits':rel})
  return records
