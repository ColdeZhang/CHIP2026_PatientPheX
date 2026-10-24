"""Actual CPU autograd contracts using the production full-document model."""
from types import SimpleNamespace
from pathlib import Path
import argparse,copy,json,torch
from torch import nn
from model import PatientSpanModel
from common import save,ref

class ToyEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.config=SimpleNamespace(hidden_size=8);self.embedding=nn.Embedding(30,8);self.linear=nn.Linear(8,8)
 def forward(self,input_ids,attention_mask,return_dict=True):return SimpleNamespace(last_hidden_state=self.linear(self.embedding(input_ids)))

def run():
 torch.set_num_threads(1);torch.manual_seed(20261024)
 row={'pmc_id':'synthetic','token_length':8,'windows':[{'start':0,'end':4,'ids':[1,3,4,5,6,2]},{'start':4,'end':8,'ids':[1,7,8,9,10,2]}],'patients':[{'patient_id':'P1','spans':[[6,7]]},{'patient_id':'P2','spans':[[7,8]]}],'candidates':[[0,0],[1,2],[2,2],[4,5]],'positive_candidates':[1],'bags':[{'patient_index':0,'instances':[1,3],'label':1},{'patient_index':1,'instances':[1,3],'label':1},{'patient_index':0,'instances':[0],'label':0},{'patient_index':1,'instances':[0],'label':0}]}
 model=PatientSpanModel(ToyEncoder(),head_size=4,encoder_batch=1,score_chunk=2);model.train();answer=model.loss(row,retain_hidden=True);answer['bag_loss'].backward()
 assert torch.isfinite(answer['loss']) and torch.isfinite(answer['bag_loss']);assert answer['hidden'].grad[6].abs().sum()>0 and answer['hidden'].grad[7].abs().sum()>0
 assert model.encoder.embedding.weight.grad[9].abs().sum()>0 and model.encoder.embedding.weight.grad[10].abs().sum()>0
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 # Max selects a latent supported instance; it does not label every bag span.
 v=torch.tensor([[-2.,3.,1.],[4.,-3.,2.]],requires_grad=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(v.max(1).values,torch.tensor([1.,0.]));loss.backward();assert torch.count_nonzero(v.grad)==2 and v.grad[0,1]<0 and v.grad[1,0]>0
 empty=copy.deepcopy(row)
 for b in empty['bags']:b['label']=0
 model.zero_grad();a=model.loss(empty);a['loss'].backward();assert torch.isfinite(a['loss']);assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.condition_head.parameters())
 # Duplicated overlap windows average to identical global representation when
 # the encoder itself has no positional/context effect; anchor occurrence remains once.
 model.eval();h,p=model.encode_document(row);dup=copy.deepcopy(row);dup['windows'].append(copy.deepcopy(row['windows'][1]));h2,p2=model.encode_document(dup);assert torch.allclose(h,h2,atol=1e-7) and torch.allclose(p,p2,atol=1e-7)
 # Target gradients really reach both encoder and patient-conditioned parameters.
 model.train();model.zero_grad();before={n:p.detach().clone() for n,p in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),lr=1e-3);a=model.loss(row);a['loss'].backward();opt.step();changed={n for n,p in model.named_parameters() if not torch.equal(before[n],p)}
 assert any(n.startswith('encoder.') for n in changed) and any(n.startswith('condition_head.') for n in changed) and any(n.startswith('patient_head.') for n in changed)
 return {'status':'PASS_PRODUCTION_CPU_FULL_DOCUMENT_GRADIENT_CONTRACTS','cross_window_anchor_gradients':True,'both_patients_can_be_positive_same_concept':True,'whole_document_max_bag':True,'true_empty_patient_negative_bags':True,'overlap_average_no_anchor_duplication':True,'finite_loss_and_gradients':True,'actual_Adam_updates':sorted(changed),'toy_only_not_GPU_cost_or_quality':True,'torch':torch.__version__,'sources':[ref(Path(__file__)),ref(Path(__file__).with_name('model.py'))]}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();r=run();save(a.output,r);print(json.dumps(r))
