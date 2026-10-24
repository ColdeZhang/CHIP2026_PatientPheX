"""CPU-only numerical/production delta contracts; no weights or tokenization."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse,copy,hashlib,json,torch
from torch import nn
from model import PatientSpanModel,GeometryModel,grouped_patient_loss
from common import ref,save,unpack

class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8);self.embedding=nn.Embedding(40,8);self.linear=nn.Linear(8,8);self.dropout=nn.Dropout(.1)
    def forward(self,input_ids,attention_mask,return_dict=True):
        return SimpleNamespace(last_hidden_state=self.dropout(self.linear(self.embedding(input_ids))))

def group(labels,concepts=None,masked=None):
    return {'patients':[{'patient_id':str(i)} for i in range(len(labels))], 'bags':[{'patient_index':i,'concept':['hpo',concepts[i] if concepts else 'HP:0000001'],'instances':[0],'label':y} for i,y in enumerate(labels)],'masked_bags':masked or []}

def run(encoded):
    torch.set_num_threads(1);checks={}
    # Single patient/group equals BCE, including extreme logits and gradients.
    for value in [-1000.,-7.,0.,7.,1000.]:
        for label in [0,1]:
            z=torch.tensor([value],requires_grad=True);loss,_=grouped_patient_loss(z,group([label]),z);bce=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([float(label)]));g=torch.autograd.grad(loss,z,retain_graph=True)[0];gb=torch.autograd.grad(bce,z)[0];assert torch.allclose(loss,bce,atol=1e-6,rtol=1e-6) and torch.allclose(g,gb,atol=1e-7,rtol=1e-6)
    checks['single_patient_equals_BCE_extreme_values_and_gradients']=True
    for labels in [[1,1,0,0],[1,1,1,1],[0,0,0,0]]:
        z=torch.tensor([-1000.,-2.,3.,1000.],requires_grad=True);row=group(labels);loss,_=grouped_patient_loss(z,row,z);pos=torch.tensor(labels).bool();manual=torch.logsumexp(torch.cat([z[~pos],z.new_zeros(1)]),0)+torch.logsumexp(torch.cat([-z[pos],z.new_zeros(1)]),0);assert torch.equal(loss,manual);loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all();assert bool((z.grad[pos]<=0).all() and (z.grad[~pos]>=0).all())
    # Hard negatives and hard positives dominate within their respective sets.
    z=torch.tensor([-8.,8.,-8.,8.],requires_grad=True);loss,_=grouped_patient_loss(z,group([0,0,1,1]),z);loss.backward();assert z.grad[1]>1000*z.grad[0] and abs(z.grad[2])>1000*abs(z.grad[3])
    checks.update(multiple_positives_all_preserved=True,all_positive_all_negative_and_finite_extremes=True,hard_positive_negative_gradient_emphasis=True)
    # Same document exact concept grouping; patient permutation leaves scalar unchanged.
    row=group([1,0,1,0],['A','A','B','B']);z=torch.tensor([.4,-.2,.7,.8],requires_grad=True);v,stats=grouped_patient_loss(z,row,z);z2=z.detach().clone().requires_grad_();r2=copy.deepcopy(row);r2['patients'].reverse()
    for b in r2['bags']:b['patient_index']=3-b['patient_index']
    w,_=grouped_patient_loss(z2,r2,z2);assert torch.equal(v,w);assert torch.equal(torch.autograd.grad(v,z)[0],torch.autograd.grad(w,z2)[0]);assert stats['concepts']==2
    # Only-masked concept contributes zero, never a constructed negative target.
    m=group([1]);m['patients'].append({'patient_id':'masked'});m['masked_bags']=[{'patient_id':'masked','concept':['hpo','HP:0000001'],'label':1},{'patient_id':'masked','concept':['nil','NIL literal; kept'],'label':1}];z=torch.tensor([2.],requires_grad=True);loss,stats=grouped_patient_loss(z,m,z);assert torch.allclose(loss,torch.nn.functional.softplus(-z[0])/2);assert stats['concepts']==2 and stats['valid_cells']==1 and stats['masked_cells']==2 and stats['all_masked_concepts']==1
    empty={'patients':[{'patient_id':'x'}],'bags':[],'masked_bags':[{'patient_id':'x','concept':['nil','unrepresentable'],'label':1}]};anchor=torch.tensor([2.],requires_grad=True);zero,stats=grouped_patient_loss(None,empty,anchor);zero.backward();assert zero==0 and anchor.grad==0 and stats['valid_cells']==0
    checks.update(patient_permutation_invariant=True,concept_groups_do_not_mix=True,masked_positives_never_negative=True,all_masked_concept_zero_in_all_C_mean=True,empty_document_graph_connected_zero=True)
    row={'pmc_id':'toy','token_length':8,'windows':[{'start':0,'end':4,'ids':[1,3,4,5,6,2]},{'start':4,'end':8,'ids':[1,7,8,9,10,2]}],'patients':[{'patient_id':'P1','spans':[[0,1],[6,7]]},{'patient_id':'P2','spans':[[7,8]]}], 'offsets':[[x*2,x*2+1] for x in range(8)],'passages':[{'index':0,'start':0,'end':7},{'index':1,'start':8,'end':15}], 'candidates':[[0,0],[1,2],[2,2],[4,5]],'positive_candidates':[1], 'bags':[{'patient_index':0,'concept':['hpo','A'],'instances':[1,3],'label':1},{'patient_index':1,'concept':['hpo','A'],'instances':[1,3],'label':1},{'patient_index':0,'concept':['nil','b'],'instances':[0],'label':0},{'patient_index':1,'concept':['nil','b'],'instances':[0],'label':0}]}
    torch.manual_seed(20261024);old=GeometryModel(ToyEncoder(),head_size=4,encoder_batch=1,score_chunk=2);rng=torch.get_rng_state().clone();torch.manual_seed(20261024);new=PatientSpanModel(ToyEncoder(),head_size=4,encoder_batch=1,score_chunk=2);assert torch.equal(rng,torch.get_rng_state());assert list(old.state_dict())==list(new.state_dict()) and all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    torch.manual_seed(97);a=old.loss(row);torch.manual_seed(97);b=new.loss(row,retain_hidden=True);assert torch.equal(a['NER_loss'],b['NER_loss']) and torch.equal(a['bag_logits'],b['bag_logits']);assert not torch.equal(a['bag_loss'],b['bag_loss']);b['loss'].backward();assert torch.isfinite(b['loss']) and all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None);assert b['hidden'].grad[6:].abs().sum()>0;assert new.geometry_table.grad.abs().sum()>0;checks.update(model_parameters_RNG_and_initial_scores_unchanged=True,only_patient_loss_scalar_changed=True,full_document_encoder_anchor_geometry_gradients=True)
    # Complete original train structure: content hash unchanged; no materialized new dataset.
    rows=unpack(Path(encoded)/'train.json.gz');count=Counter();groupstats=[]
    for r in rows:
        before=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest();z=torch.zeros(len(r['bags']),requires_grad=True);loss,st=grouped_patient_loss(z,r,z);loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all();assert before==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest();gs={}
        for b0 in r['bags']:gs.setdefault(tuple(b0['concept']),[]).append(b0['label'])
        count.update({'documents':1,'patients':len(r['patients']),'bags':len(r['bags']),'positive_bags':sum(x['label'] for x in r['bags']),'masked':len(r['masked_bags']),'masked_positive':sum(x['label'] for x in r['masked_bags']),'concepts':st['concepts'],'multiple_positive_concepts':sum(sum(v)>1 for v in gs.values()),'all_masked_concepts':st['all_masked_concepts']});groupstats.append({'pmc_id':r['pmc_id'],**st})
    assert count['documents']==32 and count['patients']==87 and count['bags']==3194 and count['positive_bags']==813 and count['masked_positive']==1
    return {'status':'PASS_GROUPED_PATIENT_MULTILABEL_CPU_CONTRACTS','checks':checks,'train_counts':dict(count),'per_document_group_shapes':groupstats,'original_ID_labels_bags_unchanged':True,'no_pretrained_weights_or_GPU_or_tokenization':True,'no_held_gold':True,'sources':{n:ref(Path(__file__).parent/n) for n in ['model.py','contracts.py','vendor/geometry_model.py','vendor/base_model.py','vendor/gp.py']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoded',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.encoded);save(a.output,r);print(json.dumps(r))
