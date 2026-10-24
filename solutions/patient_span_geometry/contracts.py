"""CUDA-hidden, no pretrained weights: production delta/autograd contracts."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from model import PatientSpanModel,BasePatientSpanModel,document_geometry,bucket_indices,optimizer_groups
from common import save,ref,unpack

class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8)
        self.embedding=nn.Embedding(40,8);self.linear=nn.Linear(8,8);self.dropout=nn.Dropout(.1)
    def forward(self,input_ids,attention_mask,return_dict=True):
        return SimpleNamespace(last_hidden_state=self.dropout(self.linear(self.embedding(input_ids))))

def run(encoded):
    torch.set_num_threads(1)
    row={'pmc_id':'toy','token_length':8,'windows':[{'start':0,'end':4,'ids':[1,3,4,5,6,2]},{'start':4,'end':8,'ids':[1,7,8,9,10,2]}],'patients':[{'patient_id':'P1','spans':[[0,1],[6,7]]},{'patient_id':'P2','spans':[[7,8]]}], 'offsets':[[x*2,x*2+1] for x in range(8)],'passages':[{'index':0,'start':0,'end':7},{'index':1,'start':8,'end':15}], 'candidates':[[0,0],[1,2],[2,2],[4,5]],'positive_candidates':[1],'bags':[{'patient_index':0,'instances':[1,3],'label':1},{'patient_index':1,'instances':[1,3],'label':1},{'patient_index':0,'instances':[0],'label':0},{'patient_index':1,'instances':[0],'label':0}]}
    torch.manual_seed(20261024);old=BasePatientSpanModel(ToyEncoder(),head_size=4,encoder_batch=1,score_chunk=2);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(20261024);new=PatientSpanModel(ToyEncoder(),head_size=4,encoder_batch=1,score_chunk=2)
    assert torch.equal(old_rng,torch.get_rng_state())
    oldp=list(old.named_parameters());newp=[(n,p) for n,p in new.named_parameters() if n!='geometry_table'];assert [n for n,p in oldp]==[n for n,p in newp]
    assert all(torch.equal(p,q) for (_,p),(_,q) in zip(oldp,newp))
    torch.manual_seed(72);oa=old.loss(row,retain_hidden=True);oa['loss'].backward();after_old=torch.get_rng_state().clone()
    torch.manual_seed(72);na=new.loss(row,retain_hidden=True);na['loss'].backward();assert torch.equal(after_old,torch.get_rng_state())
    for key in ['loss','NER_loss','bag_loss','patient_span_scores','bag_logits']:assert torch.equal(oa[key],na[key]),key
    assert all((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad) for (_,p),(_,q) in zip(oldp,newp))
    assert new.geometry_table.grad.abs().sum()>0 and torch.isfinite(new.geometry_table.grad).all()
    assert na['hidden'].grad[6:].abs().sum()>0
    groups=optimizer_groups(new);assert [id(p) for p in groups[0]['params']]==[id(p) for n,p in newp];assert groups[1]['params']==[new.geometry_table]
    opt=torch.optim.AdamW(groups);sch=get_linear_schedule_with_warmup(opt,76,768);lrs=[]
    for step in range(768):lrs.append([x['lr'] for x in opt.param_groups]);opt.step();sch.step()
    assert lrs[0]==[0.,0.] and lrs[76]==[3e-5,.01] and lrs[-1][0]>0 and opt.param_groups[0]['lr']==0.
    # Signed direction, overlap, adjacency and logarithmic saturation.
    a=torch.tensor([[10,12,0]]);pairs=torch.tensor([[12,12],[13,13],[15,15],[9,9],[8,8],[6,6],[10,11]])
    bi,sa=bucket_indices(pairs,a,torch.zeros(20,dtype=torch.long));assert bi[:,0].tolist()==[12,13,14,12,11,10,12]
    far=torch.tensor([[0,0],[20000,20000]]);bi,_=bucket_indices(far,torch.tensor([[9000,9001,0]]),torch.zeros(20001,dtype=torch.long));assert bi[:,0].tolist()==[0,24]
    # All anchors, exact duplicates ignored, sorted permutation invariance.
    r2=copy.deepcopy(row);r2['patients'][0]['spans']=[[6,7],[0,1],[6,7]]
    assert document_geometry(r2)==document_geometry(row)
    new.eval();new._geometry_cache.clear();new.set_geometry(row,torch.device('cpu'))
    with torch.no_grad():new.geometry_table.copy_(torch.arange(50).reshape(25,2).float()/10)
    pp=torch.tensor(row['candidates']);g=new.geometry_scores(pp);manual=[]
    for anchors in new._anchors:
        bucket,same=bucket_indices(pp,anchors,new._token_passage);manual.append(torch.stack([new.geometry_table[bucket[:,k],same[:,k]] for k in range(len(anchors))]).mean(0))
    assert torch.equal(g,torch.stack(manual));assert g.shape==(2,4)
    hidden,patient=new.encode_document(row);expected=new.patient_scores(hidden,patient,pp);records=new.infer(row,floor=-1e9)
    assert len(records)==len(pp) and torch.allclose(torch.tensor([r['patient_logits'] for r in records]).T,expected,atol=1e-7,rtol=1e-6)
    changed=copy.deepcopy(row);changed['pmc_id']='moved';changed['patients'][0]['spans']=[[2,3]];new.set_geometry(changed,torch.device('cpu'));assert not torch.equal(g,new.geometry_scores(pp))
    # Every distant bag instance survives exactly; empty-patient negatives finite.
    new.train();new.zero_grad();empty=copy.deepcopy(row)
    for b in empty['bags']:b['label']=0
    out=new.loss(empty);out['loss'].backward();assert torch.isfinite(out['loss']) and torch.isfinite(new.geometry_table.grad).all()
    # Validate only new metadata projection on complete reused payloads; no tokenizer.
    counts={}
    for split in ['train','held']:
        rows=unpack(Path(encoded)/(split+'.json.gz'));c={'documents':len(rows),'patients':0,'unique_anchors':0,'candidates':0,'bag_instances':0,'max_anchor_count':0}
        for r in rows:
            tp,anchors=document_geometry(r);assert len(tp)==r['token_length']
            for pair,coord in zip(r['candidates'],r['coordinates']):assert tp[pair[0]]==tp[pair[1]]==coord[2]
            assert sum(len(x) for x in anchors)==len(r['anchor_owner'])
            c['patients']+=len(anchors);c['unique_anchors']+=sum(len(x) for x in anchors);c['max_anchor_count']=max(c['max_anchor_count'],sum(len(x) for x in anchors));c['candidates']+=len(r['candidates']);c['bag_instances']+=len({i for b in r.get('bags',[]) for i in b['instances']})
        counts[split]=c
    assert counts['train']['documents']==32 and counts['train']['patients']==87 and counts['held']['documents']==16 and counts['held']['patients']==39
    return {'status':'PASS_GEOMETRY_DELTA_CPU_CONTRACTS','no_pretrained_model_loaded':True,'no_cuda':True,'no_tokenization':True,'rng_unchanged_initialization_and_dropout_backward':True,'old_parameters_order_and_values_exact':True,'initial_logits_losses_old_gradients_exact':True,'zero_table_receives_finite_nonzero_gradient':True,'full_cross_window_graph_retained':True,'all_unique_anchor_mean_permutation_invariant':True,'signed_gap_overlap_adjacent_saturated_buckets':True,'train_infer_same_geometry':True,'distant_bag_instances_not_removed':True,'empty_patient_targets_finite':True,'optimizer_old_order_exact_new_group_only50':True,'schedule_first':lrs[0],'schedule_peak':lrs[76],'schedule_last_used':lrs[-1],'coverage':counts,'sources':[ref(__file__),ref(Path(__file__).with_name('model.py')),ref(Path(__file__).parent/'vendor/base_model.py')]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoded',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.encoded);save(a.output,r);print(json.dumps(r))
