"""Reference building blocks, not a trained model. Requires torch/transformers.
No GPU is selected here. Linking must keep its independent frozen SapBERT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer

class SpanDetector(nn.Module):
    def __init__(self, model_dir, frozen=True):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_dir,local_files_only=True)
        hidden=self.encoder.config.hidden_size
        self.project=nn.Sequential(nn.LayerNorm(2*hidden),nn.Linear(2*hidden,256),nn.GELU(),nn.Dropout(.1))
        self.lexical=nn.Linear(256,1)
        self.occurrence=nn.Linear(256,1)
        self.set_frozen(frozen)

    def set_frozen(self,value):
        self.frozen=value
        for p in self.encoder.parameters():p.requires_grad_(not value)
        if value:self.encoder.eval()
        else:self.encoder.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.encoder.eval()
        return self

    def features(self,batch):
        # First sequence is candidate span; second is local context. Context
        # never comes from gold annotation other than locating training spans.
        first=batch['token_type_ids'].eq(0)&batch['attention_mask'].bool()&~batch['special_tokens_mask'].bool()
        inputs={k:batch[k] for k in ('input_ids','attention_mask','token_type_ids')}
        with torch.set_grad_enabled(self.training and not self.frozen):
            h=self.encoder(**inputs).last_hidden_state
            mean=(h*first.unsqueeze(-1)).sum(1)/first.sum(1,keepdim=True).clamp_min(1)
            return torch.cat([h[:,0],mean],dim=-1)

    def from_features(self,features):
        z=self.project(features.float())
        return {'lexical':self.lexical(z).squeeze(-1),'occurrence':self.occurrence(z).squeeze(-1)}

    def forward(self,batch):return self.from_features(self.features(batch))


def collate(rows,tokenizer,max_length=128):
    # only_second preserves candidate tokens; fail on an overlong first
    # sequence rather than silently clipping its annotation. Audit separately.
    packed=tokenizer([r['span'] for r in rows],[r['context'] for r in rows],padding=True,truncation='only_second',max_length=max_length,return_special_tokens_mask=True,return_tensors='pt')
    packed.update({key:torch.tensor([r[key] for r in rows],dtype=torch.float32) for key in ('lexical_y','occurrence_y','weight')})
    return packed


def weighted_bce(logits,labels,weights,pos_weight):
    # Fixed class weights are calculated from the complete training rows, not
    # re-estimated within each minibatch. Missing targets have label -1.
    valid=labels.ge(0)
    if not valid.any():return logits.sum()*0.
    pw=logits.new_tensor(pos_weight)
    loss=F.binary_cross_entropy_with_logits(logits[valid],labels[valid],pos_weight=pw,reduction='none')
    # Divide by raw count, so lowering a weak row's reliability lowers its
    # gradient; do not renormalize that reliability away within each class.
    return (loss*weights[valid]).sum()/valid.sum()


def class_weights(rows):
    result={}
    for target in ('lexical_y','occurrence_y'):
        mass=[sum(r['weight'] for r in rows if r[target]==v) for v in (0,1)]
        result[target]=max(.25,min(4.,mass[0]/max(mass[1],1e-6)))
    return result


def objective(outputs,batch,pos_weights):
    occurrence=weighted_bce(outputs['occurrence'],batch['occurrence_y'],batch['weight'],pos_weights['occurrence_y'])
    lexical=weighted_bce(outputs['lexical'],batch['lexical_y'],batch['weight'],pos_weights['lexical_y'])
    return occurrence+.35*lexical


def make_optimizer(model,finetune=False):
    head=list(model.project.parameters())+list(model.lexical.parameters())+list(model.occurrence.parameters())
    if finetune:
        model.set_frozen(False)
        return torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},{'params':head,'lr':2e-4}],weight_decay=.01)
    model.set_frozen(True)
    return torch.optim.AdamW(head,lr=1e-3,weight_decay=.01)

@torch.inference_mode()
def score_rows(model,tokenizer,rows,device,batch_size=128):
    model.eval();result=[]
    for start in range(0,len(rows),batch_size):
        # Inference receives raw span/context only; labels here are placeholders
        # for collate, not read from input. Zero weights are not model inputs.
        subset=[dict(r,lexical_y=-1.,occurrence_y=-1.,weight=0.) for r in rows[start:start+batch_size]]
        batch={k:v.to(device) for k,v in collate(subset,tokenizer).items()}
        output=model(batch)
        for p,q in zip(output['lexical'].sigmoid().tolist(),output['occurrence'].sigmoid().tolist()):result.append({'neural_lexical_probability':p,'neural_occurrence_probability':q})
    return result
