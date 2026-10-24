from __future__ import annotations
import json,os,sys,time,hashlib
from types import SimpleNamespace
import torch
from torch import nn
import indexed_head as indexed
CausalLMOutputWithPast=SimpleNamespace
def fixed_cross_entropy(source: torch.Tensor, target: torch.Tensor, num_items_in_batch: torch.Tensor | None=None, ignore_index: int=-100, **kwargs) -> torch.Tensor:
    reduction = 'sum' if num_items_in_batch is not None else 'mean'
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == 'sum':
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss
native_fixed_cross_entropy=fixed_cross_entropy
def selected_causal_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        padded = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = padded[..., 1:].contiguous()
    target = shift_labels.view(-1).to(logits.device)
    keep = target != ignore_index
    source = logits.view(-1, vocab_size)[keep].float()
    return native_fixed_cross_entropy(source, target[keep], num_items_in_batch, ignore_index, **kwargs)
def forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, inputs_embeds: torch.FloatTensor | None=None, labels: torch.LongTensor | None=None, use_cache: bool | None=None, logits_to_keep: int | torch.Tensor=0, **kwargs: Unpack[TransformersKwargs]) -> CausalLMOutputWithPast:
    """
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        >>> model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3_5-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3_5-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\\nI'm not conscious, but I can talk to you."
        ```"""
    outputs: BaseModelOutputWithPast = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
"""Body joined to frozen source definitions by prepare_cpu_source.py."""

def run_checks():
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(20261024)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert torch.__version__=='2.7.1+cu126'
    measurements=[];negatives=[]
    def require(condition,message):
        if not condition:
            print(json.dumps({'status':'FAILED_CPU_CONTRACT','message':message,'measurements':measurements,'negative_checks':negatives}),flush=True)
            raise AssertionError(message)
    class Backbone:
        def __init__(self,h):self.h=h;self.inputs=[]
        def __call__(self,**kwargs):
            self.inputs.append(kwargs)
            return SimpleNamespace(last_hidden_state=self.h,past_key_values=None,hidden_states=None,attentions=None)
    class Model:
        def __init__(self,h,head):
            self.model=Backbone(h);self.lm_head=head;self.config=SimpleNamespace(vocab_size=head.out_features)
            self.loss_function=selected_causal_loss;self.calls=[]
        def __call__(self,**kwargs):
            self.calls.append(kwargs)
            return forward(self,**kwargs)
    def expect_rejection(label,function):
        try:function()
        except (ValueError,RuntimeError,IndexError):negatives.append(label);return
        require(False,'Expected rejection: '+label)
    for dtype in [torch.float64,torch.float32,torch.bfloat16]:
        for row in [[-100,-100,-100,2,7,3,9],[-100,3,-100,5,-100,7,-100],[-100]*6+[9]]:
            labels=torch.tensor([row]);T=len(row);H,V=11,29
            upstream=torch.randn(H,H,dtype=torch.float64,requires_grad=True)
            inputs=torch.randn(1,T,H,dtype=torch.float64)
            hidden=(inputs@upstream).to(dtype)
            head=nn.Linear(H,V,bias=False,dtype=dtype);head.weight.requires_grad_(False)
            m=Model(hidden,head)
            batch={'input_ids':torch.arange(T).reshape(1,T),'attention_mask':torch.ones(1,T,dtype=torch.long),'labels':labels}
            full=m(**batch,use_cache=False).loss
            reduced=indexed.indexed_causal_loss(m,batch,fixed_cross_entropy)
            gf=torch.autograd.grad(full,upstream,retain_graph=True)[0]
            gr=torch.autograd.grad(reduced,upstream)[0]
            info={'dtype':str(dtype),'case':'training_path','loss_abs_diff':float((full-reduced).abs()),'upstream_gradient_max_abs_diff':float((gf-gr).abs().max()),'finite':bool(torch.isfinite(gr).all())}
            measurements.append(info)
            require(bool(torch.allclose(full,reduced,rtol=1e-5,atol=1e-6)),'per-row mean loss')
            require(bool(torch.allclose(gf,gr,rtol=1e-5,atol=1e-6)),'gradient propagated to synthetic upstream')
            require(m.calls[-1]['labels'] is None and m.calls[-1]['use_cache'] is False and m.calls[-1]['return_dict'] is True,'inference kwargs must bypass native shift')
            require(m.model.inputs[-1]['input_ids'].shape==(1,T),'full backbone context retained')
            obs=indexed.frozen_head_observations(hidden.detach(),labels,head,selected_causal_loss,fixed_cross_entropy)
            measurements.append({'dtype':str(dtype),'case':'frozen_head_contract','observations':obs})
            record=[];indexed.record_then_require('synthetic',obs,lambda p,x:record.append(x))
            require(len(record)==1 and obs['passes'],'local-head contract record')
    for label,value in [('empty',torch.full((1,5),-100)),('batch2',torch.ones(2,5,dtype=torch.long)),('wrong_dtype',torch.ones(1,5)),('rank1',torch.ones(5,dtype=torch.long))]:
        expect_rejection(label,lambda value=value:indexed.selection(value))
    ix,t=indexed.selection(torch.tensor([[2,-100,3,-100,7]]))
    require(ix.tolist()==[1,3] and t.tolist()==[[3,7]],'original causal shift; label0 excluded and final target included')
    m=Model(torch.zeros(1,5,11),nn.Linear(11,29,bias=False));m.lm_head.weight.requires_grad_(False)
    batch={'input_ids':torch.zeros(1,5,dtype=torch.long),'attention_mask':torch.ones(1,5,dtype=torch.long),'labels':torch.tensor([[-100,-100,2,3,4]])}
    expect_rejection('unknown_key',lambda:indexed.indexed_causal_loss(m,dict(batch,past_key_values=None),fixed_cross_entropy))
    expect_rejection('shape_mismatch',lambda:indexed.indexed_causal_loss(m,dict(batch,attention_mask=torch.ones(1,4,dtype=torch.long)),fixed_cross_entropy))
    bad=lambda **kwargs:SimpleNamespace(logits=torch.zeros(1,5,29))
    expect_rejection('model_ignored_indices',lambda:indexed.indexed_causal_loss(bad,batch,fixed_cross_entropy))
    expect_rejection('native_short_logits_full_labels',lambda:forward(m,**batch,use_cache=False,logits_to_keep=torch.tensor([1,2,3])))
    head=nn.Linear(11,29,bias=False)
    expect_rejection('trainable_head',lambda:indexed.frozen_head_observations(torch.zeros(1,5,11),batch['labels'],head,selected_causal_loss,fixed_cross_entropy))
    head.weight.requires_grad_(False)
    expect_rejection('hidden_shape',lambda:indexed.frozen_head_observations(torch.zeros(1,4,11),batch['labels'],head,selected_causal_loss,fixed_cross_entropy))
    hidden=torch.zeros(1,5,11);hidden[:,1,:]=float('nan')
    observed=indexed.frozen_head_observations(hidden,batch['labels'],head,selected_causal_loss,fixed_cross_entropy)
    require(not observed['passes'],'NaN must fail')
    json.dumps(observed,allow_nan=False)
    records=[]
    expect_rejection('NaN_record_before_failure',lambda:indexed.record_then_require('must_exist',observed,lambda p,x:records.append((p,x))))
    require(len(records)==1 and records[0][0]=='must_exist' and records[0][1]['native_A']['logits']['nan_count']>0,'nonfinite observations saved before gate')
    print(json.dumps({'status':'PASS_INDEXED_HEAD_HELPER_CPU_AND_NEGATIVE_CONTRACT_NO_GPU','torch':torch.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'seconds':time.perf_counter()-started,'training_cases':9,'head_cases':9,'negative_checks':negatives,'causal_shift_index_check':True,'NaN_record_before_assert':True,'measurements':measurements,'scope':'Actual new helper; native forward body with synthetic backbone and frozen head. Real PEFT/decorators/GPU/RNG-restore capture not executed.'}),flush=True)

if __name__=='__main__':run_checks()
