#!/usr/bin/env python3
"""CPU-only exact old124 encoding plus JSON-list overlap token weights; no model."""
import argparse, collections, hashlib, json, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def value_spans(answer):
    dec=json.JSONDecoder();i=0;spans=[];decoded={}
    def ws(p):
        while p<len(answer) and answer[p].isspace():p+=1
        return p
    i=ws(i);assert answer[i]=='{';i+=1
    while True:
        i=ws(i)
        if answer[i]=='}':i+=1;break
        key,n=dec.raw_decode(answer,i);assert isinstance(key,str) and key not in decoded;i=ws(n);assert answer[i]==':';i=ws(i+1)
        start=i;value,i=dec.raw_decode(answer,i);assert isinstance(value,list) and all(isinstance(v,str) for v in value)
        assert answer[start]=='[' and answer[i-1]==']';decoded[key]=value;spans.append(dict(cid=key,start=start,end=i,empty=not value))
        i=ws(i)
        if answer[i]==',':i+=1;continue
        assert answer[i]=='}';i+=1;break
    assert ws(i)==len(answer) and decoded==json.loads(answer)
    return spans

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import sys;sys.path.insert(0,str(a.base))
    import torch,transformers,qwen35_smoke as helper
    from transformers import AutoTokenizer
    start=time.monotonic();torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    assert sha(a.base/'train.jsonl')=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    assert sha(helper.__file__)=='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
    rows=[json.loads(l) for l in (a.base/'train.jsonl').read_text().splitlines()];assert len(rows)==124
    old=json.loads((a.base/'tokenizer-audit.json').read_text());oldrows={r['row_id']:r for r in old['rows']}
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);assert tok.is_fast
    records=[];counts=collections.Counter();mixed=[]
    for row in rows:
        en=helper.encode(row,tok);prefix=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        full=tok.apply_chat_template(row['messages']+[dict(role='assistant',content=row['answer'])],tokenize=False,add_generation_prompt=False,enable_thinking=False)
        t=tok(full,add_special_tokens=False,return_offsets_mapping=True);assert t['input_ids']==en['input_ids']
        assert hashlib.sha256(json.dumps(en['input_ids']).encode()).hexdigest()==oldrows[row['row_id']]['input_ids_sha256']
        spans=value_spans(row['answer']);assert [s['cid'] for s in spans]==list(row['code_map']);weights=[0]*len(en['input_ids']);positions=[];targets=[];valueset=set();empties=set()
        for s in spans:
            s['start']+=len(prefix);s['end']+=len(prefix)
            overlaps=[]
            for j,(left,right) in enumerate(t['offset_mapping']):
                if left<s['end'] and right>s['start']:
                    assert j>=en['prefix_tokens'];overlaps.append(j);valueset.add(j)
                    if s['empty']:empties.add(j)
                    if left<s['start'] or right>s['end']:
                        mixed.append(dict(row_id=row['row_id'],cid=s['cid'],position=j,token_id=en['input_ids'][j],token_text=full[left:right],outside_left=full[left:s['start']] if left<s['start'] else '',outside_right=full[s['end']:right] if right>s['end'] else ''))
            assert overlaps and min(t['offset_mapping'][j][0] for j in overlaps)<=s['start'] and max(t['offset_mapping'][j][1] for j in overlaps)>=s['end']
        for j,label in enumerate(en['labels']):
            if label!=-100:weights[j]=2 if j in valueset else 1;positions.append(j-1);targets.append(label)
        assert positions[0]==en['prefix_tokens']-1 and positions[-1]==len(en['input_ids'])-2
        assert weights[en['eos_supervised_position']]==1 and sum(w>0 for w in weights)==en['supervised_tokens']
        counts.update(rows=1,tokens=len(weights),supervised=en['supervised_tokens'],value_tokens=len(valueset),empty_value_tokens=len(empties),sum_weights=sum(weights),values=len(spans),empty_values=sum(s['empty'] for s in spans))
        records.append(dict(row_id=row['row_id'],full_tokens=len(weights),prefix_tokens=en['prefix_tokens'],input_ids_sha256=oldrows[row['row_id']]['input_ids_sha256'],labels_sha256=oldrows[row['row_id']]['labels_sha256'],weights=weights,positions=positions,targets=targets,spans=spans))
    assert counts['tokens']==402594 and counts['supervised']==7170
    torch.manual_seed(20261023);logits=torch.randn(1,17,23,requires_grad=True);labels=torch.tensor([[-100]*8+list(range(9))]);idx=torch.where(labels[0]!=-100)[0];selected=logits[:,idx-1,:];target=labels[0,idx]
    ce=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,23),labels[:,1:].reshape(-1),ignore_index=-100)
    ce2=torch.nn.functional.cross_entropy(selected[0],target);assert torch.allclose(ce,ce2,rtol=0,atol=1e-7)
    w=torch.tensor([1.,2.,1.,2.,1.,2.,1.,2.,1.]);expected=(torch.nn.functional.cross_entropy(selected[0],target,reduction='none')*w).sum()/w.sum();expected.backward();assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:,:7])==0
    fixture=['{"C000":[]}', '{"C000":["P1","P2"],"C001":[]}', '{ "C000" : ["P\\\"1"] }']
    for s in fixture:value_spans(s)
    assert not torch.cuda.is_initialized();a.output.mkdir(exist_ok=False)
    (a.output/'weights.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
    report=dict(status='CPU_old124_value_token_overlap_weights_complete_no_model_no_GPU',source_sha256=sha(__file__),train_sha256=sha(a.base/'train.jsonl'),old_audit_sha256=sha(a.base/'tokenizer-audit.json'),counts=dict(counts),max_tokens=max(r['full_tokens'] for r in records),mixed_value_boundary_tokens=mixed,weights_sha256=sha(a.output/'weights.json'),runtime=dict(torch=torch.__version__,transformers=transformers.__version__),no_GPU=True,elapsed_seconds=time.monotonic()-start,synthetic_causal_CE_equal=True,real_native_model_numerical_equivalence_not_yet_tested=True,policy='Weight2 iff actual full-string token offset overlaps complete JSON list value (including brackets/empty[]); every other assistant suffix/EOS token1; prompt0. Boundary tokens indivisible, outside punctuation explicitly recorded.',future_loss='Per-row sum(w*CE)/sum(w), then original group mean. logits_to_keep uses each supervised target j predecessor j-1, no native labels; all vocabulary logits retained at selected positions. Original full assistant JSON/EOS preserved.')
    (a.output/'cpu-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='mixed_value_boundary_tokens'},indent=2));print('mixed_boundary_count',len(mixed))
if __name__=='__main__':main()
