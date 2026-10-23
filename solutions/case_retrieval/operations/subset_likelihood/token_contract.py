#!/usr/bin/env python3
"""Real Qwen tokenizer / finite legal JSON subset contract. CPU only."""
from pathlib import Path
import argparse,hashlib,json,os,time,inspect,gzip,math
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoTokenizer,Qwen3_5ForCausalLM

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def common_prefix(seq):
 n=min(map(len,seq));i=0
 while i<n and all(x[i]==seq[0][i] for x in seq[1:]):i+=1
 return i

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();b=a.directory;start=time.monotonic();torch.set_num_threads(1);assert transformers.__version__=='5.20.2' and not torch.cuda.is_initialized()
 source=json.loads(a.source.read_text());files={f['name']:f for f in source['files']};actual={}
 for n in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt']:
  actual[n]=ref(a.model/n);assert actual[n]['sha256']==files[n]['sha256'] and actual[n]['bytes']==files[n]['size']
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);selected=json.loads((b/'selected-inputs.json').read_text());greedy={r['row_id']:r['predicted_answer'] for r in json.loads((b/'selected-greedy-context.json').read_text())};encoded=[];all_stats=[];selected_stats=[]
 assert len(selected)==12
 for item in selected:
  row=item['prompt'];pids=row['patient_ids'];cids=list(row['code_map']);prior=greedy[row['row_id']];rendered=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);prompt_ids=tok.encode(rendered,add_special_tokens=False)
  for ci,cid in enumerate(cids):
   # Training serializer orders PIDs exactly as supplied input. Predictions, never gold, provide preceding CID context.
   prior_fields=[dump(c)+':'+dump([pid for pid in pids if pid in prior[c]]) for c in cids[:ci]]
   text_prefix=rendered+'{'+(','.join(prior_fields)+',' if prior_fields else '')+dump(cid)+':'
   delimiter=',' if ci+1<len(cids) else '}'
   branches=[]
   for mask in range(1<<len(pids)):
    subset=[pid for i,pid in enumerate(pids) if mask>>i&1];value=dump(subset)+delimiter;text=text_prefix+value;ids=tok.encode(text,add_special_tokens=False)
    assert tok.decode(ids,skip_special_tokens=False)==text and len(ids)<14000
    branches.append(dict(mask=mask,subset=subset,ids=ids,value=value,naive_concatenation_equal=tok.encode(text_prefix,add_special_tokens=False)+tok.encode(value,add_special_tokens=False)==ids))
   lcp=common_prefix([x['ids'] for x in branches]);assert lcp>=len(prompt_ids) and all(x['ids'][:len(prompt_ids)]==prompt_ids for x in branches)
   lengths=[]
   for branch in branches:
    K=len(branch['ids'])-lcp;assert K>0 and len(branch['ids'][:-1][-K:])==len(branch['ids'][lcp:])==K
    # Position lcp-1 predicts token lcp; last K logits of ids[:-1] exactly align with score labels.
    assert (len(branch['ids'])-1)-K==lcp-1
    if cid==item['selected_cid']:encoded.append(dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),cid=cid,concept=row['code_map'][cid],patient_ids=pids,mask=branch['mask'],subset=branch['subset'],input_ids=branch['ids'],score_start=lcp,score_labels=branch['ids'][lcp:],logits_to_keep=K,delimiter=delimiter))
    lengths.append(K)
   stat=dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),cid=cid,patients=len(pids),branches=len(branches),prompt_tokens=len(prompt_ids),common_context_tokens=lcp,complete_tokens_total=sum(len(x['ids']) for x in branches),max_complete_tokens=max(len(x['ids']) for x in branches),min_scored_tokens=min(lengths),max_scored_tokens=max(lengths),scored_tokens_total=sum(lengths),naive_boundary_mismatch=sum(not x['naive_concatenation_equal'] for x in branches),delimiter=delimiter)
   all_stats.append(stat)
   if cid==item['selected_cid']:selected_stats.append(stat)
 assert len(selected_stats)==12 and len(encoded)==98
 # Alignment and set marginal math check, no model invocation.
 logits=torch.tensor([[1.,3.,0.],[2.,0.,1.],[0.,2.,1.],[1.,0.,3.]])
 labels=torch.tensor([1,0,1,2]);full=torch.log_softmax(logits,dim=-1);start_idx=2
 score=full[start_idx-1:].gather(1,labels[start_idx-1:,None]).sum()
 tail=torch.log_softmax(logits[-3:],dim=-1).gather(1,labels[-3:,None]).sum();assert torch.equal(score,tail)
 for P in [1,2,3,5,7]:
  uniform=torch.zeros(1<<P,dtype=torch.float64);logz=torch.logsumexp(uniform,0)
  for j in range(P):
   containing=torch.tensor([i for i in range(1<<P) if i>>j&1]);m=torch.exp(torch.logsumexp(uniform[containing],0)-logz);assert abs(m.item()-.5)<1e-12
 signature=str(inspect.signature(Qwen3_5ForCausalLM.forward));forward_source=inspect.getsource(Qwen3_5ForCausalLM.forward);assert 'logits_to_keep' in signature and 'use_cache' in signature and 'slice_indices' in forward_source
 (b/'actual-forward-source.txt').write_text(forward_source);payload=json.dumps(encoded,ensure_ascii=False,separators=(',',':')).encode();(b/'selected98-encoded.json.gz').write_bytes(gzip.compress(payload,mtime=0))
 def summary(st):return dict(CIDs=len(st),branches=sum(x['branches'] for x in st),full_prefix_recompute_tokens=sum(x['complete_tokens_total'] for x in st),common_context_prefill_tokens=sum(x['common_context_tokens'] for x in st),max_complete_tokens=max(x['max_complete_tokens'] for x in st),scored_tokens=sum(x['scored_tokens_total'] for x in st),naive_boundary_mismatches=sum(x['naive_boundary_mismatch'] for x in st),min_scored_tokens=min(x['min_scored_tokens'] for x in st),max_scored_tokens=max(x['max_scored_tokens'] for x in st))
 report=dict(status='actual_CPU_tokenizer_and_likelihood_contract_PASS_no_forward_no_weights',source=ref(__file__),selected_input=ref(b/'selected-inputs.json'),selected_greedy=ref(b/'selected-greedy-context.json'),tokenizer=actual,framework={'torch':torch.__version__,'transformers':transformers.__version__},forward_signature=signature,forward_source=ref(b/'actual-forward-source.txt'),selected12_oneCID=summary(selected_stats),same12_allCIDs=summary(all_stats),selected_rows=selected_stats,encoded=ref(b/'selected98-encoded.json.gz'),encoding_schema='98fulltokenized legal subsets incl empty/all; score_start=longest common token prefix; forward input_ids[:-1], logits_to_keep=L-score_start, sum log_softmax.gather(input_ids[score_start:]); no cache or generation',CPU_checks=['roundtrip exact all options','original promptIDs remain exact prefix','all2^P subsets present once in inputPIDorder','right comma/brace retained','label shifts align with lastKlogits','uniform restricted-set P marginals exactly.5','realQwenforward supports logits_to_keep/use_cache'],gold_read=False,CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-start)
 assert report['CUDA_initialized'] is False
 with (b/'token-contract.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['tokenizer','selected_rows','forward_signature']},indent=2))
if __name__=='__main__':main()
