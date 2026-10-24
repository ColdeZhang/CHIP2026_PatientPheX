"""Preserve all2931 training targets and1416 pure occurrences; attach public options."""
from pathlib import Path
import argparse,json,hashlib,collections,copy
from schema import SYSTEM,serialize,strict_parse

def ref(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def rows(p):return [json.loads(v) for v in Path(p).read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser()
 for n in ['train','held','cached-query','new-query','names','output']:ap.add_argument('--'+n,required=True)
 a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);train=rows(a.train);held=rows(a.held);assert len(train)==2931 and len(held)==1416
 assert not ({x['pmc_id'] for x in train}&{x['pmc_id'] for x in held})
 names=json.loads(Path(a.names).read_text());canonical={v:k for k,v in names.items()};assert len(names)==len(canonical)==19119
 cached=json.loads(Path(a.cached_query).read_text());fresh=json.loads(Path(a.new_query).read_text());assert len(cached)==1284 and len(fresh)==1034
 assert not ({x['quote'] for x in cached}&{x['quote'] for x in fresh});bank={x['quote']:x for x in cached+fresh};assert len(bank)==2318
 stats=collections.Counter();files={};targetaudit=[]
 for split,inputs in [('train',train),('held',held)]:
  output=[]
  for source in inputs:
   query=bank[source['quote']];cs=query['candidates'];assert len(cs)==8 and len({x['identifier'] for x in cs})==8
   candidates=[{'code':f'C{i+1:02d}','identifier':x['identifier'],'name':canonical[x['identifier']],'score':x['score']} for i,x in enumerate(cs)]
   assert [x['role'] for x in source['messages']]==['system','user']
   options='\n'.join(x['code']+' | '+x['identifier']+' | '+x['name'] for x in candidates)
   row={k:copy.deepcopy(source[k]) for k in ('row_id','pmc_id','quote','offset','length')};row['candidates']=candidates;row['messages']=[{'role':'system','content':SYSTEM},{'role':'user','content':source['messages'][1]['content']+'\n\nCandidate options (retrieval rank):\n'+options}]
   if split=='train':
    status=source['target_status'];gold=[names[n] for n in json.loads(source['answer'])] if status=='HPO' else []
    if status=='HPO':
     selected=[x['code'] for x in candidates if x['identifier'] in gold];missing=bool(set(gold)-{x['identifier'] for x in candidates});target={'selected':selected,'missing':missing};stats['HPO_missing']+=missing;stats['HPO_empty_OTHER']+=not selected;stats['HPO_multi_selected']+=len(selected)>1
    else:assert status in ['NIL','NO'];target={status:True}
    row['target_status']=status;row['answer']=serialize(target);parsed=strict_parse(row['answer'],candidates,eos=True,capped=False);assert parsed['status']==status;stats[status]+=1
    targetaudit.append({'row_id':row['row_id'],'gold_ids':gold,'selected_ids':parsed['ids'],'target':target,'original_target_status':status})
   else:assert 'answer' not in source and 'target_status' not in source
   output.append(row)
  assert [r['row_id'] for r in output]==[r['row_id'] for r in inputs]
  p=out/(split+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output));files[split]=ref(p)
 assert stats['HPO']==2222 and stats['NIL']==44 and stats['NO']==665
 p=out/'training-target-audit.json';p.write_text(json.dumps(targetaudit,ensure_ascii=False)+'\n');files['target_audit']=ref(p)
 result={'status':'COMPLETE_FIXED_CANDIDATE_SELECTION_DATA_NO_HELD_GOLD','train_rows':2931,'held_rows':1416,'query_count':2318,'stats':dict(stats),'files':files,'source_inputs':{n:ref(getattr(a,n)) for n in ['train','held','cached_query','new_query','names']},'unchanged_original_occurrences_context_and_order':True,'no_gold_candidate_insertion':True,'no_extra_training_row_selection':True,'both_endpoint_policy':'future strongB exactentity union allaccepted / new-only exactnonentity shell; jointfreeze before2v2; no quality now','recipe':{'model':'Qwen/Qwen3-8B','fresh_base':True,'old_adapters':False,'epochs':2,'rows':2931,'updates':734,'updates_per_epoch':367,'last_batch_size':3,'batch_size':1,'gradient_accumulation':8,'warmup_steps':37,'learning_rate':1e-4,'rank':16,'lora_alpha':32,'dropout':.05,'seed':20261023,'weight_decay':.01,'max_grad_norm':1,'epoch_selection':'epoch2 only, noeval/holdout/earlystop'},'limitation':'New candidate prompt/target and578 negatives jointly differ from2353; Torch host to be bound, no pureprompt causal claim.'}
 (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
