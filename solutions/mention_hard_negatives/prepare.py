"""Append a fixed per-document quota of strictly annotation-disjoint OOF negatives."""
import argparse,json,math,sys,time
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parent/'reuse/mention'))
from helpers import ref,write,load_jsonl
from prepare import messages
from reuse.strict import verify_span

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--original',required=True);ap.add_argument('--census',required=True);ap.add_argument('--train',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.monotonic();old=Path(a.original);census=Path(a.census);report=json.loads((census/'census.json').read_text());grouped=load_jsonl(census/'grouped-candidate-census.jsonl');assert ref(census/'grouped-candidate-census.jsonl')==report['grouped_candidates'];rows=load_jsonl(old/'train.jsonl');assert len(rows)==2353 and ref(old/'train.jsonl')['sha256']=='8b7ac322777662f52e6b8e9bf3e5870174954953f29ca78a8a43749891bb6727';assert ref(a.train)['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a';docs={str(d['pmc_id']):d for d in load_jsonl(a.train)};assert len(docs)==32
 positive=Counter(r['pmc_id'] for r in rows if r['target_status'] in ['HPO','NIL']);pools=defaultdict(list)
 for r in grouped:
  if r['category']=='unmatched_disjoint_annotation_negative':pools[r['pmc_id']].append(r)
 selected=[];new=[];perdoc={};seen={(r['pmc_id'],r['offset'],r['length']) for r in rows}
 for pmc,d in sorted(docs.items()):
  k=math.ceil(positive[pmc]/4);ranked=sorted(pools[pmc],key=lambda r:(-r['ranking_probability'],-max(c['similarity'] for c in r['candidate_atoms']),r['offset'],r['length']));picked=ranked[:k];perdoc[pmc]={'HPO_NIL_targets':positive[pmc],'requested':k,'available':len(ranked),'selected':len(picked),'shortfall':max(0,k-len(picked))}
  for rank,r in enumerate(picked):
   off,n,text=r['offset'],r['length'],r['text'];assert (pmc,off,n) not in seen;seen.add((pmc,off,n));assert all(max(off,e['offset'])>=min(off+n,e['offset']+e['length']) for e in d['entities']);pi=verify_span(d,text,off);p=d['full_text'][pi];a0=off-p['offset'];b=a0+n;left=max(0,a0-256);right=min(len(p['text']),b+256);context=p['text'][left:right]
   new.append({'row_id':f'{pmc}:{off}:{n}:mention-hard-negative','pmc_id':pmc,'quote':text,'offset':off,'length':n,'target_status':'NO','messages':messages(context,text,a0-left,b-left),'answer':'[]','supervision_origin':'OOF_annotation_disjoint_negative'})
   selected.append({'pmc_id':pmc,'offset':off,'length':n,'quote':text,'rank0':rank,'probability':r['ranking_probability'],'similarity':max(c['similarity'] for c in r['candidate_atoms']),'original_passage_index':pi,'context_chars':len(context),'candidate_atoms':r['candidate_atoms']})
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);arts={'appended':write(out/'appended.jsonl',new,jsonl=True),'full_train':write(out/'train.jsonl',rows+new,jsonl=True),'selected':write(out/'selected.json',selected)}
 # JSON object identity of every original target is preserved, including original ordering.
 assert load_jsonl(out/'train.jsonl')[:2353]==rows;total=len(rows)+len(new);updates=math.ceil(total/8)
 result={'status':'COMPLETE_FIXED_QUARTER_POSITIVE_PER_DOCUMENT_NEGATIVE_PREPARATION','source':ref(__file__),'inputs':{'original_train':ref(old/'train.jsonl'),'train32':ref(a.train),'census':ref(census/'census.json'),'census_manifest':ref(census/'manifest.json'),'candidate_census':report['grouped_candidates']},'artifacts':arts,'original_rows':2353,'appended_rows':len(new),'train_rows':total,'training_states':dict(Counter(r['target_status'] for r in rows+new)),'per_document':perdoc,'unchanged_original_objects_and_order':True,'held_inputs_unchanged_and_not_reread_here':True,'schedule_proposal':{'epochs':2,'updates_per_epoch':updates,'updates':2*updates,'last_group_size':total%8 or 8,'warmup_steps':math.ceil(.05*2*updates)},'no_held_gold_model_fit_quality':True,'seconds':time.monotonic()-start}
 write(out/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','artifacts','per_document','source']}))

if __name__=='__main__':main()
