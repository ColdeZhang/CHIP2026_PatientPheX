"""Fixed train-only crossing-boundary sample; preserve all original Top8 targets."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,math,sys,copy
from helpers import ref,write,load_jsonl
from schema import SYSTEM,serialize,strict_parse
from reuse.strict import HPO,verify_span
R=Path('/workspace/patientphex-runs/round2-20261023');OLD=R/'research/mention-candidate-selection-v1'

def select(out):
 train=R/'ontology_rules/nested-outer0-inner1/train32.jsonl';pool=R/'research/mention-hard-negative-readiness-20261024/grouped-candidate-census.jsonl'
 assert ref(train)['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a';assert ref(pool)['sha256']=='5cd972b91527a250abc67ef7ae26b416c5deb231683a9e6973ca1f3161d993a0'
 docs={str(d['pmc_id']):d for d in load_jsonl(train)};old=load_jsonl(OLD/'prepared-v1/train.jsonl');assert len(old)==2931 and ref(OLD/'prepared-v1/train.jsonl')['sha256']=='12b48f03a23eb27df708ce6e5d4ece2a776312cd43e0752b35e1ac49830de3f3'
 h=HPO('/workspace/patientphex-data/hp.obo');gold=defaultdict(lambda:defaultdict(list));norm=lambda s:' '.join(s.casefold().split());surfaces=set()
 for pmc,d in docs.items():
  for e in d['entities']:
   verify_span(d,e['text'],e['offset']);assert len(e['text'])==e['length']
   try:ids=set(h.identifiers(e['identifier']));assert 'HP:0000118' not in ids
   except (ValueError,AssertionError):ids=None
   no=str(e.get('note','')).upper()=='NO';gold[pmc][e['offset'],e['length']].append(dict(e,no=no,ids=ids))
   if not no:surfaces.add(norm(e['text']))
 pools=defaultdict(list);flags=Counter();n_overlap=0
 for r in load_jsonl(pool):
  if r['category']!='unmatched_boundary_overlap':continue
  n_overlap+=1;p=r['pmc_id'];a=r['offset'];n=r['length'];assert (a,n) not in gold[p]
  ov=[e for (b,m),ee in gold[p].items() if max(a,b)<min(a+n,b+m) for e in ee];reasons=[]
  contained=any(e['offset']<=a and a+n<=e['offset']+e['length'] for e in ov);covers=any(a<=e['offset'] and e['offset']+e['length']<=a+n for e in ov)
  if contained or covers:reasons.append('not_crossing_only')
  if len({(e['offset'],e['length']) for e in ov})!=1:reasons.append('multiple_annotation_spans')
  if any(e['no'] for e in ov):reasons.append('touches_explicit_NO')
  if any(e['ids'] is None or e['ids']=={'-1'} or len(e['ids'])!=1 for e in ov):reasons.append('touches_NIL_outbranch_compound')
  if any(c['exact'] for c in r['candidate_atoms']):reasons.append('dictionary_exact_support')
  if norm(r['text']) in surfaces:reasons.append('same_surface_positive_elsewhere')
  ids=set().union(*(set(h.identifiers(c['identifier'])) for c in r['candidate_atoms']));og=set().union(*(e['ids'] or set() for e in ov))
  if ids&og:reasons.append('same_ID_as_overlapping_annotation')
  flags.update(reasons)
  if not reasons:pools[p].append(r)
 assert n_overlap==6224 and sum(map(len,pools.values()))==809
 pos=Counter(r['pmc_id'] for r in old if r['target_status'] in ['HPO','NIL']);oldkeys={(r['pmc_id'],r['offset'],r['length']) for r in old};appended=[];selected=[];perdoc={}
 for pmc,d in sorted(docs.items()):
  cap=math.ceil(pos[pmc]/4);ranked=sorted(pools[pmc],key=lambda r:(-r['ranking_probability'],-max(c['similarity'] for c in r['candidate_atoms']),r['offset'],r['length']));picked=ranked[:cap];perdoc[pmc]=dict(eligible=len(ranked),cap=cap,selected=len(picked),shortfall=max(0,cap-len(picked)))
  for rank,r in enumerate(picked):
   off,n,quote=r['offset'],r['length'],r['text'];assert (pmc,off,n) not in oldkeys;pi=verify_span(d,quote,off);p=d['full_text'][pi];start=off-p['offset'];end=start+n;left=max(0,start-256);right=min(len(p['text']),end+256);context=p['text'][left:right];assert context[start-left:end-left]==quote
   appended.append(dict(row_id=f'{pmc}:{off}:{n}:mention-boundary',pmc_id=pmc,quote=quote,offset=off,length=n,target_status='BOUNDARY',answer=serialize({'BOUNDARY':True}),user_body=json.dumps(dict(context=context,mention=quote,mention_start=start-left,mention_end=end-left),ensure_ascii=False)))
   selected.append(dict(pmc_id=pmc,offset=off,length=n,rank0=rank,probability=r['ranking_probability'],similarity=max(c['similarity'] for c in r['candidate_atoms']),original_passage_index=pi,source_candidate=r))
 assert len(appended)==539
 prior=json.loads((OLD/'prepared-v1/manifest.json').read_text())['source_inputs'];cache=[]
 for key in ['cached_query','new_query']:
  p=Path(prior[key]['path']);assert ref(p)==prior[key];cache.extend(json.loads(p.read_text()))
 cached={r['quote'] for r in cache};assert len(cached)==2318;missing=sorted({r['quote'] for r in appended}-cached)
 out.mkdir(parents=True,exist_ok=False);art={'appended':write(out/'appended-pending.jsonl',appended,jsonl=True),'selected':write(out/'selected.json',selected),'queries':write(out/'queries.json',missing)}
 result=dict(status='COMPLETE_FIXED_539_BOUNDARY_SELECTION_NO_MODEL',source=ref(__file__),original_rows=2931,appended_rows=539,total_rows=3470,held_rows=1416,unique_appended_quotes=len({r['quote'] for r in appended}),missing_queries=len(missing),overlap_candidates=6224,strict_eligible=809,exclusion_flags=dict(flags),per_document=perdoc,artifacts=art,inputs=dict(train=ref(train),pool=ref(pool),original_top8_train=ref(OLD/'prepared-v1/train.jsonl')),cached_queries={k:prior[k] for k in ['cached_query','new_query']},recipe=dict(epochs=2,updates_per_epoch=434,updates=868,last_group_size=6,warmup_steps=44),no_held_gold_no_fit_no_quality=True)
 write(out/'selection.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_document','inputs','artifacts','cached_queries']}))

def finish(selection,query,out):
 selection=Path(selection);m=json.loads((selection/'selection.json').read_text());appended=load_jsonl(selection/'appended-pending.jsonl');prior=json.loads((OLD/'prepared-v1/manifest.json').read_text());bank={}
 for key in ['cached_query','new_query']:
  p=Path(m['cached_queries'][key]['path']);assert ref(p)==m['cached_queries'][key]
  for x in json.loads(p.read_text()):assert x['quote'] not in bank;bank[x['quote']]=x
 fresh=json.loads(Path(query).read_text());assert {r['quote'] for r in fresh}==set(json.loads((selection/'queries.json').read_text()))
 for x in fresh:assert x['quote'] not in bank;bank[x['quote']]=x
 names=json.loads(Path(prior['source_inputs']['names']['path']).read_text());canonical={v:k for k,v in names.items()};oldtrain=load_jsonl(OLD/'prepared-v1/train.jsonl');held=load_jsonl(OLD/'prepared-v1/held.jsonl');train=[]
 for original in oldtrain:
  row=copy.deepcopy(original);row['messages'][0]['content']=SYSTEM;assert {k:v for k,v in row.items() if k!='messages'}=={k:v for k,v in original.items() if k!='messages'};assert row['messages'][1:]==original['messages'][1:];train.append(row)
 for source in appended:
  cs=bank[source['quote']]['candidates'];assert len(cs)==8 and len({c['identifier'] for c in cs})==8
  candidates=[dict(code=f'C{i+1:02d}',identifier=c['identifier'],name=canonical[c['identifier']],score=c['score']) for i,c in enumerate(cs)];options='\n'.join(c['code']+' | '+c['identifier']+' | '+c['name'] for c in candidates)
  row={k:copy.deepcopy(v) for k,v in source.items() if k!='user_body'};row['candidates']=candidates;row['messages']=[dict(role='system',content=SYSTEM),dict(role='user',content=source['user_body']+'\n\nCandidate options (retrieval rank):\n'+options)];train.append(row)
 for row in train:assert strict_parse(row['answer'],row['candidates'],eos=True,capped=False)['status']==row['target_status']
 newheld=[]
 for original in held:
  row=copy.deepcopy(original);row['messages'][0]['content']=SYSTEM;assert 'answer' not in row and 'target_status' not in row;assert row['messages'][1:]==original['messages'][1:];newheld.append(row)
 assert len(train)==3470 and len(newheld)==1416;assert not {x['pmc_id'] for x in train}&{x['pmc_id'] for x in newheld}
 out.mkdir(parents=True,exist_ok=False);files={'train':write(out/'train.jsonl',train,jsonl=True),'held':write(out/'held.jsonl',newheld,jsonl=True),'names':write(out/'public-names.json',names)}
 result=dict(status='COMPLETE_3470_BOUNDARY_TOP8_DATA',files=files,source=ref(__file__),selection=ref(selection/'selection.json'),new_queries=ref(query),original_top8_manifest=ref(OLD/'prepared-v1/manifest.json'),train_rows=3470,held_rows=1416,states=dict(Counter(r['target_status'] for r in train)),original2931_targets_options_user_contexts_order_exact=True,shared_system_change='add independent BOUNDARY rejection; NO/NIL definitions unchanged',no_gold_option_insertion=True,no_held_gold=True)
 write(out/'manifest.json',result);print(json.dumps(result))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['select','finish']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--selection');ap.add_argument('--query');a=ap.parse_args();select(a.output) if a.mode=='select' else finish(a.selection,a.query,a.output)
