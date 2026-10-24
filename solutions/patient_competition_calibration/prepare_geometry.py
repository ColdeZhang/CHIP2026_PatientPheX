"""Complete train32/pureF1 geometry, exact token spans, patient-concept bags.

No patient-span evidence labels are invented. NO entity spans are not positive
NER targets or positive-bag evidence. Unsupported positive bags are masked,
never re-labelled negative. Every original patient/document remains present.
"""
from pathlib import Path
from collections import defaultdict,Counter
from bisect import bisect_left,bisect_right
import argparse,hashlib,json,time
from tokenizers import BertWordPieceTokenizer
from common import read,save,ref,packed
from vendor.strict import HPO
from vendor.doc_encode import spans_for_length,owner_for_span


def geometry(doc,tokenizer,hpo,training,max_span):
 pmc=str(doc['pmc_id']);pieces=[];passages=[];cursor=0
 for pi,p in enumerate(doc['full_text']):
  text=p['text'];pieces.append(text);passages.append({'index':pi,'start':cursor,'end':cursor+len(text),'offset':int(p['offset']),'text':text});cursor+=len(text)+2
 body='\n\n'.join(pieces);en=tokenizer.encode(body,add_special_tokens=False);ids=en.ids;offsets=en.offsets;n=len(ids);assert n
 starts=[a for a,b in offsets];ends=[b for a,b in offsets];bounds=spans_for_length(n,510,384)
 windows=[{'start':a,'end':b,'ids':[tokenizer.token_to_id('[CLS]')]+ids[a:b]+[tokenizer.token_to_id('[SEP]')]} for a,b in bounds]
 # Full token recovery; every overlapping occurrence must agree.
 recovered=[None]*n
 for w in windows:
  for i,t in enumerate(w['ids'][1:-1],w['start']):assert recovered[i] in [None,t];recovered[i]=t
 assert recovered==ids
 def locate(m):
  matches=[]
  for p in passages:
   a=int(m['offset'])-p['offset'];b=a+int(m['length'])
   if 0<=a<b<=len(p['text']) and p['text'][a:b]==m['text']:matches.append((p['start']+a,p['start']+b))
  assert matches,(pmc,'unlocatable_original_span',m['offset'],m['length'])
  return matches[0]
 def token_span(a,b):
  i=bisect_right(ends,a);j=bisect_left(starts,b);assert 0<=i<j<=n
  return i,j,starts[i]==a and ends[j-1]==b
 patients=[];anchor_owner=[]
 for p in doc['patient']:
  seen=set();spans=[]
  for m in p['mention']:
   a,b=locate(m)
   if (a,b) in seen:continue
   seen.add((a,b));i,j,exact=token_span(a,b);wi=owner_for_span((i,j),bounds)
   spans.append([i,j]);anchor_owner.append({'patient_id':str(p['patient_id']),'char_span':[a,b],'token_span':[i,j],'window':wi,'exact_wordpiece':exact})
  assert spans;patients.append({'patient_id':str(p['patient_id']),'spans':spans})
 assert len({p['patient_id'] for p in patients})==len(patients)
 # All candidate wordpiece spans, <= max_span, inside a single original
 # passage, canonical physical owner. No token/word score prefilter.
 token_pi=[]
 for a,b in offsets:
  hit=[p['index'] for p in passages if p['start']<=a<b<=p['end']];assert len(hit)==1;token_pi.append(hit[0])
 candidates=[];coordinates=[];seenphysical={};candidate_owner=[]
 for i in range(n):
  p=passages[token_pi[i]]
  for j in range(i,min(n,i+max_span)):
   if token_pi[j]!=token_pi[i]:break
   a,b=offsets[i][0],offsets[j][1];text=body[a:b];off=p['offset']+a-p['start'];key=(off,len(text),text)
   if key in seenphysical:continue
   wi=owner_for_span((i,j+1),bounds);seenphysical[key]=len(candidates);candidates.append([i,j]);coordinates.append([off,len(text),p['index'],a-p['start'],b-p['start']]);candidate_owner.append(wi)
 lookup={tuple(x):i for i,x in enumerate(candidates)}
 row={'pmc_id':pmc,'patients':patients,'windows':windows,'token_length':n,'offsets':offsets,'passages':passages,'candidates':candidates,'coordinates':coordinates,'candidate_owner':candidate_owner,'anchor_owner':anchor_owner,'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
 stats=Counter(documents=1,patients=len(patients),patient_anchors=len(anchor_owner),tokens=n,windows=len(windows),candidate_spans=len(candidates),max_span_tokens=max_span)
 issues=[]
 if training:
  targets=set();ignored_ner=set();concept_pos=defaultdict(set);concept_no=defaultdict(set);allkeys=set();positive_any=Counter()
  for ei,e in enumerate(doc['entities']):
   a,b=locate(e);i,j,exact=token_span(a,b);keyphys=(int(e['offset']),int(e['length']),e['text']);is_no=str(e.get('note','')).upper()=='NO'
   stats['NO_records' if is_no else 'positive_entity_records']+=1
   try:keys=list(map(tuple,hpo.entity_keys(e['identifier'],e['text'])))
   except ValueError as ex:keys=[];issues.append({'entity_index':ei,'reason':'unlinked_HPO','detail':str(ex),'note':e.get('note')})
   allkeys.update(keys)
   for key in keys:positive_any[key]+=not is_no
   ci=seenphysical.get(keyphys)
   if not exact or ci is None:
    if not is_no and (i,j-1) in lookup:ignored_ner.add(lookup[(i,j-1)])
    issues.append({'entity_index':ei,'reason':'nonexact_wordpiece' if not exact else 'max_span_or_owner','token_length':j-i,'note':e.get('note'),'keys':[list(k) for k in keys]});stats['unrepresentable_NO' if is_no else 'unrepresentable_positive']+=1;continue
   assert candidates[ci]==[i,j-1]
   if not is_no:targets.add(ci)
   for k in keys:(concept_no if is_no else concept_pos)[k].add(ci)
  gold={str(p['patient_id']):set() for p in doc['patient']};invalid=[]
  for a in doc['association']:
   for v in a['phenotype']:
    try:gold[str(a['patient_id'])].update(map(tuple,hpo.relation_keys(v)))
    except ValueError as ex:invalid.append({'patient_id':str(a['patient_id']),'literal':v,'reason':str(ex)})
  allkeys.update(k for x in gold.values() for k in x);bags=[];masked=[]
  for pi,p in enumerate(patients):
   pid=p['patient_id']
   for k in sorted(allkeys):
    label=int(k in gold[pid]);instances=sorted(concept_pos[k])
    # Negative NO-only concepts still train absent relations. Positives with
    # only negated/unrepresentable spans must never become fake empty negatives.
    if not instances and not label:instances=sorted(concept_no[k])
    if not instances:
     masked.append({'patient_id':pid,'concept':list(k),'label':label,'reason':'no_representable_positive_instance' if label else 'no_representable_instance'});continue
    bags.append({'patient_index':pi,'concept':list(k),'instances':instances,'label':label})
  row.update({'ignored_NER_candidates':sorted(ignored_ner-targets),'positive_candidates':sorted(targets),'bags':bags,'masked_bags':masked,'invalid_relation_literals':invalid,'entity_issues':issues,'gold_relation_counts':[len(gold[p['patient_id']]) for p in patients]})
  stats.update({'NER_positive_spans':len(targets),'NER_ignored_unrepresentable_covering_spans':len(ignored_ner-targets),'bags':len(bags),'positive_bags':sum(x['label'] for x in bags),'masked_bags':len(masked),'masked_positive_bags':sum(x['label'] for x in masked),'valid_gold_relation_keys':sum(map(len,gold.values())),'empty_patients':sum(not x for x in gold.values()),'NIL_bags':sum(b['concept'][0]=='nil' for b in bags),'HPO_bags':sum(b['concept'][0]=='hpo' for b in bags),'invalid_relation_literals':len(invalid),'conflicting_ner_positive_and_NO_spans':len(targets&{x for vs in concept_no.values() for x in vs})})
 return row,dict(stats)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--input',required=True);ap.add_argument('--provenance',required=True);ap.add_argument('--vocab',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 train=read(a.train);held=read(a.input);prov=json.loads(Path(a.provenance).read_text());ti={str(x['pmc_id']) for x in train};hi={str(x['pmc_id']) for x in held};assert len(ti)==len(train)==32 and len(hi)==len(held)==16 and ti.isdisjoint(hi);assert ti==set(prov['training_document_ids']) and ti.isdisjoint(prov['excluded_document_ids']);assert all('entities' not in x and 'association' not in x for x in held);assert sum(len(x['patient']) for x in train)==87 and sum(len(x['patient']) for x in held)==39
 tok=BertWordPieceTokenizer(a.vocab,lowercase=True);hpo=HPO(a.hpo)
 # Training-only maxspan check BEFORE candidate generation; covering-token
 # counts include NO/unlinked targets too, while exact boundary failures stay visible.
 maxlen=0
 for doc in train:
  for e in doc['entities']:maxlen=max(maxlen,len(tok.encode(e['text'],add_special_tokens=False).ids))
 max_span=max(32,maxlen);splits={};refs={};shapes={}
 for name,docs,training in [('train',train,True),('held',held,False)]:
  encoded=[];total=Counter();bydoc=[]
  for doc in docs:
   row,stat=geometry(doc,tok,hpo,training,max_span);encoded.append(row);total.update({k:v for k,v in stat.items() if k!='max_span_tokens'});bydoc.append({'pmc_id':row['pmc_id'],**stat})
  refs[name]=packed(out/(name+'.json.gz'),encoded);splits[name]={'counts':dict(total),'documents':bydoc};shapes[name]={'maximum_tokens':max(x['token_length'] for x in encoded),'maximum_windows':max(len(x['windows']) for x in encoded),'maximum_patients':max(len(x['patients']) for x in encoded),'maximum_candidate_spans':max(len(x['candidates']) for x in encoded),'maximum_patient_candidates':max(len(x['patients'])*len(x['candidates']) for x in encoded),'maximum_bags':max((len(x.get('bags',[])) for x in encoded),default=0)}
 save(out/'preparation.json',{'status':'COMPLETE_FULL_TRAIN32_PURE_F1_NO_QUALITY','sources':{n:ref(getattr(a,n)) for n in ['train','input','provenance','vocab','hpo']},'source':ref(__file__),'files':refs,'splits':splits,'shapes':shapes,'max_span_tokens':max_span,'max_training_entity_wordpieces_alone':maxlen,'train_ids':sorted(ti),'held_ids':sorted(hi),'configuration':{'window_tokens':510,'stride':384,'encoder_batch':4,'head_size':64,'epochs':8,'seed':20261024,'learning_rate':3e-5,'weight_decay':.01,'gradient_clip':1.,'global_NER_loss':'whole-document unique physical candidate GP multilabel categorical logsumexp positive/negative plus zero','patient_bag_loss':'mean BCEWithLogits(max_instance_logit, patient_concept_label)','global_NER_weight':1.,'patient_bag_weight':1.,'NER_raw_floor':-2.,'NER_decision_logit':0.,'patient_decision_logit':0.,'optimizer_documents_per_step':1,'scheduler':'linear, 10 percent warmup'},'no_nearest_or_all_span_patient_targets':True,'unrepresentable_positive_bags_not_negative':True,'held_gold_read':False,'seconds':time.monotonic()-started})
 print(json.dumps({'splits':{k:v['counts'] for k,v in splits.items()},'shapes':shapes,'max_span':max_span,'seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
