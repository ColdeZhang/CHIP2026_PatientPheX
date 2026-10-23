"""Diagnostic calculations, not an official evaluator. Run only against local train.
Assumptions: all supplied sections; note=NO removed; semicolon IDs split;
mention units deduplicated; gold pure -1 spans wildcard-match any predicted ID;
doc -1 units represented by exact raw text (sensitivity with omission included).
"""
import json, sys, importlib.util, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
G=[json.loads(x) for x in (ROOT/'data/train/PatientPheX-train.jsonl').read_text().splitlines()]
P=[json.loads(x) for x in (ROOT/'outputs/local_train_v10.jsonl').read_text().splitlines()]
assert len(G)==len(P)==80
assert [d['pmc_id'] for d in G]==[d['pmc_id'] for d in P]
spec=importlib.util.spec_from_file_location('old_evaluate',ROOT/'src/evaluate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def stats(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
 return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)

def canon(doc):
 out=[]
 for e in doc['entities']:
  if e.get('note')=='NO':continue
  for h in str(e['identifier']).split(';'):
   out.append((e['offset'],e['length'],h.strip(),e['text']))
 return out

def calculate(doc_noid='text',multiplicity=False):
 mention=collections.Counter(); document=collections.Counter();span=collections.Counter()
 errors=collections.Counter();g_errors=collections.Counter();examples=collections.defaultdict(list)
 neg=collections.Counter();duplicate=collections.Counter()
 for g,p in zip(G,P):
  ge=canon(g);pe=canon(p)
  span_ids=collections.defaultdict(set)
  for s,l,h,txt in ge:span_ids[(s,l)].add(h)
  unknown={(s,l):txt for s,l,h,txt in ge if span_ids[(s,l)]=={'-1'}}
  # Gold no-ID concept: exact span is the matching target, regardless predicted ID.
  def units(es):
   seq=[(s,l,'-1' if (s,l) in unknown else h) for s,l,h,txt in es]
   return collections.Counter(seq) if multiplicity else collections.Counter(set(seq))
  gu,pu=units(ge),units(pe)
  tp=sum((gu&pu).values());fp=sum((pu-gu).values());fn=sum((gu-pu).values())
  mention.update(tp=tp,fp=fp,fn=fn)
  gs={(s,l) for s,l,h in gu};ps={(s,l) for s,l,h in pu}
  span.update(tp=len(gs&ps),fp=len(ps-gs),fn=len(gs-ps))
  for unit in pu-gu:
   s,l,h=unit
   kind='id_mismatch_at_exact_positive_span' if (s,l) in gs else 'overlap_with_gold_but_wrong_boundary' if any(s<z+w and z<s+l for z,w in gs) else 'no_overlap_with_gold'
   errors[kind]+=(pu-gu)[unit]
   if len(examples['fp_'+kind])<4:examples['fp_'+kind].append(dict(pmc_id=g['pmc_id'],span=[s,l],identifier=h,text=[txt for a,b,c,txt in pe if a==s and b==l][:1]))
  for unit in gu-pu:
   s,l,h=unit
   kind='id_mismatch_at_exact_positive_span' if (s,l) in ps else 'overlap_with_prediction_but_wrong_boundary' if any(s<z+w and z<s+l for z,w in ps) else 'no_overlap_with_prediction'
   g_errors[kind]+=(gu-pu)[unit]
  def concept_set(es):
   result=set()
   for s,l,h,txt in es:
    if (s,l) in unknown:h='-1';txt=unknown[(s,l)]
    if h=='-1':
     if doc_noid=='text':result.add(('NOID_TEXT',txt))
     elif doc_noid=='sentinel':result.add(('ID','-1'))
    else:result.add(('ID',h))
   return result
  gd,pd=concept_set(ge),concept_set(pe)
  document.update(tp=len(gd&pd),fp=len(pd-gd),fn=len(gd-pd))
  gn={(e['offset'],e['length']) for e in g['entities'] if e.get('note')=='NO'}
  pn={(e['offset'],e['length']) for e in p['entities'] if e.get('note')=='NO'}
  neg.update(gold_no=len(gn),pred_positive_at_gold_no=len(ps&gn),pred_no_at_gold_no=len(pn&gn))
  duplicate.update(gold_original=len(ge)-len(set(ge)),pred_original=len(pe)-len(set(pe)))
 return dict(mention=stats(**mention),document=stats(**document),span_only_positive=stats(**span),fp_categories=dict(errors),fn_categories=dict(g_errors),examples=dict(examples),negation=dict(neg),duplicate_units=dict(duplicate))

assoc=collections.Counter();patient=[]
empty_empty=[];empty_gold_fp=[];gold_nonempty_pred_empty=[]
for g,p in zip(G,P):
 ga={x['patient_id']:set(x['phenotype']) for x in g['association']}
 pa={x['patient_id']:set(x['phenotype']) for x in p['association']}
 assert set(pa)<=set(ga)
 for pid,gs in ga.items():
  ps=pa.get(pid,set());tp=len(gs&ps);fp=len(ps-gs);fn=len(gs-ps)
  assoc.update(tp=tp,fp=fp,fn=fn)
  v=stats(tp,fp,fn)
  if not gs and not ps:
   v.update(precision=1.0,recall=1.0,f1=1.0);empty_empty.append([g['pmc_id'],pid])
  if not gs and ps:empty_gold_fp.append([g['pmc_id'],pid,len(ps)])
  if gs and not ps:gold_nonempty_pred_empty.append([g['pmc_id'],pid,len(gs)])
  patient.append(v)
association=dict(micro=stats(**assoc),macro={k:sum(v[k] for v in patient)/len(patient) for k in ('precision','recall','f1')},n_patients=len(patient),empty_empty=empty_empty,empty_gold_with_fp=empty_gold_fp,gold_nonempty_pred_empty=gold_nonempty_pred_empty)
old_values=(*old.evaluate_entities(P,G),*old.evaluate_association(P,G))
old_result=dict(zip(('menF1','docF1','micF1','macF1'),old_values));old_result['Score']=sum(old_values)/4
primary=calculate()
primary['association']=association
primary['score']=sum((primary['mention']['f1'],primary['document']['f1'],association['micro']['f1'],association['macro']['f1']))/4
sensitivity={}
for noid in ('omit','sentinel'):
 v=calculate(doc_noid=noid);sensitivity[noid+'_doc_noid']=dict(document=v['document'],score=sum((v['mention']['f1'],v['document']['f1'],association['micro']['f1'],association['macro']['f1']))/4)
v=calculate(multiplicity=True);sensitivity['multiset_mentions']=v['mention']

# Minimal counterexamples, all run through the original functions.
def e(h='HP:0000001',off=0):return dict(offset=off,length=3,identifier=h,text='abc',note=None,type='Phenotype')
def d(entities,pmc='x'):return dict(pmc_id=pmc,pmid=pmc,entities=entities,association=[])
cases={
 'wrong_id_same_span':(old.evaluate_entities([d([e('HP:9999999')])],[d([e()])]),'Both men and doc return 1 despite every HPO ID being wrong.'),
 'two_docs_one_empty_prediction':(old.evaluate_entities([d([e()],'a'),d([],'b')],[d([e()],'a'),d([e()],'b')]),'docF1 ignores the all-missed second document.'),
 'truncated_predictions':(old.evaluate_entities([d([e()],'a')],[d([e()],'a'),d([e()],'b')]),'Both metrics return 1; missing tail document silently ignored by zip.'),
 'wrong_doc_id':(old.evaluate_entities([d([e()],'WRONG')],[d([e()],'TRUE')]),'Both metrics return 1 despite mismatched document IDs.'),
 'empty_patient_both':(old.evaluate_association([dict(association=[dict(patient_id='P1',phenotype=[])])],[dict(association=[dict(patient_id='P1',phenotype=[])])]),'macro gives 0; official text requires 1 for this patient.')}
output=dict(scope='train memorization/backfit diagnostic only; no generalization estimate',assumptions=['All supplied full_text segments; exact official body filter unavailable. Gold includes TITLE and ABSTRACT.', 'NO entities excluded; semicolon HPOs split.', 'Mention units deduplicated; gold pure -1 exact spans match predicted IDs as wildcard; mixed HPO;-1 uses literal units (not specified publicly).', 'Document no-ID represented by exact text; omission/sentinel sensitivities reported.', 'Not the official platform score.'],old_evaluator=old_result,public_rule_approximation=primary,sensitivity=sensitivity,counterexamples=cases)
g_noid=[];p_noid=[];gpos=0;p_exact=0
for gd,pd in zip(G,P):
 spans={(e['offset'],e['length']) for e in pd['entities'] if e.get('note')!='NO'}
 for ent in gd['entities']:
  if ent['identifier']=='-1' and ent.get('note')!='NO':
   gpos+=1;p_exact+=(ent['offset'],ent['length']) in spans
 for ad in gd['association']:
  g_noid.extend((gd['pmc_id'],ad['patient_id'],h) for h in set(ad['phenotype']) if not h.startswith('HP:'))
 for ad in pd['association']:
  p_noid.extend((gd['pmc_id'],ad['patient_id'],h) for h in set(ad['phenotype']) if not h.startswith('HP:'))
output['no_id_coverage']=dict(positive_pure_noid_entities=gpos,exact_span_predicted_count=p_exact,gold_non_hpo_association_count=len(g_noid),pred_non_hpo_association_count=len(p_noid),examples=g_noid[:10])
out=Path('/tmp/patientphex-evaluation-audit/results.json');out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='public_rule_approximation'},ensure_ascii=False,indent=2))
print('PRIMARY',json.dumps({k:v for k,v in primary.items() if k not in ('examples','association')},ensure_ascii=False,indent=2))
print('ASSOCIATION',json.dumps(association,ensure_ascii=False,indent=2))
