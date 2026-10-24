"""Training64 OOF frozen-output diagnostic only; no model/evaluator/decoder/fit/cut sweep."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,time,sys
D=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def checked(r):
 p=Path(r['path']);assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'];assert 'dev16' not in str(p);return p

def units(row):
 out=defaultdict(set)
 for e in row['entities']:
  if str(e.get('note') or '').upper()!='NO':out[e['offset'],e['length']].update(e['identifier'].split(';'))
 return out

def overlap(a,b):return a[0]<b[0]+b[1] and b[0]<a[0]+a[1]
def split_match(t,p):
 t,p=set(t),set(p);exact=t&p;t-=exact;p-=exact
 n=len(exact)
 if '-1' in t and p:t.remove('-1');p.remove(sorted(p)[0]);n+=1
 return n,t,p

def supports(c,concept,truth):
 ids=set(c['identifier'].split(';'))
 return (concept in ids) if concept!='-1' else ('-1' in ids or bool(ids-(truth-{'-1'})))

def analyze(g,p,candidates):
 truth,guess=units(g),units(p);byspan=defaultdict(list);counts=Counter();fn=Counter();fp=Counter();groups=Counter();availability=Counter()
 for c in candidates:byspan[c['offset'],c['length']].append(c)
 groups['candidate_records']=len(candidates);groups['unique_spans']=len(byspan);groups['selected_entity_records']=len(p['entities']);groups['gold_positive_spans']=len(truth);groups['gold_mention_atoms']=sum(map(len,truth.values()));groups['mixed_linked_NIL_gold_spans']=sum('-1' in x and len(x)>1 for x in truth.values())
 for s,cs in byspan.items():
  ids={c['identifier'] for c in cs};gold=truth.get(s,set());positives=[c for c in cs if any(supports(c,x,gold) for x in gold)];negatives=[c for c in cs if not any(supports(c,x,gold) for x in gold)]
  groups['span_positive_any_gold' if gold else 'span_negative_no_exact_gold']+=1
  if len(ids)>1:
   groups['multiple_literal_ID_groups']+=1
   groups['multiple_ID_positive_span' if gold else 'multiple_ID_negative_span']+=1
  if positives and negatives:groups['span_with_correct_and_wrong_candidate_rows']+=1;groups['correct_wrong_pair_count']+=len(positives)*len(negatives)
  if gold and not positives:groups['positive_span_no_correct_ID_candidate']+=1
  if len(gold)>1:groups['candidate_spans_with_multiple_gold_atoms']+=1
 for s in set(truth)|set(guess):
  truthids=truth.get(s,set());n,missing,extra=split_match(truthids,guess.get(s,set()));counts.update(tp=n,fn=len(missing),fp=len(extra));cs=byspan.get(s,[])
  for concept in missing:
   support=[c for c in cs if supports(c,concept,truthids)]
   if support:
    best=max(c['selector_probability'] for c in support);kind='correct_candidate_below_fixed_cut' if best<.3 else 'correct_candidate_at_cut_not_selected'
    availability['missed_correct_candidate_with_selected_same_span_other_ID']+=bool(guess.get(s))
   elif cs:kind='exact_span_present_correct_literal_ID_absent'
   elif any(overlap(s,t) for t in byspan):kind='exact_span_absent_overlap_only'
   else:kind='no_overlapping_candidate_span'
   fn[kind]+=1
  for concept in extra:
   kind='correct_positive_span_wrong_or_extra_ID' if truthids else ('overlaps_gold_different_boundary' if any(overlap(s,t) for t in truth) else 'no_overlapping_positive_gold')
   fp[kind]+=1
 for s,truthids in truth.items():
  candidate_ids=set().union(*(set(c['identifier'].split(';')) for c in byspan.get(s,[]))) if byspan.get(s) else set();n,missing,_=split_match(truthids,candidate_ids);availability['available_gold_atoms_optimistic']+=n;availability['absent_gold_atoms']+=len(missing)
 return dict(counts=counts,fn_partition=fn,fp_partition=fp,groups=groups,availability=availability)

def main():
 started=time.monotonic();plan=read(D/'plan.json');assert sha(__file__)==plan['source_sha256'];assert plan['diagnostic_only'] and plan['cut']==.3
 for r in plan['references'].values():checked(r)
 truth={d['pmc_id']:d for d in rows(plan['references']['train64']['path'])};assert len(truth)==64;seen=set();folds=[];totals={k:Counter() for k in ['counts','fn_partition','fp_partition','groups','availability']}
 for fold in plan['folds']:
  scored=rows(checked(fold['scores']));pred=rows(checked(fold['prediction']));metrics=read(checked(fold['metrics']));train48={d['pmc_id'] for d in rows(checked(fold['train48']))};checked(fold['fit_manifest']);held={d['pmc_id'] for d in pred};assert len(held)==16 and len(train48)==48 and not (held&train48) and held|train48==set(truth);assert [d['pmc_id'] for d in scored]==[d['pmc_id'] for d in pred];assert not(seen&held);seen|=held
  agg={k:Counter() for k in totals}
  for s,p in zip(scored,pred):
   a=analyze(truth[p['pmc_id']],p,s['candidates'])
   for k in agg:agg[k].update(a[k])
  expected=metrics['counts']['mention'];assert all(agg['counts'][k]==expected[k] for k in ['tp','fp','fn']);assert sum(agg['fn_partition'].values())==agg['counts']['fn'];assert sum(agg['fp_partition'].values())==agg['counts']['fp']
  for k in agg:totals[k].update(agg[k])
  folds.append(dict(fold=fold['fold'],documents=16,aggregates=agg,cached_counts_match=True,held_disjoint_train48=True))
 assert seen==set(truth) and totals['groups']['mixed_linked_NIL_gold_spans']==0
 result=dict(status='complete_posthoc_train64_OOF_aggregate_diagnostic',source_sha256=sha(__file__),plan_sha256=sha(D/'plan.json'),folds=folds,totals=totals,documents=64,seconds=time.monotonic()-started,no_original_dev_gold_read=True,no_model_fit_predict_decoder_or_evaluator_call=True,no_new_predictions_or_score=True,limitations=['Descriptive development counts of already selected fixed .3 and nested predictions, not new independent evidence.','Optimistic candidate availability ignores compatibility and does not equal attainable score.','Compound literal identifiers split as frozen evaluator; no alt canonicalization/gold patch.','Correct/wrong group counts use candidate any supported atom and are an inventory, not a finalized ranking target; multi-gold groups require explicit policy.','No output, threshold or group choice is changed based on these counts.'])
 with (D/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(documents=64,totals=totals,seconds=result['seconds'])))
if __name__=='__main__':main()
