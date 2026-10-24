"""Complete F0 only: B4 union (new seed G intersect prior fixed support)."""
import argparse,copy,importlib,json,os,sys,time
from pathlib import Path
from common import B,G,ROUTE,S,checked,read,ref,save,verify_sources
D=B/'F0-postprocess-v1'
OLD=json.loads((G/'f0/pure-plan.json').read_text());GOLD=OLD['gold']
PROTOCOL=read(ROUTE/'proposal.json');SUPPORT=read(ROUTE/'evidence/support-inputs.json')

def checked(v):assert ref(v['path'])==v;return Path(v['path'])
def read(p):return json.loads(Path(p).read_text())
def publish(p,b):
 with Path(p).open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 fd=os.open(Path(p).parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def pairs(rows,ev):return {(str(d['pmc_id']),pid,h) for d in rows for pid,hs in ev.associations(d).items() for h in hs if h != 'NO'}
def primary_keys(base,constrained,original):return base | (constrained & original)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract-bind','normalize-bind','assemble','score','contract']);a=ap.parse_args();verify_sources()
 guard={'open':False}
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==os.path.realpath(GOLD['path']) and not guard['open']:raise PermissionError('Gold prohibited before complete frozen seed-ensemble endpoint')
 sys.addaudithook(audit)
 if a.stage=='contract':
  assert primary_keys({'b'},{'m','common'},{'g','common'})=={'b','common'}
  assert primary_keys({'b'},set(),{'b','old'})=={'b'}
  assert primary_keys({'b'},{'x'},set().union(set(),{'x'},set(),set()))=={'b','x'}
  sys.path.insert(0,str(G/'quality'));import support
  spans,mask=support.byte_helpers(OLD['references']['byte_helper'])
  raw='{"pmc_id":"D", "entities":[],"association":[{"patient_id":"P", "phenotype":["HP:0001250;HP:0001251", "unmapped;text"], "extra": 7}]}\n'
  rendered=support.replace_phenotypes(raw,{'P':['HP:0001250;HP:0001251','unmapped;text','HP:0001252']},spans,mask)
  assert mask(raw)==mask(rendered) and json.loads(rendered)['association'][0]['phenotype'][:2]==json.loads(raw)['association'][0]['phenotype']
  sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');import evaluate as ev
  d={'pmc_id':'D','association':[{'patient_id':'P','phenotype':['HP:0001250;HP:0001251','unmapped;text','NO']} ]}
  assert pairs([d],ev)=={('D','P','HP:0001250'),('D','P','HP:0001251'),('D','P','unmapped;text')}
  (S/'evidence').mkdir(exist_ok=True)
  save(S/'evidence/endpoint-contract.json',dict(status='PASS',pure_intersection_not_union=True,base_protected_even_when_absent_from_both_views=True,compound_old_prefix_and_nonphenotype_bytes_exact=True,literal_NIL_semicolon_not_split=True,NO_excluded=True,quality_calls=0,source=ref(__file__)))
 elif a.stage=='extract-bind':
  col=read(B/'infer-F0-v1/collection.json');assert col['inference_reported_success'] and col['all_actual_SHA_verified']
  for x in col['files'].values():checked(x)
  p=read(G/'f0/extraction-plan.json');frozen=read(S/'infer-plan.json')
  for key,n in [('complete','output/complete.json'),('wrapper','wrapper-complete.json'),('responses','output/responses.jsonl')]:p['inputs'][key]=ref(B/'infer-F0-v1/terminal'/n)
  p['inputs']['collection']=ref(B/'infer-F0-v1/collection.json');p['inputs']['prompts']=frozen['payload_sources']['held-prompts.jsonl']
  p['sources']['run.py']=ref(S/'run.py');p['sources']['extract_quotes.py']=ref(G/'f0/extract_quotes.py')
  p['remote_output']=frozen['output_dir'];p['local_output']=str(B/'infer-F0-v1/terminal/output');p['output_dir']=str(D/'quotes-v1');p['binding_source']=ref(__file__);p['original_collection']=ref(B/'infer-F0-v1/collection.json')
  save(S/'extraction-plan.json',p)
 elif a.stage=='normalize-bind':
  p=read(G/'f0/held-finalize-plan.json');prep=read(D/'normalization-preparation/manifest.json');p['preparation']=ref(D/'normalization-preparation/manifest.json');p['output_dir']=str(D/'normalized-v1');p['binding_source']=ref(__file__)
  for k,n in [('query_collection','collection.json'),('query_scores','terminal/output/query-scores.json'),('query_complete','terminal/output/complete.json')]:p[k]=ref(B/'F0-Sap-v1'/n) if prep['necessary_public_query_count'] else None
  save(S/'normalization-finalize-plan.json',p)
 elif a.stage=='assemble':
  p=copy.deepcopy(OLD);p['actual']={};ex=read(D/'quotes-v1/manifest.json');nm=read(D/'normalized-v1/manifest.json')
  p['actual'].update(collection=ref(B/'infer-F0-v1/collection.json'),extraction_plan=ref(S/'extraction-plan.json'),extraction_manifest=ref(D/'quotes-v1/manifest.json'),quotes=ex['quotes'],decisions=ex['decisions'],normalization_manifest=ref(D/'normalized-v1/manifest.json'),normalizations=nm['normalizations'],semantic_evidence=nm['semantic_evidence'])
  r=p['references'];r['baseline']=PROTOCOL['folds']['F0']['base'];r['baseline_metrics']=PROTOCOL['folds']['F0']['old_metrics'];r['original_pure']=SUPPORT['F0']['G'];r['extract_quotes']=ref(G/'f0/extract_quotes.py')
  for key in ['S','Q','C']:r['support_'+key]=SUPPORT['F0'][key]
  r['inherited_baseline']=SUPPORT['F0']['inherited_baseline']
  # Use the mature actual-raw and normalization verifier, not its old two-arm runner.
  sys.path.insert(0,str(G/'f0'));import actual as assemble;support=assemble.H
  for x in list(r.values())+list(p['actual'].values()):checked(x)
  inputs=support.jsonl(r['F0_input']['path'])
  normalized,coverage=assemble.verify_actual(p,inputs)
  for item in read(p['actual']['decisions']['path']):
   decision=item['decision']
   if decision['status']!='complete':assert not decision['anchors']
  strict=support.load_module('_copy_strict',r['strict']);hpo=strict.HPO(r['hpo']['path']);schema=support.load_module('_copy_schema',r['schema'])
  class BoundStrict:
   HPO=hpo
   verify_span=staticmethod(strict.verify_span)
  selected,decisions,counts=assemble.new_relations(normalized,inputs,BoundStrict,schema,r['normalization_policy']['sha256'])
  lines,base,ids=support.rows(r['baseline']['path']);oldpure=support.jsonl(r['original_pure']['path'])
  sys.path.insert(0,str(Path(r['evaluator']['path']).parent));import evaluate as ev,validate_predictions as va
  old=pairs(base,ev);original_view=pairs(oldpure,ev);support_view=original_view | set().union(*(pairs(support.jsonl(r[n]['path']),ev) for n in ['support_S','support_Q','support_C']))
  constrained={(did,pid,key[1]) for (did,pid),keys in selected.items() for key in keys}
  inherited=pairs(support.jsonl(r['inherited_baseline']['path']),ev);assert inherited<=old and inherited<=pairs(support.jsonl(r['support_S']['path']),ev)
  assert len(support_view-old)==141 and len((support_view-old)-original_view)==113
  assert len(old)==343 and len(original_view)==173 and len(base)==16 and sum(len(d['entities']) for d in base)==947
  assert sum(len(d['association']) for d in base)==47 and {str(d['pmc_id']) for d in oldpure}==set(ids)
  final=primary_keys(old,constrained,support_view);add=final-old;spans,mask=support.byte_helpers(r['byte_helper']);rendered=[]
  for line,d in zip(lines,base):
   values={a['patient_id']:a['phenotype']+sorted(h for did,pid,h in add if did==str(d['pmc_id']) and pid==a['patient_id']) for a in d['association']}
   rendered.append(support.replace_phenotypes(line,values,spans,mask))
  raw=''.join(rendered).encode();pred=[json.loads(x) for x in rendered];assert pairs(pred,ev)==final and all(mask(x)==mask(y) for x,y in zip(lines,rendered))
  out=D/'quality-v1';out.mkdir(exist_ok=False);publish(out/'prediction.jsonl',raw)
  fmt=va.validate_predictions(inputs,pred,va.read_hpo(r['hpo']['path']));save(out/'format.json',fmt);assert fmt['valid']
  save(out/'bound-inputs.json',dict(source=ref(__file__),execution_plan=ref(S/'infer-plan.json'),fixed_protocol=ref(ROUTE/'proposal.json'),references=r,actual=p['actual'],inherited_actual_verifier=ref(G/'f0/actual.py'),inherited_support=ref(G/'quality/metadata_recovery/support.py')))
  save(out/'freeze.json',dict(at_unix=time.time(),prediction=ref(out/'prediction.jsonl'),inputs=ref(out/'bound-inputs.json'),source=ref(__file__),formula='B4 UNION (G_new_seed_valid_rows INTERSECT (original_pure_G UNION S UNION Q UNION C))',baseline=343,constrained=len(constrained),original_G=173,support_union=len(support_view),intersection=len(constrained&support_view),added=len(add),added_pairs=sorted(add),removed=0,nonphenotype_bytes_exact=True,baseline_prefixes_exact=True,no_gold_opened=True,new_v2=0,old_v2=0,coverage=coverage,normalization_counts=counts,byteexact_to_baseline=raw==Path(r['baseline']['path']).read_bytes()))
  print(json.dumps(dict(freeze=ref(out/'freeze.json'),added=len(add),constrained=len(constrained))))
 else:
  out=D/'quality-v1';f=read(out/'freeze.json');assert f['source']==ref(__file__);checked(f['prediction']);p=read(checked(f['inputs']));r=p['references'];base=read(checked(r['baseline_metrics']))
  save(out/'evaluation-intent.json',dict(at_unix=time.time(),freeze=ref(out/'freeze.json'),new_calls=0 if f['byteexact_to_baseline'] else 1,old_calls=0))
  if f['byteexact_to_baseline']:save(out/'result.json',dict(status='CLOSED_BYTEEXACT_NO_INCREMENT',fixed_protocol=ref(ROUTE/'proposal.json'),metrics=base['metrics'],new_v2=0,old_v2=0,strict_positive=False));return
  for item in r.values():checked(item)
  assert base['metrics']['Score']==0.7281956289113009
  guard['open']=True;sys.path.insert(0,str(Path(r['evaluator']['path']).parent));import evaluate as ev,validate_predictions as va
  gold=va.load_jsonl(checked(GOLD));pred=va.load_jsonl(f['prediction']['path']);report=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text');save(out/'metrics.json',report)
  delta={k:report['metrics'][k]-v for k,v in base['metrics'].items()};assert delta['menF0']==delta['docF0']==0
  adds=set(map(tuple,f['added_pairs']));tp=len(adds & pairs(gold,ev))
  save(out/'result.json',dict(status='COMPLETE_SINGLE_FIXED_TRAINING_SEED_F0',fixed_protocol=ref(ROUTE/'proposal.json'),metrics=report['metrics'],baseline=base['metrics'],delta=delta,added=len(adds),added_TP=tp,added_FP=len(adds)-tp,strict_positive=delta['Score']>0,new_v2=1,old_v2=0,not_blind=True,NN_calls=0,fit_calls=0,freeze=ref(out/'freeze.json')));print(json.dumps(read(out/'result.json')))
if __name__=='__main__':main()
