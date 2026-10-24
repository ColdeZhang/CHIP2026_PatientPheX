"""Bind unchanged extraction/normalization; freeze the one prelocked GWA endpoint."""
import argparse,copy,importlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from synthesize import ref,save
S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F0'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
OLD=json.loads((G/'quality/plan.json').read_text());F0OLD=json.loads((G/'f0/quality-plan.json').read_text());GOLD=F0OLD['gold'];OLD['gold']=GOLD
OLD['references']['F0_input']=F0OLD['references']['input'];del OLD['references']['F1_input']
for key,name in [('extract_quotes','extract_quotes.py'),('held_prepare','held_prepare.py'),('held_finalize','held_finalize.py')]:OLD['references'][key]=ref(G/'f0'/name)

def checked(v):assert ref(v['path'])==v;return Path(v['path'])
def read(p):return json.loads(Path(p).read_text())
def publish(p,b):
 with Path(p).open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 fd=os.open(Path(p).parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def pairs(rows,ev):return {(str(d['pmc_id']),pid,h) for d in rows for pid,hs in ev.associations(d).items() for h in hs if h != 'NO'}
def primary_keys(base,mean,original):return base | (mean & original)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract-bind','normalize-bind','assemble','score','contract']);a=ap.parse_args()
 guard={'open':False}
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==os.path.realpath(GOLD['path']) and not guard['open']:raise PermissionError('Gold prohibited before single frozen GWA endpoint')
 sys.addaudithook(audit)
 if a.stage=='contract':
  assert 'F1_input' not in OLD['references'] and OLD['references']['F0_input']==F0OLD['references']['input']
  assert GOLD==F0OLD['gold']
  assert primary_keys({'b'},{'m','common'},{'g','common'})=={'b','common'}
  assert primary_keys({'b'},set(),{'b','old'})=={'b'}
  assert primary_keys({'b'},{'x'},{'x'})=={'b','x'}
  sys.path.insert(0,str(G/'quality'));import support
  spans,mask=support.byte_helpers(OLD['references']['byte_helper'])
  raw='{"pmc_id":"D", "entities":[],"association":[{"patient_id":"P", "phenotype":["HP:0001250;HP:0001251", "unmapped;text"], "extra": 7}]}\n'
  rendered=support.replace_phenotypes(raw,{'P':['HP:0001250;HP:0001251','unmapped;text','HP:0001252']},spans,mask)
  assert mask(raw)==mask(rendered) and json.loads(rendered)['association'][0]['phenotype'][:2]==json.loads(raw)['association'][0]['phenotype']
  sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');import evaluate as ev
  d={'pmc_id':'D','association':[{'patient_id':'P','phenotype':['HP:0001250;HP:0001251','unmapped;text','NO']} ]}
  assert pairs([d],ev)=={('D','P','HP:0001250'),('D','P','HP:0001251'),('D','P','unmapped;text')}
  save(S.parent/'evidence/F0-endpoint-contract-v2.json',dict(status='PASS',pure_intersection_not_union=True,base_protected_even_when_absent_from_both_views=True,compound_old_prefix_and_nonphenotype_bytes_exact=True,literal_NIL_semicolon_not_split=True,NO_excluded=True,quality_calls=0,source=ref(__file__)))
 elif a.stage=='extract-bind':
  col=read(D/'infer-v1/collection.json');assert col['inference_reported_success'] and col['all_actual_SHA_verified']
  p=read(G/'f0/extraction-plan.json');frozen=read(S/'infer-plan.json')
  for key,n in [('complete','output/complete.json'),('wrapper','wrapper-complete.json'),('responses','output/responses.jsonl')]:p['inputs'][key]=ref(D/'infer-v1/terminal'/n)
  p['inputs']['collection']=ref(D/'infer-v1/collection.json');p['inputs']['prompts']=frozen['payload_sources']['held-prompts.jsonl']
  p['sources']['run.py']=ref(S/'run.py');p['remote_output']=frozen['output_dir'];p['local_output']=str(D/'infer-v1/terminal/output');p['output_dir']=str(D/'quotes-v1');p['binding_source']=ref(__file__);p['original_collection']=ref(D/'infer-v1/collection.json')
  save(S/'extraction-plan.json',p)
 elif a.stage=='normalize-bind':
  p=read(G/'f0/held-finalize-plan.json');prep=read(D/'normalization-preparation/manifest.json');p['preparation']=ref(D/'normalization-preparation/manifest.json');p['output_dir']=str(D/'normalized-v1');p['binding_source']=ref(__file__)
  for k,n in [('query_collection','collection.json'),('query_scores','terminal/output/query-scores.json'),('query_complete','terminal/output/complete.json')]:p[k]=ref(D/'Sap-CPU100-v1'/n) if prep['necessary_public_query_count'] else None
  save(S/'normalization-finalize-plan.json',p)
 elif a.stage=='assemble':
  p=copy.deepcopy(OLD);p['actual']={};ex=read(D/'quotes-v1/manifest.json');nm=read(D/'normalized-v1/manifest.json')
  p['actual'].update(collection=ref(D/'infer-v1/collection.json'),extraction_plan=ref(S/'extraction-plan.json'),extraction_manifest=ref(D/'quotes-v1/manifest.json'),quotes=ex['quotes'],decisions=ex['decisions'],normalization_manifest=ref(D/'normalized-v1/manifest.json'),normalizations=nm['normalizations'],semantic_evidence=nm['semantic_evidence'])
  original=read(S/'execution-plan.json');r=p['references'];r['baseline']=original['baseline_prediction'];r['baseline_metrics']=original['baseline_metrics'];r['original_pure']=original['original_G_pure']
  # Use the mature actual-raw and normalization verifier, not its old two-arm runner.
  sys.path.insert(0,str(G/'quality'));import support,assemble
  for x in list(r.values())+list(p['actual'].values()):checked(x)
  inputs=support.jsonl(r['F0_input']['path']);from verify_direct import verify_actual
  normalized,coverage=verify_actual(p,inputs)
  strict=support.load_module('_mean_strict',r['strict']);hpo=strict.HPO(r['hpo']['path']);schema=support.load_module('_mean_schema',r['schema'])
  class BoundStrict:
   HPO=hpo
   verify_span=staticmethod(strict.verify_span)
  selected,decisions,counts=assemble.new_relations(normalized,inputs,BoundStrict,schema,r['normalization_policy']['sha256'])
  lines,base,ids=support.rows(r['baseline']['path']);oldpure=support.jsonl(r['original_pure']['path'])
  sys.path.insert(0,str(Path(r['evaluator']['path']).parent));import evaluate as ev,validate_predictions as va
  old=pairs(base,ev);original_view=pairs(oldpure,ev)
  mean={(did,pid,key[1]) for (did,pid),keys in selected.items() for key in keys}
  assert len(old)==343 and len(original_view)==173 and len(base)==16 and sum(len(d['entities']) for d in base)==947
  assert sum(len(d['association']) for d in base)==47 and {str(d['pmc_id']) for d in oldpure}==set(ids)
  final=primary_keys(old,mean,original_view);add=final-old;spans,mask=support.byte_helpers(r['byte_helper']);rendered=[]
  for line,d in zip(lines,base):
   values={a['patient_id']:a['phenotype']+sorted(h for did,pid,h in add if did==str(d['pmc_id']) and pid==a['patient_id']) for a in d['association']}
   rendered.append(support.replace_phenotypes(line,values,spans,mask))
  raw=''.join(rendered).encode();pred=[json.loads(x) for x in rendered];assert pairs(pred,ev)==final and all(mask(x)==mask(y) for x,y in zip(lines,rendered))
  out=D/'quality-v1';out.mkdir(exist_ok=False);publish(out/'prediction.jsonl',raw)
  # Pure new view is an unscored audit/future-reuse artifact, not a second endpoint.
  pure=copy.deepcopy(base)
  for d in pure:
   for a in d['association']:a['phenotype']=sorted(h for did,pid,h in mean if did==str(d['pmc_id']) and pid==a['patient_id'])
  assert pairs(pure,ev)==mean;publish(out/'mean-new-only.jsonl',b''.join((json.dumps(x,ensure_ascii=False)+'\n').encode() for x in pure))
  fmt=va.validate_predictions(inputs,pred,va.read_hpo(r['hpo']['path']));save(out/'format.json',fmt);assert fmt['valid']
  save(out/'bound-inputs.json',dict(source=ref(__file__),execution_plan=ref(S/'execution-plan.json'),normalization_amendment=ref(S.parent/'normalization-amendment.json'),references=r,actual=p['actual'],inherited_actual_verifier=ref(G/'quality/assemble.py'),actual_verifier=ref(S/'verify_direct.py'),inherited_support=ref(G/'quality/support.py')))
  save(out/'freeze.json',dict(at_unix=time.time(),prediction=ref(out/'prediction.jsonl'),mean_pure_unscored=ref(out/'mean-new-only.jsonl'),inputs=ref(out/'bound-inputs.json'),source=ref(__file__),formula='B4 UNION (G_meanweights_CPU_FP32query INTERSECT G_originalgreedy_FP16query)',baseline=343,mean=len(mean),original_G=173,intersection=len(mean&original_view),added=len(add),added_pairs=sorted(add),removed=0,nonphenotype_bytes_exact=True,baseline_prefixes_exact=True,no_gold_opened=True,new_v2=0,old_v2=0,coverage=coverage,normalization_counts=counts,byteexact_to_baseline=raw==Path(r['baseline']['path']).read_bytes()))
  print(json.dumps(dict(freeze=ref(out/'freeze.json'),added=len(add),mean=len(mean))))
 else:
  out=D/'quality-v1';f=read(out/'freeze.json');assert f['source']==ref(__file__);checked(f['prediction']);p=read(checked(f['inputs']));r=p['references'];base=read(checked(r['baseline_metrics']))
  save(out/'evaluation-intent.json',dict(at_unix=time.time(),freeze=ref(out/'freeze.json'),new_calls=0 if f['byteexact_to_baseline'] else 1,old_calls=0))
  if f['byteexact_to_baseline']:save(out/'result.json',dict(status='CLOSED_BYTEEXACT_NO_INCREMENT',normalization_amendment=ref(S.parent/'normalization-amendment.json'),joint_weight_and_query_precision_control=True,metrics=base['metrics'],new_v2=0,old_v2=0,strict_positive=False));return
  guard['open']=True;sys.path.insert(0,str(Path(r['evaluator']['path']).parent));import evaluate as ev,validate_predictions as va
  gold=va.load_jsonl(checked(GOLD));pred=va.load_jsonl(f['prediction']['path']);report=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text');save(out/'metrics.json',report)
  delta={k:report['metrics'][k]-v for k,v in base['metrics'].items()};assert delta['menF1']==delta['docF1']==0
  adds=set(map(tuple,f['added_pairs']));tp=len(adds & pairs(gold,ev))
  save(out/'result.json',dict(status='COMPLETE_SINGLE_PRELOCKED_ENDPOINT',normalization_amendment=ref(S.parent/'normalization-amendment.json'),joint_weight_and_query_precision_control=True,metrics=report['metrics'],baseline=base['metrics'],delta=delta,added=len(adds),added_TP=tp,added_FP=len(adds)-tp,strict_positive=delta['Score']>0,new_v2=1,old_v2=0,not_blind=True,NN_calls=0,fit_calls=0,freeze=ref(out/'freeze.json')));print(json.dumps(read(out/'result.json')))
if __name__=='__main__':main()
