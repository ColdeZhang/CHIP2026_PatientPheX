"""The sole .3/proper-nested originaldev endpoint; freeze before reading gold."""
import argparse,copy,datetime,json,sys,time
from pathlib import Path
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector';sys.path.insert(0,str(COMMON))
from prepare_pool import sha,ref,checked,read,rows,write,sources
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def nonentities(d):return {k:v for k,v in d.items() if k!='entities'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();p=read(a.plan);sources(p);assert p['cut']==.3
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
 sm=read(checked(p['score_manifest']));assert sm['status']=='all16_full64_BioBERT_scores_frozen_before_decode_or_quality'
 model=read(checked(sm['model_manifest']));checked(model['model']);assert model['actual_iterations']==160
 scores=rows(checked(sm['output']['selector-scores.jsonl']));inputs=rows(checked(sm['input']));byid={d['pmc_id']:d for d in inputs}
 baseline=rows(checked(p['baseline']));oldbyid={d['pmc_id']:d for d in baseline};assert len(oldbyid)==len(baseline)==len(byid)==len(scores)==16
 assert set(byid)==set(oldbyid)=={d['pmc_id'] for d in scores}
 frozen_old=read(checked(p['baseline_freeze']));assert frozen_old['predictions']['nested']==p['baseline']
 sys.path.insert(0,p['nested_source']);import decode_nested_selector as decoder
 _,allowed,_,_=decoder.prior.base.parse_hpo(checked(p['hpo']));allowed.discard('HP:0000118')
 assert sha(decoder.__file__)==p['sources'][str(Path(decoder.__file__).resolve())]
 new=[];diags=[]
 for row in scores:
  did=row['pmc_id'];doc=byid[did];assert set(doc)=={'pmc_id','pmid','patient','full_text'}
  assert {k:v for k,v in oldbyid[did].items() if k not in ('entities','association')}==doc
  chosen,diag=decoder.decode(row['candidates'],decoder.greedy(row['candidates'],.3),doc,allowed,cut=.3)
  pred=copy.deepcopy(oldbyid[did]);pred['entities']=chosen;assert nonentities(pred)==nonentities(oldbyid[did]);new.append(pred);diags.append(dict(pmc_id=did,**diag))
 newbyid={d['pmc_id']:d for d in new};new=[newbyid[d['pmc_id']] for d in baseline]
 with (out/'biobert.nested.jsonl').open('x') as f:
  for d in new:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 write(out/'input-only-diagnostics.json',diags);sources(p)
 write(out/'complete-freeze.json',dict(frozen_at_utc=now(),plan_sha256=sha(a.plan),source_sha256=sha(__file__),baseline=p['baseline'],new_prediction=ref(out/'biobert.nested.jsonl'),score_manifest=p['score_manifest'],model_manifest=sm['model_manifest'],new_quality_evaluation_started=False,all_nonentity_objects_unchanged=True,relation_support_not_recomputed=True,seconds_before_gold=time.monotonic()-start))
 checked(p['baseline']);checked(ref(out/'biobert.nested.jsonl'))
 # No gold file is opened until this complete full-prediction freeze exists.
 write(out/'evaluation-start.json',dict(started_at_utc=now(),complete_freeze=ref(out/'complete-freeze.json')))
 sys.path.insert(0,p['evaluation_source']);import evaluate;import validate_predictions as validator
 assert evaluate.VERSION=='patientphex-public-approx-v2'
 assert sha(evaluate.__file__)==p['sources'][str(Path(evaluate.__file__).resolve())] and sha(validator.__file__)==p['sources'][str(Path(validator.__file__).resolve())]
 gold=rows(checked(p['gold']));assert {d['pmc_id'] for d in gold}==set(byid)
 assert all({k:v for k,v in d.items() if k not in ('entities','association')}==byid[d['pmc_id']] for d in gold)
 metrics=evaluate.evaluate_report(gold,new,'all','text');valid=validator.validate_predictions(gold,new,validator.read_hpo(checked(p['hpo'])))
 old=read(checked(p['baseline_metrics']));assert valid['valid'] and valid['error_count']==valid['warning_count']==0
 assert metrics['counts']['relation']==old['counts']['relation'] and metrics['patients']==old['patients']
 assert all(metrics['metrics'][k]==old['metrics'][k] for k in ('micF1','macF1'))
 write(out/'biobert.metrics.json',metrics);write(out/'biobert.validation.json',valid)
 result=dict(status='complete_unique_originaldev_development_endpoint',plan_sha256=sha(a.plan),source_sha256=sha(__file__),freeze=ref(out/'complete-freeze.json'),baseline_metrics=p['baseline_metrics'],new_metrics=ref(out/'biobert.metrics.json'),validation=ref(out/'biobert.validation.json'),baseline=old['metrics'],biobert=metrics['metrics'],deltas={k:metrics['metrics'][k]-old['metrics'][k] for k in old['metrics']},new_counts=metrics['counts'],old_counts=old['counts'],all_nonentity_objects_unchanged=True,relation_support_not_recomputed=True,old_metrics_reused=True,originaldev_is_development_not_blind=True,official_actions=False,parameters=p['parameters'],seconds=time.monotonic()-start,finished_at_utc=now())
 sources(p);write(out/'results.json',result);print(json.dumps({'metrics':metrics['metrics'],'deltas':result['deltas'],'seconds':result['seconds']}))
if __name__=='__main__':main()
