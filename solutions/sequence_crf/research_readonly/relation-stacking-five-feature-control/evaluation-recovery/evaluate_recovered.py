"""Only recover canonical validator import; original model/plan/prediction immutable."""
import argparse,importlib.util,json,sys
from pathlib import Path

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();sys.dont_write_bytecode=True
 rp=Path(a.plan).resolve();recovery=json.loads(rp.read_text());c=load('original_five_feature_control',recovery['original_source']['path'])
 c.checked(recovery['original_source']);c.checked(recovery['source']);c.require(c.sha(__file__)==recovery['source']['sha256'],'Recovery source changed')
 c.checked(recovery['original_plan']);plan=c.read(recovery['original_plan']['path']);pr=c.plan_check(plan,recovery['original_plan']['path']);d=Path(plan['output_dir'])
 c.checked(recovery['original_failure']);failed=c.read(recovery['original_failure']['path']);c.require(failed['F0_gold_read'] is False and failed['evaluate_report_calls']==0,'Not an import-only recovery')
 c.checked(recovery['original_reservation']);c.checked(recovery['freeze']);frozen=c.read(recovery['freeze']['path']);c.require(frozen['plan']==pr and frozen['status']=='all16_F0_predictions_complete_frozen_before_F0_gold','Frozen input differs')
 for name in ['fit_state','prediction','scores','model']:c.checked(frozen[name])
 c.require(not (d/'metrics.json').exists() and not (d/'results.json').exists(),'Original endpoint may already have scored')
 out=Path(recovery['output_dir']);out.mkdir(exist_ok=False)
 c.save(out/'evaluation-started.json',{'at_utc':c.now(),'recovery_plan':c.ref(rp),'freeze':recovery['freeze'],'first_F0_quality_call_still_pending':True,'new_v2_calls':1})
 # Correct only dependency registration order; both files remain pinned to the original plan.
 validator=load('validate_predictions',plan['refs']['validator']['path'])
 evaluator=load('five_feature_evaluator_recovered',plan['refs']['evaluator']['path'])
 c.require(evaluator.VERSION=='patientphex-public-approx-v2','Evaluator version differs')
 inputs=c.rows(plan['refs']['F0_input']['path']);pred=c.rows(frozen['prediction']['path']);hpo=validator.read_hpo(plan['refs']['hpo']['path'])
 valid=validator.validate_predictions(inputs,pred,hpo);c.save(out/'validation.json',valid);c.require(valid['valid'],'Frozen F0 format invalid')
 # First F0 gold-byte access: original complete16 prediction and its original SHA freeze already verified.
 goldref=plan['gold_after_freeze'];c.require(c.sha(goldref['path'])==goldref['sha256'],'F0 gold version changed');gold=c.rows(goldref['path'])
 result=evaluator.evaluate_report(gold,pred,'all','text');c.save(out/'metrics.json',result)
 baselines={}
 for name,item in plan['comparators'].items():
  value=c.read(c.checked(item['ref']))
  for key in item['metric_path']:value=value[key]
  baselines[name]=value
 c.require(result['documents']==16 and result['patients']==47 and all(result['metrics'][key]==b[key] for b in baselines.values() for key in ['menF1','docF1']),'Fixed F0/entity scope changed')
 maxscore=max(x['Score'] for x in baselines.values());maxmacro=max(x['macF1'] for x in baselines.values());keep=result['metrics']['Score']>maxscore and result['metrics']['macF1']>=maxmacro
 summary={'status':'complete_unique_five_feature_F1_to_F0_control','decision':'retain_for_review_only' if keep else 'closed_fixed_configuration_failed_strong_comparators','retain':keep,'metrics':result['metrics'],'baselines_reused':baselines,'required_Score_strictly_above':maxscore,'required_macF1_at_least':maxmacro,'deltas':{n:{key:result['metrics'][key]-b[key] for key in ['Score','micF1','macF1']} for n,b in baselines.items()},'freeze':recovery['freeze'],'metrics_file':c.ref(out/'metrics.json'),'validation_file':c.ref(out/'validation.json'),'plan':pr,'source':recovery['original_source'],'recovery_source':c.ref(__file__),'recovery_plan':c.ref(rp),'original_failure':recovery['original_failure'],'fit_state':c.ref(d/'fit-state.json'),'quality_scope':plan['policy']['historical_context']+' F1 labels were used for meta-training, not unlabeled or OOF meta-validation.','configurations':1,'fit_count':1,'new_quality_calls':1,'automatic_originaldev_or_official':False}
 for record in plan['refs'].values():c.checked(record)
 for name in ['prediction','scores','model','fit_state']:c.checked(frozen[name])
 c.checked(recovery['original_plan']);c.checked(recovery['freeze']);c.checked(recovery['source'])
 c.save(out/'results.json',summary);print(json.dumps({'status':summary['status'],'decision':summary['decision'],'metrics':summary['metrics'],'deltas':summary['deltas']}))
if __name__=='__main__':main()
