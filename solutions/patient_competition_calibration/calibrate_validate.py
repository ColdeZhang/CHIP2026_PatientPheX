"""One F0 calibration fit, then locked F1 developmental validation; no NN."""
from pathlib import Path
from collections import defaultdict
import argparse,copy,datetime,hashlib,json,math,sys
from common import ref,read,save,unpack

def checked(r):
 p=Path(r['path']);assert ref(p)['sha256']==r['sha256'];return p

def evaluator(plan):
 checked(plan['quality_refs']['evaluator']);checked(plan['quality_refs']['validator']);sys.path.insert(0,str(Path(plan['quality_refs']['evaluator']['path']).parent));import evaluate as ev;import validate_predictions as va;assert ev.VERSION=='patientphex-public-approx-v2';return ev,va

def positive_support(inp,normdir,rawdir):
 """Complete max>0 support only. Never invent a nonpositive score/negative inf."""
 inputs={str(d['pmc_id']):d for d in inp};N=Path(normdir);R=Path(rawdir);complete=json.loads((N/'complete.json').read_text());selected=unpack(N/'selected-spans.json.gz');raw={p:unpack(R/('raw-'+p+'.json.gz')) for p in inputs};rawindex={p:{r['candidate_index']:r for r in d['records']} for p,d in raw.items()};expected={(p,r['candidate_index']) for p,d in raw.items() for r in d['records'] if r['NER_logit']>0 and any(x>0 for x in r['patient_logits'])};assert len(expected)==len(selected);seen=set();entries={}
 for s in selected:
  p=s['pmc_id'];key=(p,s['candidate_index']);assert key not in seen;seen.add(key);r=rawindex[p][s['candidate_index']];assert s['NER_logit']==r['NER_logit'] and s['patient_logits']==r['patient_logits'];pids=[str(x['patient_id']) for x in inputs[p]['patient']];assert raw[p]['patient_ids']==pids and len(pids)==len(r['patient_logits']);assert s['selected_patient_ids']==[pid for pid,z in zip(pids,r['patient_logits']) if z>0];assert all(math.isfinite(z) for z in r['patient_logits']);assert s['normalization_status'] in ['abstain','nil','linked'];assert (not s['identifiers'])== (s['normalization_status']=='abstain')
  for pid,z in zip(pids,r['patient_logits']):
   if z<=0:continue
   for literal in s['identifiers']:
    k=(p,pid,literal);source={'candidate_index':s['candidate_index'],'offset':s['offset'],'length':s['length'],'NER_logit':r['NER_logit'],'patient_logit':z,'normalization_status':s['normalization_status'],'normalization_reason':s['normalization_reason']}
    if k not in entries:entries[k]={'key':list(k),'max_logit':z,'positive_sources':[]}
    entries[k]['max_logit']=max(entries[k]['max_logit'],z);entries[k]['positive_sources'].append(source)
 assert seen==expected;assert len(entries)==complete['mapped_pairs'];return [entries[k] for k in sorted(entries)]

def select(table,threshold):
 t=math.inf if threshold=='inf' else float(threshold);assert t>=0
 return {tuple(r['key']) for r in table if r['max_logit']>t}

def relation_metrics(gold,guess,ev):
 total=[0,0,0];f=[]
 for d in gold:
  truth=ev.associations(d);p=str(d['pmc_id']);ids={str(x['patient_id']) for x in d['patient']};assert set(truth)==ids
  for pid in sorted(ids):
   predicted={c for pp,pi,c in guess if pp==p and pi==pid};counts=ev.compare_sets(truth[pid],predicted);total=[a+b for a,b in zip(total,counts)];f.append(ev.prf(*counts)['f1'])
 assert all(p in {str(d['pmc_id']) for d in gold} and pid in {str(x['patient_id']) for d in gold if str(d['pmc_id'])==p for x in d['patient']} for p,pid,c in guess)
 micro=ev.prf(*total)['f1'];macro=sum(f)/len(f);return {'task2_micF1':micro,'task2_macF1':macro,'objective':(micro+macro)/2,'tp':total[0],'fp':total[1],'fn':total[2],'patients':len(f),'relations':len(guess)}

def calibrate(plan,planpath,root,out):
 O=Path(out);O.mkdir(parents=True,exist_ok=False);ev,va=evaluator(plan);D=Path(root);native=D/'collected/run/native';norm=D/'collected/run/normalization';assert json.loads((norm/'complete.json').read_text())['status']=='COMPLETE_FIXED_JOINT_NORMALIZATION_FULL_F0';inp=read(checked(plan['f0_input']));table=positive_support(inp,norm,native);save(O/'positive-relation-support.json',{'scope':'complete positive max>0 support only; all nonpositive full-candidate maxima unknown','records':table,'input':plan['f0_input'],'normalization':ref(norm/'complete.json'),'selected':ref(norm/'selected-spans.json.gz')})
 save(O/'calibration-intent.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':ref(planpath),'source':ref(__file__),'grid':plan['grid'],'objective':plan['calibration'],'table':ref(O/'positive-relation-support.json'),'gold_path':plan['f0_gold']['path'],'no_B0_CID_or_other_learned_candidates':True,'F0_is_calibration_training_not_independent_validation':True,'complete_prediction_files_generated':0,'v2_calls':0})
 gold=va.load_jsonl(plan['f0_gold']['path']);assert {str(d['pmc_id']) for d in gold}=={str(d['pmc_id']) for d in inp};assert all(d['patient']==next(x['patient'] for x in inp if str(x['pmc_id'])==str(d['pmc_id'])) for d in gold);rows=[]
 for t in plan['grid']:rows.append({'threshold':t,**relation_metrics(gold,select(table,t),ev)})
 best=max(rows,key=lambda r:(r['objective'],math.inf if r['threshold']=='inf' else r['threshold']));save(O/'locked-threshold.json',{'status':'F0_THRESHOLD_FROZEN_BEFORE_F1_DECODE','threshold':best['threshold'],'grid_results':rows,'selected':best,'gold':ref(plan['f0_gold']['path']),'positive_support':ref(O/'positive-relation-support.json'),'intent':ref(O/'calibration-intent.json'),'plan':ref(planpath),'training_objective':'mean of frozen-v2 relation micro/macro, all16/47; tie higher t','v2_calls':0,'new_full_prediction_files':0,'not_F0_validation':True});print(json.dumps({'calibration_selected':best}),flush=True)

def assemble(plan,planpath,lockpath,out):
 O=Path(out);O.mkdir(parents=True,exist_ok=False);lock=json.loads(Path(lockpath).read_text());assert lock['status']=='F0_THRESHOLD_FROZEN_BEFORE_F1_DECODE' and lock['plan']==ref(planpath);t=lock['threshold'];assert t in plan['grid'];inp=read(checked(plan['f1_input']));table=positive_support(inp,plan['old_f1_normalization'],plan['old_f1_raw']);new=select(table,t);base=read(checked(plan['baseline']));assert {str(d['pmc_id']) for d in base}=={str(d['pmc_id']) for d in inp};ev,va=evaluator(plan);bp={(str(d['pmc_id']),pid,c) for d in base for pid,cs in ev.associations(d).items() for c in cs};assert len(bp)==524 and sum(len(d['entities']) for d in base)==1189;endpoints={};formats={};
 for name,target in [('primary',bp|new),('diagnostic',new)]:
  rows=copy.deepcopy(base)
  for d in rows:
   for a in d['association']:a['phenotype']=sorted(c for p,pid,c in target if p==str(d['pmc_id']) and pid==str(a['patient_id']))
  assert all({k:v for k,v in d.items() if k!='association'}=={k:v for k,v in old.items() if k!='association'} for d,old in zip(rows,base));assert {(str(d['pmc_id']),pid,c) for d in rows for pid,cs in ev.associations(d).items() for c in cs}==target
  f=O/(name+'.prediction.jsonl')
  if name=='primary' and target==bp:f.write_bytes(checked(plan['baseline']).read_bytes())
  else:f.write_text(''.join(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n' for d in rows))
  endpoints[name]=ref(f);formats[name]=va.validate_predictions(inp,read(f),va.read_hpo(checked(plan['quality_refs']['hpo'])));assert formats[name]['valid']
 save(O/'positive-relation-support.json',{'scope':'complete positive max>0 support only','records':table,'selected_count':len(new)});save(O/'formats.json',formats);save(O/'joint-freeze.json',{'status':'BOTH_COMPLETE_FIXED_F1_ENDPOINTS_FROZEN','plan':ref(planpath),'threshold':ref(lockpath),'t':t,'endpoints':endpoints,'baseline':plan['baseline'],'formats':ref(O/'formats.json'),'source':ref(__file__),'selected_relations':len(new),'new_beyond_B4':len(new-bp),'primary_byteexact_B4':endpoints['primary']['sha256']==plan['baseline']['sha256'],'all524_B4_preserved':True,'entities':1189,'documents':16,'patients':39,'all_nonassociation_objects_preserved':True,'F1_development_not_blind':True});print(json.dumps({'threshold':t,'selected':len(new),'new_beyond_B4':len(new-bp)}),flush=True)

def quality(plan,planpath,freeze_path,out):
 O=Path(out);O.mkdir(parents=True,exist_ok=False);f=json.loads(Path(freeze_path).read_text());assert f['status']=='BOTH_COMPLETE_FIXED_F1_ENDPOINTS_FROZEN' and f['plan']==ref(planpath);ev,va=evaluator(plan);old=json.loads(checked(plan['baseline_metrics']).read_text())['metrics'];assert old['Score']==plan['baseline_score'];preds={k:read(checked(v)) for k,v in f['endpoints'].items()};save(O/'evaluation-intent.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze':ref(freeze_path),'new_v2':1 if f['primary_byteexact_B4'] else 2,'old_v2':0,'single_threshold_F1':True,'source':ref(__file__)});gold=va.load_jsonl(checked(plan['quality_refs']['gold']));results={};bp={(str(d['pmc_id']),pid,c) for d in read(checked(plan['baseline'])) for pid,cs in ev.associations(d).items() for c in cs};gp={(str(d['pmc_id']),pid,c) for d in gold for pid,cs in ev.associations(d).items() for c in cs}
 for name,rows in preds.items():
  inherited=name=='primary' and f['primary_byteexact_B4'];m=json.loads(checked(plan['baseline_metrics']).read_text()) if inherited else ev.evaluate_report(gold,rows,body_scope='all',no_id_doc='text');m['prediction_sha256']=f['endpoints'][name]['sha256'];m['gold_sha256']=plan['quality_refs']['gold']['sha256'];save(O/(name+'.metrics.json'),m);pairs={(str(d['pmc_id']),pid,c) for d in rows for pid,cs in ev.associations(d).items() for c in cs};added=pairs-bp;results[name]={'metrics':m['metrics'],'delta':{k:m['metrics'][k]-old[k] for k in old},'prediction':f['endpoints'][name],'inherited_old_no_v2':inherited,'relations':len(pairs),'added':len(added),'added_TP':len(added&gp),'added_FP':len(added-gp),'metrics_file':ref(O/(name+'.metrics.json'))}
 save(O/'result.json',{'status':'COMPLETE_F0_CALIBRATION_FIXED_F1_DEVELOPMENT','baseline':old,'threshold':f['t'],'endpoints':results,'retain':results['primary']['metrics']['Score']>old['Score'],'new_v2':1 if f['primary_byteexact_B4'] else 2,'old_v2':0,'originaldev':False,'official':False,'joint_freeze':ref(freeze_path)});print(json.dumps({k:v['metrics'] for k,v in results.items()}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['calibrate','assemble','quality']);ap.add_argument('--plan',required=True);ap.add_argument('--root');ap.add_argument('--lock');ap.add_argument('--freeze');ap.add_argument('--output',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text())
 if a.mode=='calibrate':calibrate(p,a.plan,a.root,a.output)
 elif a.mode=='assemble':assemble(p,a.plan,a.lock,a.output)
 else:quality(p,a.plan,a.freeze,a.output)
if __name__=='__main__':main()
