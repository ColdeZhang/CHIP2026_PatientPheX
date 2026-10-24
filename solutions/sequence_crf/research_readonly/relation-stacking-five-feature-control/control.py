"""One authorized five-original-scalar LR control: F1 train, frozen F0 endpoint.
No imports of sklearn/evaluator until their named stages; no automatic retries.
"""
import argparse,copy,hashlib,importlib.util,json,math,os,pickle,sys,time,warnings
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

FEATURES=['old9B_selected','reverse_same_adapter_selected','binary32_probability','oldLR_probability_max_occurrence','rules_selected_mean_occurrence']
PARAMETERS=dict(penalty='l2',C=1.0,solver='lbfgs',max_iter=1000,tol=1e-8,class_weight=None,fit_intercept=True,random_state=None)

def require(c,m):
 if not c:raise ValueError(m)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(r):
 p=Path(r['path']);require(p.is_absolute() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Artifact changed: '+str(p));return p
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write_rows(p,values):
 with Path(p).open('x') as f:
  for v in values:f.write(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
def key(row):
 require(all(type(row[k]) is str for k in ['pmc_id','patient_id','concept']),'Key string coercion forbidden')
 return row['pmc_id'],row['patient_id'],row['concept']
def keyed(values):
 result={key(x):x for x in values};require(len(result)==len(values),'Duplicate feature key');return result
def unit(v):
 require(type(v) in (int,float,bool) and math.isfinite(v) and 0<=v<=1,'Invalid original unit-valued feature');return float(v)
def vector(k,old,reverse,binary,continuous,rule):
 return [float(k in old),float(k in reverse),unit(binary[k]),unit(continuous[k]['features']['relation_lr_probability']),unit(rule[k]['features']['rule_selected_mean'])]
def relations(documents):
 result=set()
 for doc in documents:
  did=str(doc['pmc_id']);patients=set()
  for relation in doc['association']:
   pid=relation['patient_id'];require(type(pid)is str and pid not in patients,'Patient duplicate/nonstring');patients.add(pid)
   require(len(relation['phenotype'])==len(set(relation['phenotype'])),'Duplicate relation')
   for c in relation['phenotype']:require(type(c)is str,'Concept not string');result.add((did,pid,c))
 return result

def plan_check(plan,p):
 require(plan['execution_ready'] is True and plan['features']==FEATURES and plan['parameters']==PARAMETERS and plan['threshold']==.5 and plan['fit_predict_reload_cap_seconds']==60,'Frozen experiment changed')
 require(plan['source']['sha256']==sha(__file__),'Source changed')
 for r in plan['refs'].values():checked(r)
 return ref(p)

def association_span(line):
    """Locate the one top-level value without reserializing any other JSON bytes."""
    decoder, position, names, found = json.JSONDecoder(), 0, set(), None
    def whitespace(index):
        while index < len(line) and line[index].isspace():
            index += 1
        return index
    position = whitespace(position)
    require(line[position:position + 1] == '{', 'Require JSON object line')
    position += 1
    while True:
        position = whitespace(position)
        if line[position:position + 1] == '}':
            position += 1
            break
        key, position = decoder.raw_decode(line, position)
        require(isinstance(key, str) and key not in names, 'Duplicate/nonstring top-level key')
        names.add(key)
        position = whitespace(position)
        require(line[position:position + 1] == ':', 'Missing object colon')
        begin = whitespace(position + 1)
        value, end = decoder.raw_decode(line, begin)
        if key == 'association':
            require(isinstance(value, list), 'Association must be a list')
            found = (begin, end)
        position = whitespace(end)
        if line[position:position + 1] == ',':
            position += 1
            continue
        require(line[position:position + 1] == '}', 'Invalid object separator')
        position += 1
        break
    require(not line[position:].strip() and found is not None, 'Missing association/trailing JSON')
    return found

def nonrelation_bytes(lines):
    output = []
    for line in lines:
        begin, end = association_span(line)
        output.append((line[:begin] + '[]' + line[end:]).encode('utf-8'))
    return b''.join(output)


def prepare(plan,planref,d):
 refs=plan['refs'];meta=read(refs['base_preparation']['path']);inv=read(refs['prior_inventory']['path']);isolation=read(refs['isolation_review']['path'])
 require(all(isolation['checks'].values()) and inv['source_isolation']['F1_base_train32_excludes_F0_F1'] is True,'Prior source isolation is not complete')
 roles={r['role']:r for r in meta['roles']};m0,m1=roles['outer_F0'],roles['meta_F1'];F0=set(m0['input_ids']);F1=set(m1['input_ids']);train32=set(m1['train_ids']);train48=set(m0['train_ids'])
 require(len(F0)==len(F1)==16 and not F0&F1 and len(train32)==32 and not train32&(F0|F1) and train48==train32|F1 and len(train48)==48,'32→F1 /48→F0 metadata conflict')
 require(m1['positive_labels']==365 and m1['output_rows']==refs['F1_continuous'] and m0['output_rows']==refs['F0_continuous'],'Frozen mixed-table source/label count differs')
 report={'status':'prepared_whitelisted_five_features_F1_labels_read_F0_no_gold','plan':planref,'features':FEATURES,'folds':{},'source_isolation_reused':refs['isolation_review'],'base_ids':{'F1_train32':sorted(train32),'F1_meta_train16':sorted(F1),'F0_base_train48':sorted(train48),'F0_holdout16':sorted(F0)},'binary_both_folds_train32':True,'label_scope':'F1 target was opened and used for meta-training. This is neither unlabeled preparation nor OOF meta-validation. Original literal label365, not canonical366. F0 is repeatedly researched development, not blind.'}
 for fold,expected,docids in [('F1',1557,F1),('F0',1166,F0)]:
  prompts=rows(refs[fold+'_prompts']['path']);baseline=rows(refs[fold+'_old']['path']);reverse=rows(refs[fold+'_reverse']['path'])
  require([x['pmc_id'] for x in baseline]==[x['pmc_id'] for x in reverse] and {x['pmc_id'] for x in baseline}==docids,'Fold prediction identities differ')
  require(all({k:v for k,v in a.items() if k!='association'}=={k:v for k,v in b.items() if k!='association'} for a,b in zip(baseline,reverse)),'Reverse changed nonrelations')
  ordered=[];byrow={};patients={}
  for prompt in prompts:
   require(set(prompt)=={'row_id','pmc_id','code_map','patient_ids','messages'} and prompt['row_id'] not in byrow,'Unexpected labeled/duplicate prompt')
   byrow[prompt['row_id']]=prompt;did=prompt['pmc_id'];require(did in docids and (did not in patients or patients[did]==prompt['patient_ids']),'Patient list changed')
   patients[did]=prompt['patient_ids']
   require(all(type(x)is str for x in prompt['patient_ids']) and len(prompt['patient_ids'])==len(set(prompt['patient_ids'])),'Invalid patient strings')
   for cid,c in prompt['code_map'].items():
    for pid in prompt['patient_ids']:ordered.append({'row_id':prompt['row_id'],'CID':cid,'pmc_id':did,'patient_id':pid,'concept':c})
  universe={key(x) for x in ordered};require(len(ordered)==len(universe)==expected,'Fixed pair coverage differs')
  oldset,revset=relations(baseline),relations(reverse);require(oldset<=universe and revset<=universe,'Source selection outside frozen candidate universe')
  binary={};probrows=rows(refs[fold+'_binary']['path']);require([x['row_id'] for x in probrows]==[x['row_id'] for x in prompts],'Binary row order differs')
  for row in probrows:
   p=byrow[row['row_id']];require(row['pmc_id']==p['pmc_id'] and len(row['keys'])==len(row['probabilities']),'Binary source identity differs')
   for (cid,pid),prob in zip(row['keys'],row['probabilities']):
    require(cid in p['code_map'] and pid in p['patient_ids'],'Binary key out of pool');k=(p['pmc_id'],pid,p['code_map'][cid]);require(k not in binary,'Duplicate binary pair');binary[k]=unit(prob)
  # Authorized F1 mixed container is opened here; only TWO feature fields and top-level target are projected.
  cont=keyed(rows(refs[fold+'_continuous']['path']));rule=keyed(rows(refs[fold+'_rule']['path'])) if fold=='F1' else cont
  require(set(binary)==set(cont)==set(rule)==universe,'Source key omission/addition')
  output=[]
  for item in ordered:
   k=key(item);c=cont[k];require(c['row_id']==item['row_id'] and c['CID']==item['CID'],'Continuous-key metadata drift')
   require(unit(rule[k]['features']['rule_selected_mean'])==unit(c['features']['rule_selected_mean']),'Original rule mean changed in copied table')
   x=vector(k,oldset,revset,binary,cont,rule);row={k:v for k,v in item.items()};row['x']=x
   if fold=='F1':require(type(c['target'])is int and c['target'] in (0,1),'Target is not original binary int');row['y']=c['target']
   else:require(not {'target','y','labels','answer','gold'}&set(c),'F0 feature container contains labels')
   output.append(row)
  if fold=='F1':require(sum(x['y'] for x in output)==365,'Original literal F1 target count changed')
  target=d/(fold+'.five.jsonl');write_rows(target,output)
  report['folds'][fold]={'matrix_rows':ref(target),'rows':expected,'documents':16,'patients':sum(map(len,patients.values())),'positive_training_labels':sum(x['y'] for x in output) if fold=='F1' else None,'keys_sha256':hashlib.sha256(json.dumps([list(key(x)) for x in output],separators=(',',':')).encode()).hexdigest(),'column_min':[min(x['x'][i] for x in output) for i in range(5)],'column_max':[max(x['x'][i] for x in output) for i in range(5)],'no_added_context_interaction_or_scaling':True,'source_selection_counts':{'old':len(oldset),'reverse':len(revset)}}
 require(not F0&F1,'F0 entered meta-training')
 for r in refs.values():checked(r)
 save(d/'preparation.json',report)
 print(json.dumps({'status':report['status'],'F1_labels':365,'F1_rows':1557,'F0_rows':1166}))


def fit(plan,planref,d):
 import signal
 def timeout(signum,frame):raise TimeoutError('Single60second fit/predict/reload/freeze window expired')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60);started=time.monotonic()
 try:
  import numpy as np,sklearn
  from sklearn.linear_model import LogisticRegression
  from sklearn.exceptions import ConvergenceWarning
  prep=read(d/'preparation.json');require(prep['plan']==planref and prep['features']==FEATURES,'Prepared source/plan differs')
  training=rows(checked(prep['folds']['F1']['matrix_rows']));held=rows(checked(prep['folds']['F0']['matrix_rows']))
  X=np.asarray([x['x'] for x in training],dtype=np.float64);y=np.asarray([x['y'] for x in training],dtype=np.int64);T=np.asarray([x['x'] for x in held],dtype=np.float64)
  require(X.shape==(1557,5) and T.shape==(1166,5) and set(y.tolist())=={0,1} and int(y.sum())==365,'Matrix/labels shape differs')
  require(np.isfinite(X).all() and np.isfinite(T).all() and ((X>=0)&(X<=1)).all() and ((T>=0)&(T<=1)).all(),'Features were transformed or invalid')
  model=LogisticRegression(**PARAMETERS)
  with warnings.catch_warnings(record=True) as seen:
   warnings.simplefilter('always');t=time.monotonic();model.fit(X,y);fit_seconds=time.monotonic()-t
  require(not any(issubclass(w.category,ConvergenceWarning) for w in seen),'Single fixed model failed to converge; no retry')
  require(model.n_features_in_==5 and model.coef_.shape==(1,5) and model.intercept_.shape==(1,) and model.classes_.tolist()==[0,1],'Actual learned state differs')
  t=time.monotonic();prob=model.predict_proba(T)[:,1];predict_seconds=time.monotonic()-t
  require(np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),'Invalid F0 probabilities')
  with (d/'model.pkl').open('xb') as f:pickle.dump(model,f,protocol=5);f.flush();os.fsync(f.fileno())
  t=time.monotonic()
  with (d/'model.pkl').open('rb') as f:loaded=pickle.load(f)
  again=loaded.predict_proba(T)[:,1];require(np.array_equal(prob,again),'Actual saved-model reload differs');reload_seconds=time.monotonic()-t
  scores=[dict(row,probability=float(p),selected=bool(p>=.5)) for row,p in zip(held,prob)];write_rows(d/'F0.scores.jsonl',scores)
  baseline=Path(plan['refs']['F0_old']['path']).read_bytes().decode('utf-8').splitlines(keepends=True);docs=[json.loads(x) for x in baseline];rel=defaultdict(lambda:defaultdict(set))
  for row,p in zip(held,prob):
   if p>=.5:rel[row['pmc_id']][row['patient_id']].add(row['concept'])
  lines=[]
  for raw,doc in zip(baseline,docs):
   assoc=[]
   for old in doc['association']:
    require(type(old['patient_id'])is str,'Patient string identity changed');assoc.append({'patient_id':old['patient_id'],'phenotype':sorted(rel[doc['pmc_id']][old['patient_id']])})
   begin,end=association_span(raw);lines.append(raw[:begin]+json.dumps(assoc,ensure_ascii=False)+raw[end:])
  require(len(lines)==16 and nonrelation_bytes(lines)==nonrelation_bytes(baseline),'Incomplete F0 or nonrelation byte drift')
  with (d/'F0.pred.jsonl').open('xb') as f:f.write(''.join(lines).encode('utf-8'));f.flush();os.fsync(f.fileno())
  state={'status':'single_five_feature_LR_fit_F0_predict_actual_reload_complete','plan':planref,'source':ref(__file__),'preparation':ref(d/'preparation.json'),'model':ref(d/'model.pkl'),'scores':ref(d/'F0.scores.jsonl'),'prediction':ref(d/'F0.pred.jsonl'),'features':FEATURES,'parameters':model.get_params(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'learned_numeric_coefficients_including_intercept':6,'actual_n_iter':model.n_iter_.tolist(),'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,'actual_reload_predict_seconds':reload_seconds,'reload_probabilities_array_equal':True,'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in seen],'runtime':{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__},'F1_labels_read':True,'F0_gold_read':False,'F1_resubstitution_predictions_or_quality_generated':False,'selected_pairs':int((prob>=.5).sum()),'cap_seconds':60,'elapsed_seconds':time.monotonic()-started}
  save(d/'fit-state.json',state)
  frozen={'status':'all16_F0_predictions_complete_frozen_before_F0_gold','at_utc':now(),'plan':planref,'fit_state':ref(d/'fit-state.json'),'prediction':ref(d/'F0.pred.jsonl'),'scores':ref(d/'F0.scores.jsonl'),'model':ref(d/'model.pkl'),'F0_rows':1166,'F0_documents':16,'nonrelation_bytes_exact':True,'masked_nonrelation_sha256':hashlib.sha256(nonrelation_bytes(baseline)).hexdigest(),'F0_gold_read':False,'new_quality_calls':0}
  save(d/'F0-freeze.json',frozen);require(time.monotonic()-started<60,'Budget exceeded before freeze')
  print(json.dumps({'status':state['status'],'elapsed_seconds':time.monotonic()-started,'coef':state['coef'],'selected_pairs':state['selected_pairs']}))
 finally:signal.setitimer(signal.ITIMER_REAL,0)


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def evaluate(plan,planref,d):
 freeze=read(d/'F0-freeze.json');require(freeze['plan']==planref and freeze['status']=='all16_F0_predictions_complete_frozen_before_F0_gold','No complete immutable F0 freeze')
 for name in ['fit_state','prediction','scores','model']:checked(freeze[name])
 require(not (d/'evaluation-started.json').exists(),'Unique evaluation already reserved')
 save(d/'evaluation-started.json',{'at_utc':now(),'freeze':ref(d/'F0-freeze.json'),'new_v2_calls':1,'old_metrics_reused':True})
 # The first F0 gold-byte access is below, after complete prediction and freeze SHA checks.
 evaluator=module('five_feature_evaluator',plan['refs']['evaluator']['path']);validator=module('five_feature_validator',plan['refs']['validator']['path'])
 require(evaluator.VERSION=='patientphex-public-approx-v2','Frozen evaluator version differs')
 inputs=rows(plan['refs']['F0_input']['path']);pred=rows(freeze['prediction']['path']);hpo=validator.read_hpo(plan['refs']['hpo']['path']);valid=validator.validate_predictions(inputs,pred,hpo);save(d/'validation.json',valid);require(valid['valid'],'F0 prediction invalid')
 goldref=plan['gold_after_freeze'];require(sha(goldref['path'])==goldref['sha256'],'F0 gold version changed');gold=rows(goldref['path'])
 result=evaluator.evaluate_report(gold,pred,'all','text');save(d/'metrics.json',result)
 baselines={}
 for name,item in plan['comparators'].items():
  value=read(checked(item['ref']))
  for k in item['metric_path']:value=value[k]
  baselines[name]=value
 require(result['documents']==16 and result['patients']==47 and all(result['metrics'][k]==b[k] for b in baselines.values() for k in ['menF1','docF1']),'Fixed F0/entity metric scope changed')
 maxscore=max(x['Score'] for x in baselines.values());maxmacro=max(x['macF1'] for x in baselines.values());keep=result['metrics']['Score']>maxscore and result['metrics']['macF1']>=maxmacro
 summary={'status':'complete_unique_five_feature_F1_to_F0_control','decision':'retain_for_review_only' if keep else 'closed_fixed_configuration_failed_strong_comparators','retain':keep,'metrics':result['metrics'],'baselines_reused':baselines,'required_Score_strictly_above':maxscore,'required_macF1_at_least':maxmacro,'deltas':{n:{k:result['metrics'][k]-b[k] for k in ['Score','micF1','macF1']} for n,b in baselines.items()},'freeze':ref(d/'F0-freeze.json'),'metrics_file':ref(d/'metrics.json'),'validation_file':ref(d/'validation.json'),'plan':planref,'source':ref(__file__),'fit_state':ref(d/'fit-state.json'),'quality_scope':'F1 labels used for meta-training; F0 is previously studied development, not blind or OOF meta-validation. Existing303-column HGB and2-columnLR already tried; this is a fixed low-capacity5-source combination, not first stacking or a pure single-variable comparison.','configurations':1,'new_quality_calls':1,'automatic_originaldev_or_official':False}
 for r in plan['refs'].values():checked(r)
 for name in ['prediction','scores','model','fit_state']:checked(freeze[name])
 save(d/'results.json',summary);print(json.dumps({'status':summary['status'],'decision':summary['decision'],'metrics':summary['metrics'],'deltas':summary['deltas']}))


def selftest():
 k=('S','OII-1','literal');old={k};reverse=set();b={k:.3};c={k:{'features':{'relation_lr_probability':.4,'unused_leak':999},'target':0}};rule={k:{'features':{'rule_selected_mean':.2}}}
 a=vector(k,old,reverse,b,c,rule);c[k]['target']=1;c[k]['features']['unused_leak']=-999;require(vector(k,old,reverse,b,c,rule)==a==[1.,0.,.3,.4,.2],'Target/context entered whitelist')
 raw=' {"pmc_id":"S", "entities":[{"text":"é"}], "association" : [],"extra":1}\r\n';begin,end=association_span(raw);new=raw[:begin]+'[{"patient_id":"OII-1","phenotype":["literal"]}]'+raw[end:];require(nonrelation_bytes([raw])==nonrelation_bytes([new]),'Nonrelation bytes changed')
 return {'synthetic_target_feature_exclusion_and_nonrelation_bytes':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--stage',choices=['prepare','fit','evaluate']);ap.add_argument('--self-test',action='store_true');a=ap.parse_args();sys.dont_write_bytecode=True
 if a.self_test:print(json.dumps(selftest()));return
 require(a.plan and a.stage,'Plan and stage required');plan=read(a.plan);pr=plan_check(plan,a.plan);d=Path(plan['output_dir']);{'prepare':prepare,'fit':fit,'evaluate':evaluate}[a.stage](plan,pr,d);require(sha(a.plan)==pr['sha256'],'Plan changed')
if __name__=='__main__':main()
