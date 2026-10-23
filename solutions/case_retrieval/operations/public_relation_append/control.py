#!/usr/bin/env python3
"""Preserve old entities/relations, append only fixed public-new-concept decisions."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,subprocess,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def checked(r):assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes'],r['path']
def save(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def save_rows(p,x):
 with Path(p).open('x') as f:f.write(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in x))
def load(b):
 plan=read(b/'run-plan.json');assert sha(__file__)==plan['control_sha256']
 for x in plan['references'].values():checked(x)
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 h=module('public_append_tree',plan['references']['tree_helper']['path']);f=module('public_append_manifest',plan['references']['fusion_helper']['path']);h.verify_parser(plan['references']['parser']['path']);return plan,h,f

def command(argv,path):
 r=subprocess.run(argv,text=True,capture_output=True);save(path,dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr

def prepare(b):
 plan,h,f=load(b);r=plan['references'];oldprompts=rows(r['old_prompts']['path']);oldraw=rows(r['old_raw']['path']);newprompts=rows(r['prompts']['path'])
 f.verify_sft_response(h,read(r['old_manifest']['path']),oldraw,r['old_raw']['path'],r['old_prompts']['path'],plan['adapter_files'],read(r['model_source']['path']))
 oldpairs=h.pair_rows(oldprompts);newpairs=h.pair_rows(newprompts);assert len(oldpairs)==1557 and len(newpairs)==1251 and len(newprompts)==62
 oldkeys={(x['pmc_id'],x['patient_id'],x['concept']) for x in oldpairs};assert not oldkeys&{(x['pmc_id'],x['patient_id'],x['concept']) for x in newpairs}
 frozen=read(r['input_only_difference_freeze']['path']);assert frozen['statistics']['union']['new_concepts']==449 and frozen['artifacts']['union']['prompts']['sha256']==sha(r['prompts']['path'])
 pred=b/'baseline.pred.jsonl';command([plan['python'],r['parser']['path'],'--input',r['input']['path'],'--entities',r['old_entities']['path'],'--prompts',r['old_prompts']['path'],'--responses',r['old_raw']['path'],'--output',str(pred),'--fallback','baseline'],b/'baseline-merge-command.json')
 oldentity={str(x['pmc_id']):x for x in rows(r['old_entities']['path'])};baseline=rows(pred);assert len(baseline)==16 and sum(len(x['entities']) for x in baseline)==1189
 for x in baseline:assert x['entities']==oldentity[str(x['pmc_id'])]['entities']
 flags,_=h.llm_features(oldprompts,oldraw,oldpairs);expected={(x['pmc_id'],x['patient_id'],x['concept']) for x,v in zip(oldpairs,flags) if v[0]};actual={(str(d['pmc_id']),a['patient_id'],c) for d in baseline for a in d['association'] for c in a['phenotype']};assert expected==actual and len(actual)==383
 command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',str(pred),'--hpo',r['hpo']['path'],'--output',str(b/'baseline.validation.json')],b/'baseline-validation-command.json');assert read(b/'baseline.validation.json')['valid']
 result=dict(status='old_SFT32_baseline_prepared_input_only',plan=ref(b/'run-plan.json'),prediction=ref(pred),merge=ref(b/'baseline.pred.manifest.json'),validation=ref(b/'baseline.validation.json'),old_entities=1189,old_relations=383,newconcepts=449,newpairs=1251,no_new_gold_read=True);save(b/'baseline-prepared.json',result);return result

def freeze(b):
 plan,h,f=load(b);r=plan['references'];prior=read(b/'baseline-prepared.json');assert prior['plan']==ref(b/'run-plan.json');checked(prior['prediction'])
 frozen=read(b/'responses-freeze.json');assert frozen['plan_sha256']==sha(b/'run-plan.json') and frozen['raw_sha256']==sha(b/'responses.jsonl') and frozen['manifest_sha256']==sha(b/'responses.manifest.json')
 newp=rows(r['prompts']['path']);raw=rows(b/'responses.jsonl');meta=read(b/'responses.manifest.json');f.verify_sft_response(h,meta,raw,b/'responses.jsonl',r['prompts']['path'],plan['adapter_files'],read(r['model_source']['path']))
 pairs=h.pair_rows(newp);flags,stats=h.llm_features(newp,raw,pairs);addition={(x['pmc_id'],x['patient_id'],x['concept']) for x,v in zip(pairs,flags) if v[0]};baseline=rows(prior['prediction']['path']);augmented=copy.deepcopy(baseline);before={(str(x['pmc_id']),a['patient_id'],v) for x in baseline for a in x['association'] for v in a['phenotype']};assert not before&addition
 for old,new in zip(baseline,augmented):
  did=str(old['pmc_id'])
  for oa,na in zip(old['association'],new['association']):
   extra=sorted(c for doc,pid,c in addition if doc==did and pid==oa['patient_id']);na['phenotype'].extend(extra);assert na['phenotype'][:len(oa['phenotype'])]==oa['phenotype'] and len(na['phenotype'])==len(set(na['phenotype']))
  assert new['entities']==old['entities'] and {k:v for k,v in new.items() if k!='association'}=={k:v for k,v in old.items() if k!='association'}
 after={(str(x['pmc_id']),a['patient_id'],v) for x in augmented for a in x['association'] for v in a['phenotype']};assert after==before|addition
 save_rows(b/'augmented.pred.jsonl',augmented);save(b/'addition-keys.json',sorted(addition))
 result=dict(status='both_full_predictions_frozen',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(b/'run-plan.json'),baseline=prior['prediction'],augmented=ref(b/'augmented.pred.jsonl'),raw_freeze=ref(b/'responses-freeze.json'),new_associations=ref(b/'addition-keys.json'),new_association_count=len(addition),old_associations=383,entities_each=1189,old_all_preserved=True,parse=stats,evaluation_performed=False,no_new_control_gold_read=True)
 save(b/'paired-predictions-freeze.json',result);return result

def evaluate(b):
 plan,h,f=load(b);r=plan['references'];frozen=read(b/'paired-predictions-freeze.json');assert frozen['status']=='both_full_predictions_frozen' and frozen['plan']==ref(b/'run-plan.json') and not frozen['evaluation_performed']
 for key in ['baseline','augmented','raw_freeze','new_associations']:checked(frozen[key])
 save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=ref(b/'paired-predictions-freeze.json'),scope='Already examined F1 training-meta16, not blind validation; two fixed new arms once'))
 assert sha(plan['gold_path'])==plan['historical_F1_gold_sha256'];metrics={}
 for arm in ['baseline','augmented']:
  pred=frozen[arm]['path'];validation=b/(arm+'.final.validation.json');mp=b/(arm+'.metrics.json')
  command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',pred,'--hpo',r['hpo']['path'],'--output',str(validation)],b/(arm+'.final.validation-command.json'));assert read(validation)['valid']
  command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',pred,'--body-scope','all','--no-id-doc','text','--output',str(mp)],b/(arm+'.evaluation-command.json'));value=read(mp);assert value['documents']==16 and value['patients']==39
  metrics[arm]=dict(metrics=value['metrics'],counts=value['counts'],prediction=frozen[arm],metrics_file=ref(mp),format=ref(validation));checked(frozen[arm])
 assert all(metrics['baseline']['metrics'][x]==metrics['augmented']['metrics'][x] for x in ['menF1','docF1'])
 # Public v2's literal raw gold convention. Canonical oracle diagnosis was separate.
 gold=set()
 compound=h.re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
 for doc in rows(plan['gold_path']):
  for a in doc['association']:
   for value in a['phenotype']:
    for concept in value.split(';') if compound.fullmatch(value) else [value]:gold.add((str(doc['pmc_id']),a['patient_id'],concept))
 extra={tuple(x) for x in read(b/'addition-keys.json')};tp=extra&gold;fp=extra-gold
 assert metrics['augmented']['counts']['relation']['tp']-metrics['baseline']['counts']['relation']['tp']==len(tp)
 assert metrics['augmented']['counts']['relation']['fp']-metrics['baseline']['counts']['relation']['fp']==len(fp)
 result=dict(status='complete_fixed_public_new_relations_F1_control',scope='StrictSFT32->F1, same1189entities and old383associations; F1 alreadyused for training-side diagnostics, not blind/originaldev/outer0',arms=metrics,delta={k:metrics['augmented']['metrics'][k]-metrics['baseline']['metrics'][k] for k in metrics['baseline']['metrics']},new_relations=dict(total=len(extra),tp=len(tp),fp=len(fp),fn_reduced=len(tp),fp_concept_in_other_supplied_patient=sum(any(g[0]==k[0] and g[2]==k[2] for g in gold) for k in fp),fp_concept_in_no_supplied_patient=sum(not any(g[0]==k[0] and g[2]==k[2] for g in gold) for k in fp)),freeze=ref(b/'paired-predictions-freeze.json'),plan=ref(b/'run-plan.json'),entities_fixed=True,old_relationships_and_text_preserved=True,no_training=True,no_originaldev_or_outer0=True,official_submission=False)
 save(b/'paired-summary.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','freeze','evaluate']);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();result=globals()[a.phase](a.directory.resolve());print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
