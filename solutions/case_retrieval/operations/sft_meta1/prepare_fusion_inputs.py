#!/usr/bin/env python3
"""CPU-only matched LR and context inputs for single F1 meta training; no outer gold."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,copy,gzip,hashlib,json,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'rt') as f:return [json.loads(l) for l in f if l.strip()]
def ref(p):return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def key(r):return str(r['pmc_id']),str(r['patient_id']),r['concept']
def save(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rows(p,d):
 with Path(p).open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in d))
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory;o=b/'fusion-inputs';o.mkdir();sys.path.insert(0,str(b));import run as base
 import numpy as np,sklearn,joblib
 assert sha(b/'run.py')=='41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0'
 R=Path('/workspace/patientphex-runs/round2-20261023');Q=R/'case_retrieval/qwen35-preparation';D=Path('/workspace/patientphex-data');S=R/'sequence_crf';O=R/'ontology_rules';hpo=D/'hp.obo'
 specs=[{'name':'meta_F1','train':O/'nested-outer0-inner1/train32.jsonl','input':D/'round2-inner4/fold-1/validation16.input.jsonl','entities':S/'relation-sft-fold0/inner1/candidates.jsonl','prompts':S/'relation-sft-fold0/inner1/prompts.jsonl','features':Q/'strict-meta-calibration-run/meta-rows.jsonl','training':True}, {'name':'outer_F0','train':D/'round2-inner4/fold-0/train48.jsonl','input':D/'round2-inner4/fold-0/validation16.input.jsonl','entities':S/'crf-semantic-fold0/lr.pred.jsonl','prompts':S/'qwen-pair-fold0/prompts.jsonl','features':O/'aggregate-relation-fold0/input-rows.jsonl.gz','training':False}]
 # Only meta F1 labels are read; outer0 labels are not referenced anywhere.
 meta_goldfile=D/'round2-inner4/fold-1/validation16.gold.jsonl';meta_gold={str(d['pmc_id']):d for d in read(meta_goldfile)};reports=[]
 for spec in specs:
  started=time.monotonic();name=spec['name'];train=read(spec['train']);docs=read(spec['input']);by={str(d['pmc_id']):d for d in docs};trainids={str(d['pmc_id']) for d in train};assert not trainids&set(by) and len(by)==16;assert len(trainids)==(32 if spec['training'] else 48)
  assert all('entities' not in d and 'association' not in d for d in docs)
  if spec['training']:assert set(meta_gold)==set(by)
  entities={str(d['pmc_id']):d for d in read(spec['entities'])};prompts=read(spec['prompts']);features={key(r):r for r in read(spec['features'])};assert all('target' not in r for r in features.values());assert set(entities)==set(by)
  linker=base.Linker(train,str(hpo));vectorizer,model,stats=base.fit_relation(train,linker);joblib.dump({'vectorizer':vectorizer,'model':model},o/f'{name}.lr.joblib')
  probabilities={};predictions=[]
  for doc in docs:
   did=str(doc['pmc_id']);es=entities[did]['entities'];xs=[];mapping=[]
   for e in es:
    for patient in doc['patient']:xs.append(base.pair_features(doc,e,patient));mapping.append((e,str(patient['patient_id'])))
   probs=model.predict_proba(vectorizer.transform(xs))[:,1] if xs else []
   values={str(p['patient_id']):set() for p in doc['patient']}
   for (e,pid),prob in zip(mapping,probs):
    for concept in base.association_keys(e):
     k=(did,pid,concept);probabilities[k]=max(probabilities.get(k,0.),float(prob))
     if prob>=.55:values[pid].add(concept)
   pred=copy.deepcopy(entities[did]);pred['association']=[{'patient_id':p['patient_id'],'phenotype':sorted(values[str(p['patient_id'])])} for p in doc['patient']];predictions.append(pred)
  output=[];labels=[];seen=set()
  for prompt in prompts:
   did=str(prompt['pmc_id']);assert did in by and 'answer' not in prompt
   gold=base.gold_associations(meta_gold[did]) if spec['training'] else None;answer={}
   for cid,concept in prompt['code_map'].items():
    if gold is not None:answer[cid]=[pid for pid in prompt['patient_ids'] if concept in gold.get(pid,set())]
    for pid in prompt['patient_ids']:
     k=(did,pid,concept);assert k in probabilities and k in features and k not in seen;seen.add(k);feats=copy.deepcopy(features[k]['features']);assert 'relation_lr_probability' not in feats;feats.update(relation_lr_probability=probabilities[k],relation_lr_selected=int(probabilities[k]>=.55));row={'row_id':prompt['row_id'],'pmc_id':did,'patient_id':pid,'concept':concept,'CID':cid,'features':feats}
     if gold is not None:row['target']=int(pid in answer[cid])
     output.append(row)
   if gold is not None:labels.append(dict(prompt,answer=json.dumps(answer,ensure_ascii=False,separators=(',',':'))))
  assert len(output)==(1557 if spec['training'] else 1166) and set(features)>=seen;rows(o/f'{name}.rows.jsonl',output);rows(o/f'{name}.lr.pred.jsonl',predictions)
  if spec['training']:
   rows(o/'meta-training-labels.jsonl',labels)
   assert sha(S/'relation-sft-fold0/train.jsonl')=='eb133c3107c1aada2e156892ba430e5b5860315c058d6c7dd5387fe45b065f77';old=read(S/'relation-sft-fold0/train.jsonl');old={r['row_id']:r for r in old};assert all(row==old[row['row_id']] for row in labels)
  else:
   assert all({str(a['patient_id']):set(a['phenotype']) for a in pred['association']}=={str(a['patient_id']):set(a['phenotype']) for a in entities[str(pred['pmc_id'])]['association']} for pred in predictions),'Fresh CPU LR48 selection changed vs frozen fullLR'
  reports.append({'role':name,'sources':{k:ref(spec[k]) for k in ['train','input','entities','prompts','features']},'train_ids':sorted(trainids),'input_ids':sorted(by),'fit_stats':stats,'rows':len(output),'positive_labels':sum(r.get('target',0) for r in output) if spec['training'] else None,'lr_features':['relation_lr_probability','relation_lr_selected'],'aggregation':'max occurrence probability perpatient-concept; selected iffmax>=original.55','output_rows':ref(o/f'{name}.rows.jsonl'),'model':ref(o/f'{name}.lr.joblib'),'predictions':ref(o/f'{name}.lr.pred.jsonl'),'elapsed_seconds':time.monotonic()-started,'outer_LR_selection_exact_old_fullLR':not spec['training']})
 report={'status':'CPU_complete_matched_base_features_no_meta_tree_fit_no_outergold','source':ref(__file__),'base_source':ref(b/'run.py'),'hpo':ref(hpo),'meta_training_labels':ref(meta_goldfile),'labels':ref(o/'meta-training-labels.jsonl'),'numpy':np.__version__,'sklearn':sklearn.__version__,'roles':reports,'old299_tree_baseline_not_reused':'Both new arms share F1-only meta fitting and +LR2columns; feature vocabulary onlyfit F1. NewwithoutSFT baseline mandatory.'};save(o/'preparation.json',report);print(json.dumps({'roles':[(r['role'],r['rows'],r['positive_labels'],r['elapsed_seconds']) for r in reports]}))
if __name__=='__main__':main()
