"""Aggregate development diagnosis of one already frozen prediction, no new predictions."""
import argparse,collections,datetime,hashlib,json,os,sys
from pathlib import Path

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def load(p):return {d['pmc_id']:d for d in map(json.loads,Path(p).read_text().splitlines())}
def put(p,j):
 with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--gold',required=True);ap.add_argument('--evaluator',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 assert ref(a.prediction)['sha256']=='05cfa90fdcba16371ddc6fb06aabdabfbcfd1cf631fc30101d98b3e40dd0d5ba'
 assert ref(a.evaluator)['sha256']=='ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
 O=Path(a.output);O.mkdir(exist_ok=False,parents=True)
 put(O/'intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction=ref(a.prediction),source=ref(__file__),purpose='Aggregate fixed errors by patient relation count, single/multiple patients, document concept availability. No patient IDs/text examples in summary, no new prediction or threshold.',new_v2=0,fit_calls=0,model_calls=0))
 assert ref(a.gold)['sha256']=='c1f0f24a58b62bb898eb672550dc58247ddb9a0e8768c6f899812cfc327cc050'
 sys.path.insert(0,str(Path(a.evaluator).parent));import evaluate as ev
 pred,gold=load(a.prediction),load(a.gold);assert pred.keys()==gold.keys() and len(pred)==16
 counts=collections.Counter();bins=collections.defaultdict(collections.Counter);patient_f1=[]
 for key,g in gold.items():
  p=pred[key];gs,ps=ev.associations(g),ev.associations(p);assert gs.keys()==ps.keys()
  gold_all=set().union(*gs.values());pred_concepts=ev.document_concepts(p,g,'all','text');gold_concepts=ev.document_concepts(g,g,'all','text')
  for pid,target in gs.items():
   proposed=ps[pid];tp=len(target&proposed);fp=len(proposed-target);fn=len(target-proposed);f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
   patient_f1.append(f1)
   size='0' if not target else '1-2' if len(target)<=2 else '3-5' if len(target)<=5 else '6-15' if len(target)<=15 else '16+'
   for group in ['gold_relations_'+size, 'single_patient_doc' if len(gs)==1 else 'multi_patient_doc']:
    bins[group].update(dict(patients=1,tp=tp,fp=fp,fn=fn,sum_patient_f1=f1,gold_relations=len(target),predicted_relations=len(proposed)))
   counts.update(dict(patients=1,tp=tp,fp=fp,fn=fn,zero_patient_f1=int(f1==0),perfect_patient_f1=int(f1==1),no_predicted_relations=int(not proposed)))
   for label in target-proposed:
    kind='hpo' if ev.re.fullmatch(r'HP:[0-9]{7}',label) else 'unlinked_text'
    counts['FN_Task1_document_supported' if (kind,label) in pred_concepts else 'FN_Task1_document_missing']+=1
   for label in proposed-target:
    if label in gold_all:counts['FP_other_gold_patient_same_concept']+=1
    else:
     kind='hpo' if ev.re.fullmatch(r'HP:[0-9]{7}',label) else 'unlinked_text'
     counts['FP_no_gold_relation_but_gold_entity' if (kind,label) in gold_concepts else 'FP_no_gold_relation_or_entity']+=1
 for b in bins.values():b['patient_macro_f1']=b.pop('sum_patient_f1')/b['patients']
 assert counts['tp']==324 and counts['fp']==96 and counts['fn']==129 and counts['patients']==36
 assert counts['FN_Task1_document_supported']+counts['FN_Task1_document_missing']==129
 assert sum(counts[k] for k in ['FP_other_gold_patient_same_concept','FP_no_gold_relation_but_gold_entity','FP_no_gold_relation_or_entity'])==96
 put(O/'result.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction=ref(a.prediction),gold=ref(a.gold),evaluator=ref(a.evaluator),source=ref(__file__),counts=dict(counts),groups=dict(bins),relation_macro=sum(patient_f1)/36,new_v2=0,new_predictions=0,fit_calls=0,model_calls=0,development_gold_diagnosis=True,causality='Descriptive partitions only; gold-entity support does not prove predicted mention boundaries, no per-patient rule or label-derived resource created.'))
 print(json.dumps(dict(counts=counts,groups=bins,relation_macro=sum(patient_f1)/36)))
if __name__=='__main__':main()
