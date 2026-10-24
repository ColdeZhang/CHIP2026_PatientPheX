"""Freeze both complete prewritten endpoints; no evaluator or held annotations."""
from pathlib import Path
import argparse,json,copy
from common import ref,read,save
ENDPOINTS=('primary-B-union-new','diagnostic-new-only')
def checked(r):
 assert ref(r['path'])==r,r['path'];return Path(r['path'])
def relation_set(docs):return {(str(d['pmc_id']),str(a['patient_id']),v) for d in docs for a in d['association'] for v in a['phenotype']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--normalization',required=True);ap.add_argument('--driver-result',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());D=Path(a.normalization);complete=json.loads((D/'complete.json').read_text());driver=json.loads(Path(a.driver_result).read_text());assert complete['status']=='COMPLETE_FIXED_JOINT_NORMALIZATION_FULL_F1' and complete['documents']==16 and complete['patients']==39 and not driver['timed_out'] and driver['returncode']==0;assert driver['normalization_complete_exists'];refs=plan['quality_refs'];inp=read(checked(refs['input']));base=read(checked(plan['baseline']));assert len(inp)==len(base)==16;relations=json.loads((D/'relations.json').read_text());assert ref(D/'relations.json')['sha256']==complete['outputs']['relations.json']['sha256'];assert {str(d['pmc_id']) for d in inp}==set(relations)=={str(d['pmc_id']) for d in base};O=Path(a.output);O.mkdir(parents=True,exist_ok=False);outputs={};counts={}
 for name in ENDPOINTS:
  docs=copy.deepcopy(base)
  for d in docs:
   pmc=str(d['pmc_id']);expected={str(p['patient_id']) for p in next(x for x in inp if str(x['pmc_id'])==pmc)['patient']};assert expected==set(relations[pmc])=={str(x['patient_id']) for x in d['association']};old=next(x for x in base if str(x['pmc_id'])==pmc)
   for r in d['association']:
    new=set(relations[pmc][str(r['patient_id'])]);r['phenotype']=sorted((set(r['phenotype'])|new) if name==ENDPOINTS[0] else new)
   assert {k:v for k,v in d.items() if k!='association'}=={k:v for k,v in old.items() if k!='association'}
   for r,oldr in zip(d['association'],old['association']):assert {k:v for k,v in r.items() if k!='phenotype'}=={k:v for k,v in oldr.items() if k!='phenotype'}
  if name==ENDPOINTS[0]:assert relation_set(base)<=relation_set(docs)
  p=O/(name+'.pred.jsonl');p.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs));outputs[name]=ref(p);counts[name]={'relations':len(relation_set(docs)),'added':len(relation_set(docs)-relation_set(base)),'removed':len(relation_set(base)-relation_set(docs))}
 assert len(relation_set(base))==471 and sum(len(d['entities']) for d in base)==1189;save(O/'joint-freeze.json',{'status':'BOTH_COMPLETE_PREDICTIONS_FROZEN_NO_QUALITY','documents':16,'patients':39,'entities_each':1189,'baseline_relations':471,'all_baseline_relations_preserved_primary':True,'all_nonphenotype_objects_preserved':True,'phenotype_lists_sorted_not_prefix_claim':True,'endpoints':outputs,'counts':counts,'plan':ref(a.plan),'normalization_complete':ref(D/'complete.json'),'driver_result':ref(a.driver_result),'normalization_relations':ref(D/'relations.json'),'source':ref(__file__),'gold_read':False})
if __name__=='__main__':main()
