"""Fixed joint score gate followed by frozen strict/public Sap normalization."""
from pathlib import Path
from collections import Counter
import argparse,json,time,math
from common import ref,read,save,unpack,packed
from vendor.semantic import GroundedNormalizer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--native',required=True);ap.add_argument('--encoded',required=True);ap.add_argument('--input',required=True);ap.add_argument('--train',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--bank',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();D=Path(a.native);O=Path(a.output);O.mkdir(parents=True,exist_ok=False);start=time.monotonic();complete=json.loads((D/'complete.json').read_text());assert complete['status']=='COMPLETE_FRESH8EPOCH_FULL_F1_RAW' and complete['global_step']==256 and complete['all_documents']==16 and complete['all_patients']==39
 inputs={str(d['pmc_id']):d for d in read(a.input)};rows=unpack(Path(a.encoded)/'held.json.gz');assert set(inputs)=={r['pmc_id'] for r in rows};quotes=set();selected=[];rawrefs=[]
 for row in rows:
  rawpath=D/('raw-'+row['pmc_id']+'.json.gz');raw=unpack(rawpath);item=next(p for p in complete['predictions'] if p['pmc_id']==row['pmc_id']);assert ref(rawpath)['sha256']==item['compressed']['sha256'];rawrefs.append(ref(rawpath));assert raw['pmc_id']==row['pmc_id'] and raw['patient_ids']==[p['patient_id'] for p in row['patients']];assert len(raw['records'])==item['records'];assert raw['all_candidate_count']==len(row['candidates']) and raw['global_logit_floor']==-2. and raw['patient_logit_floor'] is None;seen=set()
  for r in raw['records']:
   ci=r['candidate_index'];assert ci not in seen;seen.add(ci);assert -2.<=r['NER_logit'] and math.isfinite(r['NER_logit']);assert len(r['patient_logits'])==len(row['patients']) and all(math.isfinite(x) for x in r['patient_logits'])
   ps=[p['patient_id'] for p,s in zip(row['patients'],r['patient_logits']) if s>0]
   if r['NER_logit']<=0 or not ps:continue
   off,length,pi,a0,b0=row['coordinates'][ci];quote=inputs[row['pmc_id']]['full_text'][pi]['text'][a0:b0];assert len(quote)==length and int(inputs[row['pmc_id']]['full_text'][pi]['offset'])+a0==off;quotes.add(quote);selected.append({'pmc_id':row['pmc_id'],'candidate_index':ci,'offset':off,'length':length,'quote':quote,'selected_patient_ids':ps,'NER_logit':r['NER_logit'],'patient_logits':r['patient_logits']})
 save(O/'pre-normalization.json',{'full16_raw':rawrefs,'selected_spans':len(selected),'unique_selected_quotes':len(quotes),'joint_gate':{'NER_gt':0.,'patient_gt':0.},'threshold_search':False});normalizer=GroundedNormalizer(a.hpo,a.train,model_dir=a.model,bank_dir=a.bank,policy_path=Path(__file__).parent/'vendor/policy_v1.json',device='cuda:0');isolation=normalizer.strict.assert_disjoint_input(a.input);ordered=sorted(quotes);norms=normalizer.normalize_many(ordered);lookup=dict(zip(ordered,norms));pairs={pmc:{str(p['patient_id']):set() for p in doc['patient']} for pmc,doc in inputs.items()}
 for r in selected:
  norm=lookup[r['quote']];r['normalization_status']=norm['status'];r['normalization_reason']=norm['reason'];r['identifiers']=[] if norm['status']=='abstain' else ([r['quote']] if norm['status']=='nil' else norm['ids'])
  for pid in r['selected_patient_ids']:pairs[r['pmc_id']][pid].update(r['identifiers'])
 save(O/'normalizations.json',norms);packed(O/'selected-spans.json.gz',selected);save(O/'relations.json',{pmc:{pid:sorted(values) for pid,values in ps.items()} for pmc,ps in pairs.items()});save(O/'complete.json',{'status':'COMPLETE_FIXED_JOINT_NORMALIZATION_FULL_F1','documents':len(pairs),'patients':sum(map(len,pairs.values())),'selected_spans':len(selected),'unique_quotes':len(ordered),'mapped_pairs':sum(len(v) for ps in pairs.values() for v in ps.values()),'normalization_status':dict(Counter(n['status'] for n in norms)),'normalization_reason':dict(Counter(n['reason'] for n in norms)),'policy':ref(Path(__file__).parent/'vendor/policy_v1.json'),'public_bank_reencoded':False,'raw':rawrefs,'native_complete':ref(D/'complete.json'),'train':ref(a.train),'input':ref(a.input),'isolation':isolation,'normalizer_manifest':normalizer.strict.manifest,'encoder_manifest':normalizer.encoder.manifest if normalizer.encoder else None,'bank_files':{n:ref(Path(a.bank)/n) for n in ['public-bank.npy','public-labels.json','public-bank-manifest.json']},'outputs':{p.name:ref(p) for p in O.iterdir() if p.is_file()},'seconds':time.monotonic()-start,'no_quality':True})
if __name__=='__main__':main()
