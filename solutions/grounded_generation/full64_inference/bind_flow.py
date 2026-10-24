"""Freeze parser, normalization, and pure relation supply; no quality stage."""
import argparse,json
from pathlib import Path
from actual import H
S=Path(__file__).resolve().parent;BASE=S.parent;N=BASE/'normalization'
R=Path('/workspace/patientphex-runs/round2-20261023');B=R/'research/grounded-generation-full64-v1'
GOLD={'path':'/workspace/patientphex-data/day1/dev16.gold.jsonl','scope':'deny-open/hash only; no gold bytes accessed'}
INPUT=Path('/workspace/patientphex-data/day1/dev16.input.jsonl')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extraction','normalization-finalize','pure']);a=ap.parse_args();guard=H.gold_guard({'gold':GOLD})
 if a.stage=='extraction':
  d=B/'infer-dev36-v1';c=H.read(d/'collection.json');assert c['inference_reported_success'] and c['all_actual_SHA_verified']
  for item in c['files'].values():H.checked(item)
  p=H.read(S/'infer-plan.json');assert H.read(d/'package.json')['plan']==H.ref(S/'infer-plan.json')
  inputs={k:H.ref(d/'terminal'/name) for k,name in [('complete','output/complete.json'),('wrapper','wrapper-complete.json'),('responses','output/responses.jsonl')]}
  inputs.update(dev_input=H.ref(INPUT),prompts=p['payload_sources']['held-prompts.jsonl'],collection=H.ref(d/'collection.json'))
  sources=[S/'extract_quotes.py',BASE/'inference/decode.py',BASE/'schema.py',S/'run.py']
  v=dict(schema='grounded-extract-complete36-v1',sources={q.name:H.ref(q) for q in sources},inputs=inputs,gold_metadata_only=GOLD,
   remote_output=p['output_dir'],local_output=str(d/'terminal/output'),output_dir=str(B/'dev36-quotes-v1'),binding_source=H.ref(__file__),original_collection=H.ref(d/'collection.json'));target=S/'extraction-plan.json'
 elif a.stage=='normalization-finalize':
  prep=B/'dev36-normalization-preparation/manifest.json';p=H.read(prep);assert p['patients']==36 and p['documents']==16
  d=B/'normalization-dev36-v1'
  if p['necessary_public_query_count']:
   c=H.read(d/'collection.json');assert c['all_actual_SHA_verified'] and c['query_reported_success'];x=dict(query_collection=H.ref(d/'collection.json'),query_scores=H.ref(d/'terminal/output/query-scores.json'),query_complete=H.ref(d/'terminal/output/complete.json'))
  else:x=dict(query_collection=None,query_scores=None,query_complete=None)
  sources=[S/'held_prepare.py',S/'held_finalize.py']+[N/n for n in ['strict.py','semantic.py','public_bank.py','policy_v1.json']]
  v=dict(schema='grounded-held-normalization-finalize-v1',preparation=H.ref(prep),sources={q.name:H.ref(q) for q in sources},output_dir=str(B/'dev36-normalized-v1'),binding_source=H.ref(__file__),**x);target=S/'held-finalize-plan.json'
 else:
  ex=H.read(B/'dev36-quotes-v1/manifest.json');norm=H.read(B/'dev36-normalized-v1/manifest.json')
  # The scaffold is only a carrier for complete docs/patients/entities. All its
  # associations are replaced by the complete new-only keys, not inherited.
  old=H.read(BASE/'f0/quality-plan-template.json')['references']
  refs={k:old[k] for k in ['hpo','byte_helper','validator']}
  refs.update(baseline=H.ref(R/'research/support-plus-patient-six-delta/prediction.jsonl'),dev_input=H.ref(INPUT),normalization_policy=H.ref(N/'policy_v1.json'))
  assert refs['baseline']['sha256']=='9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67'
  refs.update({name:H.ref(path) for name,path in [('strict',N/'strict.py'),('semantic',N/'semantic.py'),('public_bank',N/'public_bank.py'),('held_prepare',S/'held_prepare.py'),('held_finalize',S/'held_finalize.py'),('extract_quotes',S/'extract_quotes.py'),('decode',BASE/'inference/decode.py'),('schema',BASE/'schema.py'),('actual_chain_source',S/'actual.py')]})
  actual=dict(collection=H.ref(B/'infer-dev36-v1/collection.json'),extraction_plan=H.ref(S/'extraction-plan.json'),extraction_manifest=H.ref(B/'dev36-quotes-v1/manifest.json'),quotes=ex['quotes'],decisions=ex['decisions'],normalization_manifest=H.ref(B/'dev36-normalized-v1/manifest.json'),normalizations=norm['normalizations'],semantic_evidence=norm['semantic_evidence'])
  v=dict(schema='grounded-dev36-complete-pure-relation-supply-v1',source=H.ref(S/'build_pure.py'),gold=GOLD,references=refs,actual=actual,output_directory=str(B/'dev36-pure-relations-v1'));target=S/'pure-plan.json'
 assert not guard['enabled'];H.save(target,v);print(json.dumps(H.ref(target)))

if __name__=='__main__':main()
