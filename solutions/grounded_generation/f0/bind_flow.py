"""Bind actual immutable F0 outputs to fixed parser, normalizer and one quality arm."""
import argparse,json
from pathlib import Path
from actual import H
S=Path(__file__).resolve().parent;BASE=S.parent;N=BASE/'normalization'
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
GOLD=dict(path='/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl',bytes=507921,sha256='630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extraction','normalization-finalize','pure','quality']);a=ap.parse_args();guard=H.gold_guard({'gold':GOLD})
 if a.stage=='extraction':
  d=B/'infer-F0-v1';c=H.read(d/'collection.json');assert c['inference_reported_success'] and c['all_actual_SHA_verified']
  for x in c['files'].values():H.checked(x)
  p=H.read(S/'infer-plan.json');assert H.read(d/'package.json')['plan']==H.ref(S/'infer-plan.json')
  inputs={k:H.ref(d/'terminal'/name) for k,name in [('complete','output/complete.json'),('wrapper','wrapper-complete.json'),('responses','output/responses.jsonl')]}
  inputs.update(F0_input=H.ref('/workspace/patientphex-data/round2-inner4/fold-0/validation16.input.jsonl'),prompts=p['payload_sources']['held-prompts.jsonl'],collection=H.ref(d/'collection.json'))
  sources=[S/'extract_quotes.py',BASE/'inference/decode.py',BASE/'schema.py',S/'run.py']
  v=dict(schema='grounded-extract-complete47-v1',sources={q.name:H.ref(q) for q in sources},inputs=inputs,gold_metadata_only=GOLD,remote_output=p['output_dir'],local_output=str(d/'terminal/output'),output_dir=str(B/'F0-quotes-v1'),binding_source=H.ref(__file__),original_collection=H.ref(d/'collection.json'));target=S/'extraction-plan.json'
 elif a.stage=='normalization-finalize':
  prep=B/'F0-normalization-preparation/manifest.json';p=H.read(prep);assert p['patients']==47 and p['documents']==16
  d=B/'normalization-held-F0-v1'
  if p['necessary_public_query_count']:
   c=H.read(d/'collection.json');assert c['all_actual_SHA_verified'] and c['query_reported_success'];x=dict(query_collection=H.ref(d/'collection.json'),query_scores=H.ref(d/'terminal/output/query-scores.json'),query_complete=H.ref(d/'terminal/output/complete.json'))
  else:x=dict(query_collection=None,query_scores=None,query_complete=None)
  sources=[S/'held_prepare.py',S/'held_finalize.py']+[N/n for n in ['strict.py','semantic.py','public_bank.py','policy_v1.json']]
  v=dict(schema='grounded-held-normalization-finalize-v1',preparation=H.ref(prep),sources={q.name:H.ref(q) for q in sources},output_dir=str(B/'F0-normalized-v1'),binding_source=H.ref(__file__),**x);target=S/'held-finalize-plan.json'
 elif a.stage=='pure':
  ex=H.read(B/'F0-quotes-v1/manifest.json');norm=H.read(B/'F0-normalized-v1/manifest.json');template=H.read(S/'quality-plan-template.json');r=template['references']
  r['F0_input']=r.pop('input');r['normalization_policy']=H.ref(N/'policy_v1.json')
  r.update({name:H.ref(path) for name,path in [('strict',N/'strict.py'),('semantic',N/'semantic.py'),('public_bank',N/'public_bank.py'),('held_prepare',S/'held_prepare.py'),('held_finalize',S/'held_finalize.py'),('extract_quotes',S/'extract_quotes.py'),('decode',BASE/'inference/decode.py'),('schema',BASE/'schema.py'),('actual_chain_source',S/'actual.py')]})
  actual=dict(collection=H.ref(B/'infer-F0-v1/collection.json'),extraction_plan=H.ref(S/'extraction-plan.json'),extraction_manifest=H.ref(B/'F0-quotes-v1/manifest.json'),quotes=ex['quotes'],decisions=ex['decisions'],normalization_manifest=H.ref(B/'F0-normalized-v1/manifest.json'),normalizations=norm['normalizations'],semantic_evidence=norm['semantic_evidence'])
  v=dict(schema='grounded-F0-complete-pure-relation-supply-v1',source=H.ref(S/'build_pure.py'),gold=GOLD,references=r,actual=actual,output_directory=str(B/'F0-pure-relations-v1'));target=S/'pure-plan.json'
 else:
  v=H.read(S/'quality-plan-template.json');assert v.pop('actual_grounded_references_pending') is True and v['baseline_score']==.7115075800266846
  freeze=B/'F0-pure-relations-v1/freeze.json';p=H.read(freeze);assert p['all47_complete'] and p['no_gold_no_score'] and p['baseline_relationships_excluded']
  v['references'].update(grounded_freeze=H.ref(freeze),grounded_prediction=p['predictions']['new-only']);v['bound_from_template']=H.ref(S/'quality-plan-template.json');target=S/'quality-plan.json'
 assert not guard['enabled'];H.save(target,v);print(json.dumps(H.ref(target)))
if __name__=='__main__':main()
