"""Bind immutable completed outputs to the predeclared CPU extraction/quality stages.

This writes plans only; it never runs NN, parser, mapper, evaluator, or opens gold.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
S=Path(__file__).resolve().parent.parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
GOLD=dict(path='/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',bytes=525710,sha256='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3')

def ref(p):p=Path(p).resolve();v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
def checked(v):assert ref(v['path'])==v;return Path(v['path'])
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 d=p.parent
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent
 print(json.dumps(ref(p)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extraction','normalization-finalize','quality']);a=ap.parse_args()
 forbidden=os.path.realpath(GOLD['path'])
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==forbidden:raise PermissionError('Gold forbidden in plan binding')
 sys.addaudithook(audit)
 if a.stage=='extraction':
  d=B/'infer-F1-v1';collection=read(d/'collection.json');assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
  for x in collection['files'].values():checked(x)
  frozen=read(S/'inference/infer-plan.json');package=read(d/'package.json');assert package['plan']==ref(S/'inference/infer-plan.json')
  assert frozen['payload_sources']['run.py']==ref(S/'inference/run.py')
  inputs={k:ref(d/'terminal'/f) for k,f in [('complete','output/complete.json'),('wrapper','wrapper-complete.json'),('responses','output/responses.jsonl')]}
  inputs.update(F1_input=ref('/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl'),prompts=frozen['payload_sources']['held-prompts.jsonl'],collection=ref(d/'collection.json'))
  sourcepaths=[S/'inference/extract_quotes.py',S/'inference/decode.py',S/'schema.py',S/'inference/run.py']
  p=dict(schema='grounded-extract-complete39-v1',sources={x.name:ref(x) for x in sourcepaths},inputs=inputs,gold_metadata_only=GOLD,remote_output=frozen['output_dir'],local_output=str(d/'terminal/output'),output_dir=str(B/'F1-quotes-v1'),binding_source=ref(__file__),original_collection=ref(d/'collection.json'))
  save(S/'inference/extraction-plan.json',p)
 elif a.stage=='normalization-finalize':
  prep=B/'F1-normalization-preparation/manifest.json';v=read(prep);assert v['patients']==39 and v['documents']==16
  count=v['necessary_public_query_count'];query=B/'normalization-held-F1-v1';inputs={}
  if count:
   c=read(query/'collection.json');assert c['all_actual_SHA_verified'] and c['query_reported_success']
   inputs=dict(query_collection=ref(query/'collection.json'),query_scores=ref(query/'terminal/output/query-scores.json'),query_complete=ref(query/'terminal/output/complete.json'))
  else:inputs=dict(query_collection=None,query_scores=None,query_complete=None)
  p=dict(schema='grounded-held-normalization-finalize-v1',preparation=ref(prep),sources={n:ref(S/'normalization'/n) for n in ['held_finalize.py','held_prepare.py','strict.py','semantic.py','public_bank.py','policy_v1.json']},output_dir=str(B/'F1-normalized-v1'),binding_source=ref(__file__),**inputs)
  save(S/'normalization/held-finalize-plan.json',p)
 else:
  template=S/'quality/plan-template.json';p=read(template);assert p['state']=='TEMPLATE_UNBOUND_COMPLETE39' and p['actual'] is None and not p['quality_execution_authorized']
  for x in list(p['sources'].values())+list(p['references'].values()):checked(x)
  ex=read(B/'F1-quotes-v1/manifest.json');normal=read(B/'F1-normalized-v1/manifest.json');assert normal['extraction_manifest']==ref(B/'F1-quotes-v1/manifest.json') and normal['documents']==16 and normal['patients']==39
  p['actual']=dict(collection=ref(B/'infer-F1-v1/collection.json'),extraction_plan=ref(S/'inference/extraction-plan.json'),extraction_manifest=ref(B/'F1-quotes-v1/manifest.json'),quotes=ex['quotes'],decisions=ex['decisions'],normalization_manifest=ref(B/'F1-normalized-v1/manifest.json'),normalizations=normal['normalizations'],semantic_evidence=normal['semantic_evidence'])
  for x in p['actual'].values():checked(x)
  p.update(state='BOUND_COMPLETE39_AND_NORMALIZATIONS',quality_execution_authorized=True,authorization='Parent /root approved the two predeclared complete endpoints, highest Score strictly above N; primary hypothesis separate, no added arm',binding_source=ref(__file__),unbound_template=ref(template),output_directory=str(B/'quality-F1-v1'))
  save(S/'quality/plan.json',p)
if __name__=='__main__':main()
