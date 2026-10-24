#!/usr/bin/env python3
"""Create proposed train-only teacher inputs; text construction only, no model/tokenizer call."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.append('/usr/lib/python3/dist-packages')
from jinja2.sandbox import ImmutableSandboxedEnvironment

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--census',required=True);ap.add_argument('--r1-plan',required=True);ap.add_argument('--tokenizer-config',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 c=json.loads(Path(a.census).read_text());rp=json.loads(Path(a.r1_plan).read_text());out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 train=c['references']['train'];assert ref(train['path'])==train
 docs={str(d['pmc_id']):d for d in map(json.loads,Path(train['path']).read_text().splitlines())}
 sel=c['artifacts']['hard-probe-selection.json'];assert ref(sel['path'])==sel
 selected=json.loads(Path(sel['path']).read_text())
 hpo_ref=c['references']['hpo'];assert ref(hpo_ref['path'])==hpo_ref
 names={};ident=None
 for line in Path(hpo_ref['path']).read_text().splitlines():
  if line=='[Term]':ident=None
  elif line.startswith('id: HP:'):ident=line[4:]
  elif ident and line.startswith('name: '):names[ident]=line[6:]
 # Render with cached published R1 template. Native tokenizer IDs are a later activation check.
 cfg=ref(a.tokenizer_config);config=json.loads(Path(cfg['path']).read_text())
 env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
 def err(s):raise ValueError(s)
 env.globals['raise_exception']=err;template=env.from_string(config['chat_template'])
 rows=[];counts=[]
 system='You are constructing weak supervision from a provided training label. The label can be incomplete. Find textual support, not a post-hoc story. Copy exact source evidence and admit if attribution cannot be established. Proximity alone is not support.'
 for r in selected:
  d=docs[r['pmc_id']];units=[dict(unit_id=f'U{i:04d}',offset=u['offset'],text=u['text']) for i,u in enumerate(d['full_text'])]
  body=dict(target_patient=r['patient_id'],known_training_positive_concept=dict(key=r['key'],public_HPO_name=names.get(r['key'][1]) if r['key'][0]=='hpo' else None),all_patients=d['patient'],report=units)
  user='This is a TRAINING record. The provided positive association is a hint, not evidence. Return final JSON with supported (boolean), patient_anchor, phenotype_anchor, bridge (0 to 3 exact source snippets), and a brief rationale (at most 80 words). Every anchor/snippet must be [unit_id, exact_quote, zero_based_occurrence]. Cite evidence that connects the phenotype to this patient, including group membership or coreference when needed. If the report only supports another case, or the chain is unclear, return supported=false and empty arrays. Do not infer support from the provided label or nearest distance.\n'+json.dumps(body,ensure_ascii=False,separators=(',',':'))
  messages=[dict(role='system',content=system),dict(role='user',content=user)]
  text=template.render(messages=messages,add_generation_prompt=True,**{k:v for k,v in config.items() if k.endswith('_token')})
  rows.append(dict(row_id=f"{r['pmc_id']}:{r['patient_id']}:{r['key'][1]}",pmc_id=r['pmc_id'],messages=messages,train_label_conditioned=True))
  counts.append(dict(row_id=rows[-1]['row_id'],rendered_characters=len(text),rendered_sha256=hashlib.sha256(text.encode()).hexdigest()))
 p=out/'teacher-inputs.proposed.jsonl'
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 put(out/'input-census.json',dict(status='PROPOSED_INPUTS_TEXT_ONLY_NO_MODEL',source=ref(__file__),census=ref(a.census),hpo=hpo_ref,public_r1_full80_plan=ref(a.r1_plan),tokenizer_config=cfg,input=ref(p),rows=counts,count=len(rows),total_rendered_characters=sum(r['rendered_characters'] for r in counts),max_rendered_characters=max(r['rendered_characters'] for r in counts),tokenization_status='Not performed; any GPU activation needs actual native complete-input IDs and length gate',proposed_max_new_tokens=3072,model_calls=0,held_gold_reads=0,fit_calls=0,quality_calls=0))
 print((out/'input-census.json').read_text())
if __name__=='__main__':main()
