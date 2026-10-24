"""One conservative four-explicit-negative veto; no model, fit or threshold search."""
import argparse
from collections import Counter,defaultdict
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from literal_helpers import association_span,nonrelation_bytes

HPO=re.compile(r'HP:[0-9]{7}\Z')
COMPOUND=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+\Z')
STREAMS=('S1','S2','Q','C')
def atoms(value):return value.split(';') if COMPOUND.fullmatch(value) else [value]
def atomic_negative(value):return bool(HPO.fullmatch(value)) or (bool(value) and value!='-1' and 'HP:' not in value)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ref(path):
 p=Path(path).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
 assert ref(r['path'])==r,r['path']
 return Path(r['path'])
def put(path,value):
 with Path(path).open('x') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def index(records):
 d={x['row_id']:x for x in records};assert len(d)==len(records);return d

def stream_evidence(prompts,canonical,raw,patients):
 """Canonical maps already decoded by each original strict owner; never reparse raw text."""
 cm,rm=index(canonical),index(raw);pi=index(prompts)
 assert not (set(cm)-set(pi)) and not (set(rm)-set(pi))
 evidence=defaultdict(list)
 for row in prompts:
  doc=str(row['pmc_id']);rid=row['row_id'];c=cm.get(rid);r=rm.get(rid)
  assert doc in patients and len(row['patient_ids'])==len(set(row['patient_ids']))
  assert set(row['patient_ids'])<=patients[doc]
  mapping=c.get('candidate_patients',{}) if c else {}
  if c:assert str(c['pmc_id'])==doc
  if r:assert str(r['pmc_id'])==doc
  strict=bool(c and r and c.get('strict_complete') is True and c.get('abstention') is None and r.get('natural_eos') is True and r.get('hit_token_limit') is False)
  if c and 'natural_eos' in c:strict=strict and c['natural_eos'] is True and c.get('hit_token_limit') is False
  strict=strict and isinstance(mapping,dict) and set(mapping)==set(row['code_map'])
  if strict:
   strict=all(isinstance(v,list) and len(v)==len(set(v)) and all(type(pid)is str and pid in row['patient_ids'] for pid in v) for v in mapping.values())
  for cid,value in row['code_map'].items():
   assert isinstance(value,str) and value
   for pid in sorted(patients[doc]):
    covered=pid in row['patient_ids']
    state='unknown';reason='row_not_strict_or_incomplete'
    if not covered:reason='patient_not_in_candidate_row'
    elif strict:
     if pid in mapping[cid]:state='positive';reason='strict_positive'
     elif atomic_negative(value):state='negative';reason='explicit_atomic_negative'
     else:reason='compound_or_nonliteral_negative_not_atomic'
    for atom in atoms(value):
     evidence[(doc,pid,atom)].append(dict(row_id=rid,cid=cid,candidate=value,state=state,reason=reason))
 return dict(evidence)

def classify(observations):
 if not observations:return 'missing'
 if any(x['state']=='positive' for x in observations):return 'positive'
 if all(x['state']=='negative' for x in observations):return 'negative'
 return 'unknown'

def veto_keys(base,streams,grounded):
 decisions={};removed=set()
 for key in sorted(base):
  statuses={s:classify(streams[s].get(key)) for s in STREAMS}
  remove=key not in grounded and all(v=='negative' for v in statuses.values())
  if remove:removed.add(key)
  decisions[key]=dict(streams=statuses,grounded_protect=key in grounded,remove=remove)
 return removed,decisions

def edit_lines(lines,removed):
 result=[]
 for line in lines:
  d=json.loads(line);new=copy.deepcopy(d['association']);changed=False
  for a in new:
   values=[]
   for value in a['phenotype']:
    aa=atoms(value);kept=[x for x in aa if (str(d['pmc_id']),a['patient_id'],x) not in removed]
    if kept==aa:values.append(value)
    else:
     changed=True
     if kept:values.append(';'.join(kept))
   a['phenotype']=values
  if not changed:result.append(line)
  else:
   begin,end=association_span(line)
   result.append(line[:begin]+json.dumps(new,ensure_ascii=False,separators=(',',':'))+line[end:])
 assert nonrelation_bytes(lines)==nonrelation_bytes(result)
 return result

def all_pairs(documents):
 return {(str(d['pmc_id']),a['patient_id'],atom) for d in documents for a in d['association'] for x in a['phenotype'] for atom in atoms(x)}

def main(plan_path,mode):
 p=json.loads(Path(plan_path).read_text());o=Path(p['output']);refs=p['refs']
 assert p['streams']==list(STREAMS) and p['formula']=='B4 minus (S1neg intersect S2neg intersect Qneg intersect Cneg minus Gpositive)'
 for x in p['source_files'].values():checked(x)
 checked(refs['decision']);evpath=checked(refs['evaluator']);checked(refs['validator'])
 sys.path.insert(0,str(evpath.parent));ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions')
 assert ev.VERSION=='patientphex-public-approx-v2' and ev.COMPOUND_HPO.pattern==r'HP:[0-9]{7}(?:;HP:[0-9]{7})+'
 if mode=='assemble':
  forbidden=os.path.realpath(refs['gold']['path'])
  def guard(event,args):
   if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==forbidden:raise PermissionError('Gold forbidden before full freeze')
  sys.addaudithook(guard)
  o.mkdir(parents=True,exist_ok=False);put(o/'started.json',dict(at_utc=now(),plan=ref(plan_path)))
  basepath=checked(refs['baseline']);lines=basepath.read_text().splitlines(keepends=True);base=[json.loads(x) for x in lines]
  inp=va.load_jsonl(checked(refs['input']));patients={d['pmc_id']:{a['patient_id'] for a in d['association']} for d in base}
  assert len(base)==len(inp)==len(patients)==16 and set(patients)=={d['pmc_id'] for d in inp}
  assert sum(map(len,patients.values()))==p['patients'] and sum(len(d['entities']) for d in base)==p['entities']
  for d in inp:assert patients[d['pmc_id']]=={x['patient_id'] for x in d['patient']}
  old=all_pairs(base);assert len(old)==p['baseline_relations']
  g=va.load_jsonl(checked(refs['grounded_pure']));assert set(patients)=={d['pmc_id'] for d in g};grounded=all_pairs(g)
  streams={};counts={}
  for name in STREAMS:
   rr=p['stream_refs'][name];pp=rows(checked(rr['prompts']));cc=rows(checked(rr['canonical']));raw=rows(checked(rr['raw']))
   assert len(pp)==len(cc)==len(raw)==p['rows_per_stream'];assert set(index(pp))==set(index(cc))==set(index(raw))
   streams[name]=stream_evidence(pp,cc,raw,patients)
   counts[name]=dict(Counter(classify(v) for v in streams[name].values()))
  removed,decisions=veto_keys(old,streams,grounded)
  output_lines=edit_lines(lines,removed);pred=[json.loads(x) for x in output_lines]
  assert all_pairs(pred)==old-removed
  for before,after in zip(base,pred):
   assert {k:v for k,v in before.items() if k!='association'}=={k:v for k,v in after.items() if k!='association'}
   for a,b in zip(before['association'],after['association']):assert {k:v for k,v in a.items() if k!='phenotype'}=={k:v for k,v in b.items() if k!='phenotype'}
  valid=va.validate_predictions(inp,pred,va.read_hpo(checked(refs['hpo'])));put(o/'format.json',valid);assert valid['valid']
  for d in pred:assert ev.associations(d)=={a['patient_id']:set(y for v in a['phenotype'] for y in atoms(v)) for a in d['association']}
  pp=o/'prediction.jsonl'
  with pp.open('x') as f:f.writelines(output_lines);f.flush();os.fsync(f.fileno())
  if not removed:assert pp.read_bytes()==basepath.read_bytes()
  with (o/'pair-decisions.jsonl').open('x') as f:
   for k,v in decisions.items():f.write(json.dumps(dict(key=k,**v,evidence={s:streams[s].get(k,[]) for s in STREAMS}),ensure_ascii=False)+'\n')
   f.flush();os.fsync(f.fileno())
  put(o/'freeze.json',dict(at_utc=now(),plan=ref(plan_path),prediction=ref(pp),format=ref(o/'format.json'),evidence=ref(o/'pair-decisions.jsonl'),removed_pairs=sorted(removed),removed=len(removed),baseline_pairs=len(old),final_pairs=len(old-removed),stream_status_counts=counts,nonassociation_bytes_exact=True,gold_read=False,NN_calls=0,fit_calls=0))
 else:
  f=json.loads((o/'freeze.json').read_text());assert f['plan']==ref(plan_path);checked(f['prediction']);checked(f['format']);checked(f['evidence'])
  prior=json.loads(checked(refs['baseline_metrics']).read_text())['metrics'];assert prior['Score']==p['baseline_score']
  if not f['removed']:
   assert checked(f['prediction']).read_bytes()==checked(refs['baseline']).read_bytes()
   put(o/'result.json',dict(status='NO_DELETION_BYTEEXACT_CLOSED_ZERO_V2',at_utc=now(),baseline=prior,retain=False,removed=0,new_v2=0,old_v2=0,freeze=ref(o/'freeze.json')));return
  put(o/'evaluation-intent.json',dict(at_utc=now(),freeze=ref(o/'freeze.json'),new_v2=1,old_v2=0))
  gold=va.load_jsonl(checked(refs['gold']));pred=va.load_jsonl(f['prediction']['path'])
  metrics=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text');put(o/'metrics.json',metrics)
  delta={k:v-prior[k] for k,v in metrics['metrics'].items()};assert delta['menF1']==delta['docF1']==0
  removed=set(map(tuple,f['removed_pairs']));tp=len(removed&all_pairs(gold))
  put(o/'result.json',dict(status='COMPLETE_FIXED_FOLD_VETO',fold=p['fold'],at_utc=now(),baseline=prior,metrics=metrics['metrics'],delta=delta,removed=len(removed),removed_TP=tp,removed_FP=len(removed)-tp,retain=delta['Score']>0,new_v2=1,old_v2=0,NN_calls=0,fit_calls=0,freeze=ref(o/'freeze.json'),development_not_blind=True))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('mode',choices=['assemble','evaluate']);x=a.parse_args();main(x.plan,x.mode)
