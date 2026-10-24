"""Fixed B4 deletion: document concept support OR two pure positive sources."""
import argparse,copy,datetime,hashlib,importlib,json,os,re,sys
from collections import Counter
from pathlib import Path
from literal_helpers import association_span,nonrelation_bytes
sys.dont_write_bytecode=True
FORMULA='B4 INTERSECT (Task1_document_support UNION at_least_two_pure_G_S_Q_C)'
HPO=re.compile(r'HP:[0-9]{7}\Z')
COMPOUND=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+\Z')
def atoms(value):return value.split(';') if COMPOUND.fullmatch(value) else [value]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ref(path):
 p=Path(path).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert ref(r['path'])==r,r['path'];return Path(r['path'])
def put(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def pairs(documents):return {(str(d['pmc_id']),a['patient_id'],atom) for d in documents for a in d['association'] for x in a['phenotype'] for atom in atoms(x)}
def doc_supported(key,concepts):return ('hpo' if HPO.fullmatch(key[2]) else 'unlinked_text',key[2]) in concepts.get(key[0],set())
def select(base,concepts,sources):
 assert set(sources)==set('GSQC')
 decisions={}
 for key in sorted(base):
  votes={s:key in sources[s] for s in 'GSQC'};support=doc_supported(key,concepts)
  decisions[key]=dict(document_supported=support,positive_sources=votes,votes=sum(votes.values()),retain=support or sum(votes.values())>=2)
 return {k for k,v in decisions.items() if v['retain']},decisions
def edit_lines(lines,keep):
 result=[]
 for line in lines:
  d=json.loads(line);new=copy.deepcopy(d['association']);changed=False
  for a in new:
   values=[]
   for value in a['phenotype']:
    aa=atoms(value);kept=[x for x in aa if (str(d['pmc_id']),a['patient_id'],x) in keep]
    if kept==aa:values.append(value)
    else:
     changed=True
     if kept:values.append(';'.join(kept))
   a['phenotype']=values
  if not changed:result.append(line)
  else:
   begin,end=association_span(line);result.append(line[:begin]+json.dumps(new,ensure_ascii=False,separators=(',',':'))+line[end:])
 assert nonrelation_bytes(lines)==nonrelation_bytes(result)
 return result
def source_pairs(documents,patients):
 assert len(documents)==len(patients) and {str(d['pmc_id']) for d in documents}==set(patients)
 for d in documents:
  ids=[a['patient_id'] for a in d['association']]
  assert len(ids)==len(set(ids)) and set(ids)==patients[str(d['pmc_id'])]
 return pairs(documents)
def sample_intersection(diagnostics,patients,expected_rows):
 seen={'S1':{},'S2':{}};sets={'S1':set(),'S2':set()}
 for r in diagnostics:
  stream=r['stream']
  if stream=='G':continue # Historical greedy diagnostics are not this pure grounded source.
  assert stream in seen and r['row_id'] not in seen[stream]
  assert r['strict_complete'] and r['actual_natural_eos'] and r['fresh']
  assert not r['actual_hit_token_limit'] and not r['actual_has_think_marker'] and not r['whole_row_abstention'] and not r['parser_error']
  doc=str(r['pmc_id']);assert doc in patients
  for d,p,v in r['relations']:
   assert str(d)==doc and p in patients[doc] and isinstance(v,str) and v
   sets[stream].update((doc,p,a) for a in atoms(v))
  seen[stream][r['row_id']]=doc
 assert seen['S1']==seen['S2'] and len(seen['S1'])==expected_rows and set(seen['S1'].values())==set(patients)
 return sets['S1']&sets['S2'],{s:len(v) for s,v in sets.items()}
def main(plan_path,mode):
 p=json.loads(Path(plan_path).read_text());r=p['refs'];out=Path(p['output']);allow_gold=False
 assert p['formula']==FORMULA and p['positive_source_count']==2
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and '.gold.jsonl' in os.fsdecode(args[0]):
   if not allow_gold or os.path.realpath(os.fsdecode(args[0]))!=os.path.realpath(r['gold']['path']):raise PermissionError('Gold only after complete frozen endpoint')
 sys.addaudithook(audit)
 for q in p['source_files'].values():checked(q)
 checked(r['decision']);checked(r['validator']);evpath=checked(r['evaluator']);sys.path.insert(0,str(evpath.parent));ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions')
 assert ev.VERSION=='patientphex-public-approx-v2'
 if mode=='assemble':
  out.mkdir(parents=True,exist_ok=False);put(out/'started.json',dict(at_utc=now(),plan=ref(plan_path)))
  bp=checked(r['baseline']);bl=bp.read_text().splitlines(keepends=True);base=[json.loads(x) for x in bl];inp=va.load_jsonl(checked(r['input']))
  patients={str(d['pmc_id']):{q['patient_id'] for q in d['patient']} for d in inp}
  assert len(base)==len(inp)==len(patients)==16 and sum(map(len,patients.values()))==p['patients'] and sum(len(d['entities']) for d in base)==p['entities']
  original=source_pairs(base,patients);assert len(original)==p['baseline_relations']
  task1=va.load_jsonl(checked(r['task1']));assert len(task1)==16 and {str(d['pmc_id']) for d in task1}==set(patients)
  concepts={str(d['pmc_id']):ev.document_concepts(d,d,'all','text') for d in task1}
  sources={s:source_pairs(va.load_jsonl(checked(p['pure_sources'][s])),patients) for s in 'GQC'}
  sources['S'],samplecounts=sample_intersection(rows(checked(p['sampled_diagnostics'])),patients,p['sample_rows_per_seed'])
  keep,decisions=select(original,concepts,sources);removed=original-keep;pl=edit_lines(bl,keep);pred=[json.loads(x) for x in pl]
  assert pairs(pred)==keep and keep<=original
  for a,b in zip(base,pred):
   assert {k:v for k,v in a.items() if k!='association'}=={k:v for k,v in b.items() if k!='association'}
   for aa,bb in zip(a['association'],b['association']):assert {k:v for k,v in aa.items() if k!='phenotype'}=={k:v for k,v in bb.items() if k!='phenotype'}
  val=va.validate_predictions(inp,pred,va.read_hpo(checked(r['hpo'])));put(out/'format.json',val);assert val['valid']
  for d in pred:assert ev.associations(d)=={a['patient_id']:set(y for x in a['phenotype'] for y in atoms(x)) for a in d['association']}
  with (out/'prediction.jsonl').open('x') as f:f.writelines(pl);f.flush();os.fsync(f.fileno())
  if not removed:assert (out/'prediction.jsonl').read_bytes()==bp.read_bytes()
  with (out/'support-audit.jsonl').open('x') as f:
   for k,v in decisions.items():f.write(json.dumps(dict(key=k,**v),ensure_ascii=False)+'\n')
   f.flush();os.fsync(f.fileno())
  put(out/'freeze.json',dict(at_utc=now(),plan=ref(plan_path),prediction=ref(out/'prediction.jsonl'),format=ref(out/'format.json'),audit=ref(out/'support-audit.jsonl'),baseline_pairs=len(original),final_pairs=len(keep),removed=len(removed),removed_pairs=sorted(removed),source_counts={s:len(v) for s,v in sources.items()},sample_raw_counts=samplecounts,nonassociation_bytes_exact=True,other_objects_exact=True,added=0,gold_read=False,NN_calls=0,fit_calls=0))
 else:
  fr=json.loads((out/'freeze.json').read_text());assert fr['plan']==ref(plan_path)
  for q in ['prediction','format','audit']:checked(fr[q])
  old=json.loads(checked(r['baseline_metrics']).read_text())['metrics'];assert old['Score']==p['baseline_score']
  if fr['removed']==0:
   assert checked(fr['prediction']).read_bytes()==checked(r['baseline']).read_bytes()
   put(out/'result.json',dict(status='ZERO_DELETION_BYTEEXACT_CLOSED',retain=False,new_v2=0,old_v2=0,baseline=old,freeze=ref(out/'freeze.json')));return
  put(out/'evaluation-intent.json',dict(at_utc=now(),freeze=ref(out/'freeze.json'),new_v2=1,old_v2=0));allow_gold=True
  gold=va.load_jsonl(checked(r['gold']));pred=va.load_jsonl(fr['prediction']['path']);metrics=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text');put(out/'metrics.json',metrics)
  delta={k:v-old[k] for k,v in metrics['metrics'].items()};assert delta['menF1']==delta['docF1']==0
  deleted=set(map(tuple,fr['removed_pairs']));tp=len(deleted&pairs(gold))
  put(out/'result.json',dict(status='COMPLETE_FIXED_DOCUMENT_SUPPORT_FILTER',at_utc=now(),fold=p['fold'],baseline=old,metrics=metrics['metrics'],delta=delta,removed=len(deleted),removed_TP=tp,removed_FP=len(deleted)-tp,retain=delta['Score']>0,new_v2=1,old_v2=0,NN_calls=0,fit_calls=0,freeze=ref(out/'freeze.json'),development_not_blind=True))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('mode',choices=['assemble','evaluate']);x=a.parse_args();main(x.plan,x.mode)
