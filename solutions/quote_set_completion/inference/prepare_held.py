"""One fixed held completion input: original G32 accepted quotes only, no held labels."""
from pathlib import Path
from collections import Counter
import argparse,copy,gzip,hashlib,importlib.util,json,os,signal,sys,time

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(p,v,jsonl=False):
 raw=(''.join(compact(r)+'\n' for r in v) if jsonl else json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);args=a.parse_args();start=time.monotonic()
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   name=os.fsdecode(args[0])
   if '.gold.jsonl' in name or 'validation16.' in name:raise PermissionError('Pure held preparation forbids gold')
 sys.addaudithook(audit)
 plan=json.loads(Path(args.plan).read_text());assert plan['source']==ref(__file__) and plan['known_source']=='originalG32greedy_accepted_only'
 for item in plan['references'].values():assert ref(item['path'])==item
 sys.path.insert(0,plan['lightweight_libraries'])
 schema=module('original_grounded_schema',plan['references']['schema']['path']);native=module('original_native',plan['references']['native']['path'])
 construct=module('fixed_training_constructor',plan['references']['constructor']['path'])
 tokenizer=native.FrozenTokenizer(plan['references']['tokenizer']['path'],plan['references']['template']['path'])
 rows=[json.loads(l) for l in Path(plan['references']['input']['path']).read_text().splitlines()]
 norms=[json.loads(l) for l in Path(plan['references']['normalizations']['path']).read_text().splitlines()]
 assert len(rows)==39 and len({r['pmc_id'] for r in rows})==16 and len({r['row_id'] for r in rows})==39
 assert len(norms)==573 and Counter(n['status'] for n in norms)['mapped']+Counter(n['status'] for n in norms)['nil']==328
 by={r['row_id']:[] for r in rows}
 for n in norms:
  assert n['row_id'] in by and n['policy_sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
  if n['status'] in ['mapped','nil']:by[n['row_id']].append(n)
 output=[];encoded=[];audits=[];knowns=[]
 for original in rows:
  assert 'answer' not in original and 'labels' not in original
  entries=sorted(by[original['row_id']],key=lambda n:(n['offset'],n['length'],n['unit_id'],n['occurrence_index'],n['quote']))
  triples=[];seen=set();units={u['unit_id']:u for u in original['units']}
  for n in entries:
   assert n['pmc_id']==original['pmc_id'] and n['target_patient_id']==original['target_patient_id']
   t=[n['unit_id'],n['quote'],n['occurrence_index']];key=compact(t)
   grounded=schema.grounded(t,units);assert grounded['offset']==n['offset'] and grounded['length']==n['length']
   if key not in seen:seen.add(key);triples.append(t)
  # Full-known view supplies exactly the training instruction and known suffix.
  # Here these are model predictions, never label targets; remove training fields.
  synthetic=copy.deepcopy(original);synthetic['answer']=compact(triples)
  prepared=construct.views(synthetic)[2]
  del prepared['answer'];del prepared['completion_metadata'];prepared['row_id']=original['row_id']+':completion-held'
  prepared['completion_input_metadata']=dict(original_row_id=original['row_id'],known=triples,known_source='originalG32greedy_accepted_fixedNormalizer',known_are_training_gold=False,passes=1)
  assert prepared['units']==original['units'] and prepared['messages'][1]['content'].startswith(original['messages'][1]['content'])
  value=native.encode(prepared,tokenizer);assert 'labels' not in value and 'answer' not in prepared
  output.append(prepared);encoded.append(value)
  knowns.append(dict(row_id=prepared['row_id'],original_row_id=original['row_id'],pmc_id=original['pmc_id'],target_patient_id=original['target_patient_id'],triples=triples,source_entries=len(entries)))
  audits.append(dict(index=len(audits),row_id=prepared['row_id'],original_row_id=original['row_id'],pmc_id=original['pmc_id'],target_patient_id=original['target_patient_id'],tokens=len(value['input_ids']),input_ids_sha256=hashlib.sha256(json.dumps(value['input_ids']).encode()).hexdigest(),rendered_prefix_sha256=value['rendered_prefix_sha256'],known_count=len(triples)))
 out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
 files={'prompts':save(out/'prompts.jsonl',output,True),'known':save(out/'known.json',knowns),'audit':save(out/'audit.json',audits)}
 with (out/'encoded.json.gz').open('xb') as f:f.write(gzip.compress((compact({'train':[],'held':encoded})+'\n').encode(),mtime=0));f.flush();os.fsync(f.fileno())
 files['encoded']=ref(out/'encoded.json.gz')
 result=dict(status='COMPLETE_PURE39_KNOWN_ONLY_INPUT_LOCAL_NATIVE',source=ref(__file__),plan=ref(args.plan),files=files,rows=39,documents=16,original_normalizations=573,accepted_source_entries=328,known_triples=sum(len(x['triples']) for x in knowns),empty_known_rows=sum(not x['triples'] for x in knowns),input_tokens=sum(x['tokens'] for x in audits),max_input_tokens=max(x['tokens'] for x in audits),no_held_gold=True,no_GPU=True,no_quality=True,HF_actual_replay_pending=True,seconds=time.monotonic()-start)
 print(json.dumps(save(out/'manifest.json',result)))
if __name__=='__main__':main()
