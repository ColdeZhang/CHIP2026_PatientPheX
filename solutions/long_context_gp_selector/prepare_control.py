"""Four ancestor-exact GP groups, unchanged admission, complete candidate mapping."""
import ast,gzip,hashlib,importlib.util,json,re,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/long-context-gp-selector-readiness-v1'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 start=time.monotonic();plan=json.loads((S/'plan.json').read_text());old=json.loads(Path(plan['references']['old_gp_plan']['path']).read_text())
 sys.path[:0]=[plan['cpu_libraries'],plan['new_gp_source']]
 import prepare as enc
 tok=enc.Tokenizer.from_file(plan['references']['tokenizer']['path']);tok.no_truncation();tok.no_padding()
 hposrc=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/run.py')
 fn=next(n for n in ast.parse(hposrc.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parse_hpo');ns={'Path':Path,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(hposrc),'exec'),ns)
 _,allowed,aliases,version=ns['parse_hpo']('/workspace/patientphex-data/hp.obo')
 olddata=json.loads(gzip.decompress(Path(plan['references']['new_gp_data']['path']).read_bytes()))
 contracts={};jobs=[];summaries=[]
 for j in old['jobs']:
  for key in ['train','input','targets']:assert ref(j[key]['path'])==j[key]
  train=read(j['train']['path']);pure=read(j['input']['path']);targets=read(j['targets']['path'])
  trids={str(d['pmc_id']) for d in train};heldids={str(d['pmc_id']) for d in pure}
  assert trids==set(j['train_ids']) and heldids==set(j['input_ids']) and trids.isdisjoint(heldids|set(old['outer1_held_ids']))
  tr,tc,cov=enc.encode_docs(train,tok,allowed,aliases,True);va,hc,_=enc.encode_docs(pure,tok,allowed,aliases,False)
  assert len(tr)==j['training_documents'] and all('positive' not in b for row in va for b in row['blocks'])
  if j['name']=='inner0':assert json.loads(json.dumps(tr))==olddata['train'],'GP23 input/targets/order exact'
  if j['name']=='final48':assert json.loads(json.dumps(va))==olddata['held'],'F1 full input exact'
  for c in tc+hc:
   if c['pmc_id'] in contracts:assert contracts[c['pmc_id']]==c
   contracts[c['pmc_id']]=c
  bydoc={p:[] for p in heldids}
  for row in va:
   for bi,b in enumerate(row['blocks']):
    bydoc[row['pmc_id']].append((row['row_id'],bi,{x+b['passage_offset']:i for i,(x,y) in enumerate(b['offsets'])},{y+b['passage_offset']:i for i,(x,y) in enumerate(b['offsets'])}))
  maps={r['row_id']:[] for r in va};records=[]
  for ti,t in enumerate(targets):
   p=str(t['pmc_id']);a=int(t['offset']);z=a+int(t['length']);ssseen=eesseen=False;valid=False
   for rowid,bi,ss,ee in bydoc[p]:
    ssseen|=a in ss;eesseen|=z in ee
    if a in ss and z in ee and ss[a]<=ee[z]:maps[rowid].append(dict(target_index=ti,block_index=bi,i=ss[a],j=ee[z]));valid=True
   records.append(dict(**t,lcgp_span_valid=valid,missing_reason=None if valid else ('no_common_window' if ssseen and eesseen else 'unaligned_endpoint')))
  job=dict(name=j['name'],reuse=j['name']=='inner0',train=tr,held=va,targets=records,window_targets=maps,source_refs={k:j[k] for k in ['train','input','targets']},train_ids=sorted(trids),input_ids=sorted(heldids),coverage=cov)
  jobs.append(job);summaries.append(dict(name=j['name'],reuse=job['reuse'],train_docs=len(train),train_windows=len(tr),held_docs=len(pure),held_windows=len(va),target_spans=len(targets),representable=sum(r['lcgp_span_valid'] for r in records),accepted=len(cov['accepted_keys']),aligned=len(cov['aligned_keys']),unrepresentable=len(cov['unrepresentable']),max_train_tokens=max(len(r['ids']) for r in tr),max_held_tokens=max(len(r['ids']) for r in va),actual_adam_updates=0 if job['reuse'] else 8*len(tr)))
 assert len(contracts)==64 and len(jobs)==4 and sum(not j['reuse'] for j in jobs)==3
 out=D/'formal-v1';out.mkdir(exist_ok=True);assert not list(out.iterdir());raw=json.dumps(dict(jobs=jobs,native_contracts=list(contracts.values())),ensure_ascii=False,separators=(',',':')).encode();p=out/'data.json.gz';p.write_bytes(gzip.compress(raw,mtime=0))
 summary=dict(status='FULL_ANCESTOR_EXACT_DATA_AND_ALL_CANDIDATE_MAPS_NO_NN',data=ref(p),source=ref(__file__),old_plan=ref(plan['references']['old_gp_plan']['path']),prepare_source=ref(Path(plan['new_gp_source'])/'prepare.py'),hpo=ref('/workspace/patientphex-data/hp.obo'),admission_source=ref(hposrc),jobs=summaries,native_documents=64,F1_gold_reads=0,GP23_train_byteobjects_exact=True,F1_held_byteobjects_exact=True,seconds=time.monotonic()-start)
 (out/'local-preparation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
