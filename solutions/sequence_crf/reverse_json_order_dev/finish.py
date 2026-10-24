#!/usr/bin/env python3
"""Freeze reverse-only and fixed old-or-reverse relation union before quality."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,re,signal,subprocess,sys,time
COMPOUND=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_bytes().splitlines()]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def values(v):
 assert isinstance(v,str)
 return v.split(';') if COMPOUND.fullmatch(v) else [v]
def relations(doc):
 out={}
 for a in doc['association']:
  assert a['patient_id'] not in out
  out[a['patient_id']]={v for x in a['phenotype'] for v in values(x)}
 return out

def union(old,new):
 assert old['pmc_id']==new['pmc_id']
 assert {k:v for k,v in old.items() if k!='association'}=={k:v for k,v in new.items() if k!='association'}
 before,extra=relations(old),relations(new);assert set(before)==set(extra)
 result=copy.deepcopy(old);log=[]
 for a in result['association']:
  pid=a['patient_id'];added=sorted(extra[pid]-before[pid]);a['phenotype'].extend(added)
  log.extend(dict(pmc_id=old['pmc_id'],patient_id=pid,phenotype=v) for v in added)
 assert relations(result)=={p:before[p]|extra[p] for p in before}
 for a,z in zip(old['association'],result['association']):
  assert z['phenotype'][:len(a['phenotype'])]==a['phenotype']
  assert {k:v for k,v in a.items() if k!='phenotype'}=={k:v for k,v in z.items() if k!='phenotype'}
 return result,log

def contracts():
 base=dict(pmc_id='fixture',pmid=None,entities=[{'offset':0,'length':1,'note':None}],extra={'x':True},association=[{'patient_id':'P1','phenotype':['HP:0000001;HP:0000002','None','nil;text','-1'],'unknown':'preserved'},{'patient_id':'P2','phenotype':[]}])
 rev=copy.deepcopy(base);rev['association'][0]['phenotype']=['HP:0000002;HP:0000003','None','none','MESH:D1;HP:0000001','nil;text'];rev['association'][1]['phenotype']=['a;b']
 out,log=union(base,rev);assert out['association'][0]['phenotype'][:4]==base['association'][0]['phenotype'];assert len(log)==4
 assert relations(out)['P1']=={'HP:0000001','HP:0000002','HP:0000003','None','none','nil;text','-1','MESH:D1;HP:0000001'}
 assert out['association'][1]['phenotype']==['a;b'];assert union(base,base)[0]==base
 absent=copy.deepcopy(rev);absent['association'].pop()
 try:union(base,absent)
 except AssertionError:pass
 else:raise AssertionError('Missing patient accepted')
 modified=copy.deepcopy(rev);modified['entities']=[]
 try:union(base,modified)
 except AssertionError:pass
 else:raise AssertionError('Changed entities accepted')
 return dict(compound_exact=True,literal_NIL_and_nonHPO_semicolon_preserved=True,case_sensitive=True,old_objects_order_preserved=True,duplicate_noop=True,patient_and_nonassociation_guards=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['contracts','assemble','evaluate']);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--gold',type=Path);ap.add_argument('--scripts',type=Path);ap.add_argument('--hpo',type=Path);a=ap.parse_args()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU>30s')));signal.alarm(30);tick=time.monotonic();b=a.directory;p=load(b/'plan.json');refs=p['local_references'];assert ref(__file__)['sha256']==p['control_sources']['finish.py']
 for r in refs.values():assert ref(r['path'])['sha256']==r['sha256']
 if a.mode=='contracts':
  save(b/'assembly-contracts.json',dict(source=ref(__file__),checks=contracts(),seconds=time.monotonic()-tick));return
 complete=load(b/'terminal/complete.json');assert complete['plan_sha256']==ref(b/'plan.json')['sha256'] and complete['responses_sha256']==ref(b/'terminal/responses.jsonl')['sha256']
 if a.mode=='assemble':
  assert a.gold is a.scripts is a.hpo is None
  assert ref(b/'legacy/merge_pair_rag.py')['sha256']==p['files']['legacy/merge_pair_rag.py'];out=b/'assembled';out.mkdir(exist_ok=False)
  cmd=[sys.executable,str(b/'legacy/merge_pair_rag.py'),'--input',refs['input']['path'],'--entities',refs['entities']['path'],'--final-entities',refs['baseline']['path'],'--prompts',str(b/'prompts.jsonl'),'--responses',str(b/'terminal/responses.jsonl'),'--output',str(out/'reverse.jsonl'),'--fallback','baseline']
  rc=subprocess.run(cmd,text=True,capture_output=True,timeout=20);save(out/'command.json',dict(argv=cmd,returncode=rc.returncode,stdout=rc.stdout,stderr=rc.stderr));assert rc.returncode==0
  old,new=rows(refs['baseline']['path']),rows(out/'reverse.jsonl');assert sum(len(d['entities']) for d in old)==1072 and sum(len(x['phenotype']) for d in old for x in d['association'])==368;assert [x['pmc_id'] for x in old]==[x['pmc_id'] for x in new]
  full=[];added=[]
  for x,y in zip(old,new):
   z,log=union(x,y);full.append(z);added.extend(log)
  with (out/'union.jsonl').open('x') as f:
   for d in full:f.write(json.dumps(d,ensure_ascii=False)+'\n')
  save(out/'added-relations.json',added)
  save(out/'freeze.json',dict(frozen_at_utc=now(),plan=ref(b/'plan.json'),responses=ref(b/'terminal/responses.jsonl'),baseline=refs['baseline'],predictions={n:ref(out/(n+'.jsonl')) for n in ['reverse','union']},parser=ref(out/'reverse.manifest.json'),added=ref(out/'added-relations.json'),all_nonassociation_fields_exact=True,old_relations_all_preserved_union=True,gold_read=False,seconds=time.monotonic()-tick));print((out/'freeze.json').read_text())
 else:
  fr=load(b/'assembled/freeze.json');assert fr['gold_read'] is False
  for r in fr['predictions'].values():assert r==ref(r['path'])
  out=b/'evaluation';out.mkdir(exist_ok=False);save(out/'before-gold.json',dict(time_utc=now(),freeze=ref(b/'assembled/freeze.json'),gold_read=False,evaluator=ref(a.scripts/'evaluate.py'),validator=ref(a.scripts/'validate_predictions.py')))
  sys.path.insert(0,str(a.scripts));import evaluate as ev;import validate_predictions as va
  assert ev.VERSION=='patientphex-public-approx-v2' and ev.COMPOUND_HPO.pattern==COMPOUND.pattern
  inputs=va.load_jsonl(refs['input']['path']);gold=va.load_jsonl(a.gold);assert {d['pmc_id'] for d in inputs}=={d['pmc_id'] for d in gold}
  hpo=va.read_hpo(a.hpo);assert hpo['version']=='hp/releases/2026-06-23';base=load(refs['baseline_metrics']['path']);old={d['pmc_id']:d for d in rows(refs['baseline']['path'])};goldmap={d['pmc_id']:d for d in gold};reports={}
  for name in ['union']:
   pred=va.load_jsonl(b/'assembled'/f'{name}.jsonl');v=va.validate_predictions(inputs,pred,hpo);save(out/f'{name}.validation.json',v);assert v['valid'];m=ev.evaluate_report(gold,pred,'all','text');save(out/f'{name}.metrics.json',m)
   delta={k:m['metrics'][k]-z for k,z in base['metrics'].items()};assert delta['menF1']==delta['docF1']==0
   changes=dict(added_tp=0,added_fp=0,removed_tp=0,removed_fp=0)
   for d in pred:
    pid=d['pmc_id'];g=relations(goldmap[pid]);o=relations(old[pid]);n=relations(d)
    for patient in o:
     added=n[patient]-o[patient];removed=o[patient]-n[patient]
     changes['added_tp']+=len(added&g[patient]);changes['added_fp']+=len(added-g[patient]);changes['removed_tp']+=len(removed&g[patient]);changes['removed_fp']+=len(removed-g[patient])
   reports[name]=dict(metrics=m['metrics'],delta=delta,changes=changes)
  keep=reports['union']['delta']['Score']>0 and reports['union']['delta']['macF1']>=0
  save(out/'results.json',dict(completed_at_utc=now(),baseline_metrics=base['metrics'],arms=reports,keep_union=keep,primary='fixed old-or-reverse literal-v2 union',keep_rule='Union Score strictly increases and macro does not decrease',baseline_reused=True,seconds=time.monotonic()-tick,scope='Originaldev development, not blind. Original64 old-pool prompts/full64 adapter. Only union with current baac evaluated; reverse-alone unscored. All1072 entities and368 old relations retained, including35 publicnew support relations; no support recomputation or further combination.'));print((out/'results.json').read_text())
if __name__=='__main__':main()
