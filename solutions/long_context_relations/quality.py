"""Single predeclared B4 protected relation union. No candidate/threshold scans."""
from copy import deepcopy
from datetime import datetime,timezone
import gzip,importlib.util,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from common import save,ref

def atoms(values):
 out=set()
 for v in values:out.update(v.split(';')if re.fullmatch(r'HP:\d{7}(;HP:\d{7})+',v)else[v])
 return out

def assemble(baseline,positive):
 result=deepcopy(baseline);added=[]
 for old,new in zip(baseline,result):
  pmc=str(old['pmc_id'])
  for a,b in zip(old['association'],new['association']):
   pid=str(a['patient_id']);incoming=positive.get((pmc,pid),set())
   assert all(re.fullmatch(r'HP:\d{7}',cid)for cid in incoming)
   extra=sorted(incoming-atoms(a['phenotype']));b['phenotype']=a['phenotype']+extra
   assert b['phenotype'][:len(a['phenotype'])]==a['phenotype']
   assert {k:v for k,v in a.items()if k!='phenotype'}=={k:v for k,v in b.items()if k!='phenotype'}
   added.extend([pmc,pid,c]for c in extra)
  assert {k:v for k,v in old.items()if k!='association'}=={k:v for k,v in new.items()if k!='association'}
 return result,added

def contract():
 old=[dict(pmc_id='a',entities=[dict(offset=8,text='keep')],pmid='fixed',association=[dict(patient_id='P1',phenotype=['HP:0000001;HP:0000002','literal NIL'],extra='keep'),dict(patient_id='P2',phenotype=[])])]
 new,added=assemble(old,{('a','P1'):{'HP:0000002','HP:0000003'},('a','P2'):{'HP:0000004'}})
 assert added==[['a','P1','HP:0000003'],['a','P2','HP:0000004']]
 assert new[0]['association'][0]['phenotype']==['HP:0000001;HP:0000002','literal NIL','HP:0000003']
 assert assemble(old,{})==(old,[])
 try:assemble(old,{('a','P1'):{'HP:0000001;HP:0000002'}})
 except AssertionError:pass
 else:raise AssertionError('compound model candidate unexpectedly permitted')
 print('PROTECTED_UNION_CONTRACT_PASS_NO_GOLD_OR_V2')

def main():
 S=Path(__file__).resolve().parent;p=json.loads((S/'quality-plan.json').read_text());out=Path(p['output']);assert not(out/'freeze.json').exists()
 for name in ('baseline','baseline_metrics','held_geometry','input','evaluator'):
  assert ref(p[name]['path'])['sha256']==p[name]['sha256'],name
 terminal=Path(p['terminal']);complete=json.loads((terminal/'output/complete.json').read_text());end=json.loads((terminal/'job/ended.json').read_text())
 assert end['returncode']==0 and not end['timed_out']
 assert complete['status']=='COMPLETE_FRESH30EP240_AND_ALL16_HELD_RELATIONS' and complete['actual_adam_updates']==240 and complete['held_pairs']==2874
 scorepath=terminal/'output/held-scores.jsonl';assert ref(scorepath)['sha256']==complete['held_scores']['sha256']
 held=json.loads(gzip.decompress(Path(p['held_geometry']['path']).read_bytes()));expected={tuple(key)for d in held for key in d['keys']};assert len(expected)==2874 and all('labels'not in d for d in held)
 scores=[json.loads(x)for x in scorepath.read_text().splitlines()];assert len(scores)==2874 and {tuple(x['key'])for x in scores}==expected
 positive={}
 for row in scores:
  key=tuple(row['key']);logit=float(row['logit']);prob=float(row['probability']);assert math.isfinite(logit) and 0<=prob<=1
  assert abs(prob-1/(1+math.exp(-max(-80.,min(80.,logit)))))<1e-7
  if logit>0:positive.setdefault(key[:2],set()).add(key[2])
 baseline=[json.loads(x)for x in Path(p['baseline']['path']).read_text().splitlines()]
 inp=[json.loads(x)for x in Path(p['input']['path']).read_text().splitlines()];by={str(x['pmc_id']):x for x in inp}
 assert len(baseline)==len(inp)==16 and sum(len(x['patient'])for x in inp)==39
 for d in baseline:
  pids=[str(x['patient_id'])for x in d['association']];assert len(pids)==len(set(pids)) and set(pids)=={str(x['patient_id'])for x in by[str(d['pmc_id'])]['patient']}
 prediction,added=assemble(baseline,positive)
 spec=importlib.util.spec_from_file_location('longcontext_relation_v2',p['evaluator']['path']);ev=importlib.util.module_from_spec(spec);sys.path.insert(0,str(Path(p['evaluator']['path']).parent));spec.loader.exec_module(ev)
 report=ev.validate_predictions(inp,prediction);assert report['valid'];save(out/'format.json',report)
 predpath=out/'prediction.jsonl'
 if added:predpath.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n'for x in prediction))
 else:predpath.write_bytes(Path(p['baseline']['path']).read_bytes())
 save(out/'added.json',added)
 save(out/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),prediction=ref(predpath),scores=ref(scorepath),source=ref(__file__),plan=ref(S/'quality-plan.json'),baseline=p['baseline'],gold_opened=False,cut=0.,single_primary=True))
 old=json.loads(Path(p['baseline_metrics']['path']).read_text())
 if added:
  save(out/'quality-start.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),new_v2=1,old_v2=0));assert ref(p['gold']['path'])['sha256']==p['gold']['sha256']
  metrics=ev.evaluate([json.loads(x)for x in Path(p['gold']['path']).read_text().splitlines()],prediction);calls=1
 else:assert ref(predpath)['sha256']==p['baseline']['sha256'];metrics=old;calls=0
 save(out/'metrics.json',metrics);save(out/'result.json',dict(status='COMPLETE_FIXED_FULL_CONTEXT_RELATION_UNION',baseline=old['metrics'],new=metrics['metrics'],delta={k:metrics['metrics'][k]-old['metrics'][k]for k in old['metrics']},retain=metrics['metrics']['Score']>old['metrics']['Score'],added=len(added),new_v2=calls,old_v2=0,documents=16,patients=39,all_non_relation_objects_unchanged=True))
 print((out/'result.json').read_text())
if __name__=='__main__':
 if '--contract'in sys.argv:contract()
 else:main()
