import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import local_predict as m
A=m.build_automaton(str(ROOT/'dict/hpo_dict_augmented.json'))
rows=[json.loads(l) for l in (ROOT/'data/train/PatientPheX-train.jsonl').open()]
count=collections.Counter(); examples=collections.defaultdict(list); surfaces=collections.defaultdict(collections.Counter)
assoc=collections.Counter(); assoc_examples=collections.defaultdict(list)
output=[]
def add(reason,surf,ids,sec,s,d,gold,ngold):
 k=(sec['offset']+s,len(surf)); good=bool(ids & gold.get(k,set())); neg=bool(ids&ngold.get(k,set()))
 count[reason]+=1;count[reason+'_gold_span']+=k in gold; count[reason+'_gold_id']+=good;count[reason+'_negated_gold_id']+=neg
 surfaces[reason][surf]+=1
 if len(examples[reason])<12 and (good or reason=='internal_boundary'):
  examples[reason].append({'pmc':d['pmc_id'],'surface':surf,'ids':sorted(ids),'gold':sorted(gold.get(k,set())),'context':sec['text'][max(0,s-45):s+len(surf)+45]})
for d in rows:
 gold=collections.defaultdict(set);ngold=collections.defaultdict(set)
 for e in d['entities']:
  (ngold if e['note']=='NO' else gold)[(e['offset'],e['length'])].update(e['identifier'].split(';'))
 pred=m.process_doc(d,A)
 output.append(pred)
 for sec in d['full_text']:
  text=sec['text'];lo=text.lower()
  for end,(surf,ids) in A.iter(lo):
   s=end-len(surf)+1
   if surf in m.FP_STOPWORDS:
    add('blacklist',surf,ids,sec,s,d,gold,ngold);continue
   if len(surf)<=4 and surf.isalpha() and ((lo[s-1] if s>0 else ' ').isalpha() or (lo[end+1] if end+1<len(lo) else ' ').isalpha()):continue
   if any(n in lo[max(0,s-30):s] for n in m.NEG):
    add('negation',surf,ids,sec,s,d,gold,ngold);continue
   if len(surf)<3:
    add('short_discard',surf,ids,sec,s,d,gold,ngold);continue
   if ((lo[s-1] if s>0 else ' ').isalpha() or (lo[end+1] if end+1<len(lo) else ' ').isalpha()):add('internal_boundary',surf,ids,sec,s,d,gold,ngold)
   if len(ids)>1:add('multi_id',surf,ids,sec,s,d,gold,ngold)
 ga={(a['patient_id'],h) for a in d['association'] for h in a['phenotype']}
 pa={(a['patient_id'],h) for a in pred['association'] for h in a['phenotype']}
 globalh={h for e in pred['entities'] for h in e['identifier'].split(';') if h.startswith('HP:')}
 provenance=collections.defaultdict(set)
 for sec in d['full_text']:
  start=sec['offset'];end=start+len(sec['text']);pids={p['patient_id'] for p in d['patient'] if any(start<=x['offset']<end for x in p['mention'])}
  for e in pred['entities']:
   if start<=e['offset']<end:
    for pid in pids:
     for h in e['identifier'].split(';'):
      if h.startswith('HP:'):provenance[(pid,h)].add('multi' if len(pids)>1 else 'single')
 for key in pa-ga:
  prov=provenance[key];assoc['fp_'+','.join(sorted(prov))]+=1
  if len(assoc_examples['fp_'+','.join(sorted(prov))])<4:assoc_examples['fp_'+','.join(sorted(prov))].append((d['pmc_id'],key))
 for key in ga-pa:assoc['fn_entity_absent' if key[1] not in globalh else 'fn_no_patient_cooccurrence']+=1
 assoc['tp']+=len(pa&ga);assoc['fp']+=len(pa-ga);assoc['fn']+=len(ga-pa)
 # Gold entity oracle: same-paragraph co-occurrence relation rule.
 oracle=set()
 for sec in d['full_text']:
  start=sec['offset'];end=start+len(sec['text']);pids={p['patient_id'] for p in d['patient'] if any(start<=x['offset']<end for x in p['mention'])}
  for e in d['entities']:
   if e['note']!='NO' and start<=e['offset']<end:
    for pid in pids:
     for h in e['identifier'].split(';'):
      if h.startswith('HP:'):oracle.add((pid,h))
 assoc['oracle_tp']+=len(oracle&ga);assoc['oracle_fp']+=len(oracle-ga);assoc['oracle_fn']+=len(ga-oracle)
 if len(assoc_examples['no_patient'])<6:
  for pid,h in ga-pa:
   if h in globalh:
    assoc_examples['no_patient'].append((d['pmc_id'],pid,h));break
 # False positive entities overlapping a gold entity span but not same boundaries.
 spans={(e['offset'],e['length']) for e in d['entities'] if e['note']!='NO'}
 for e in pred['entities']:
  s=e['offset'];n=e['length'];ids=set(e['identifier'].split(';'))
  if (s,n) not in spans:
   containing=[g for g in d['entities'] if g['note']!='NO' and g['offset']<=s and s+n<=g['offset']+g['length']]
   if containing:
    count['nested_inside_gold']+=1
    if len(examples['nested_inside_gold'])<10:examples['nested_inside_gold'].append((d['pmc_id'],e['text'],[(g['text'],g['identifier']) for g in containing]))
print(json.dumps({'counts':count,'surface_counts':{k:v.most_common(20) for k,v in surfaces.items()},'examples':examples,'association':assoc,'association_examples':assoc_examples},ensure_ascii=False,indent=2))
Path('/tmp/patientphex_rule_audit_train_predictions.jsonl').write_text('\n'.join(json.dumps(d,ensure_ascii=False) for d in output)+'\n')
