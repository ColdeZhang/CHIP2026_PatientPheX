import sys,json,types,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
import local_predict as m, ahocorasick
rows=[json.loads(l) for l in (ROOT/'data/train/PatientPheX-train.jsonl').open()]
base=json.load((ROOT/'dict/hpo_dict.json').open())['surface2ids'];aug=json.load((ROOT/'dict/hpo_dict_augmented.json').open())['combined']
ABBREV=m.ABBREV.copy();STOP=m.FP_STOPWORDS.copy();NEG=m.NEG.copy();lookup=m.dict_lookup
out=Path('/tmp/patientphex_rule_ablations');out.mkdir(exist_ok=True)
def automaton(combined,abbr):
 A=ahocorasick.Automaton()
 for s,ids in combined.items():
  if len(s)>=3:A.add_word(s,(s,set(ids)))
 for s,h in abbr.items():A.add_word(s,(s,{h}))
 A.make_automaton();return A
A=automaton(aug,ABBREV)
def full_boundary(text,A):
 hits=lookup(text,A)
 return {(s,e):h for (s,e),h in hits.items() if not ((text[s-1] if s>0 else ' ').isalpha() or (text[e] if e<len(text) else ' ').isalpha())}
def longest(text,A):
 hits=lookup(text,A)
 return {(s,e):h for (s,e),h in hits.items() if not any(a<=s and e<=b and (a,b)!=(s,e) for a,b in hits)}
def stats(preds):
 totals=collections.Counter()
 for g,p in zip(rows,preds):
  gs={(e['offset'],e['length'],h) for e in g['entities'] if e['note']!='NO' for h in e['identifier'].split(';')}
  ps={(e['offset'],e['length'],h) for e in p['entities'] if e['note']!='NO' for h in e['identifier'].split(';')}
  ga={(a['patient_id'],h) for a in g['association'] for h in a['phenotype']};pa={(a['patient_id'],h) for a in p['association'] for h in a['phenotype']}
  totals['entity_tp']+=len(ps&gs);totals['entity_fp']+=len(ps-gs);totals['entity_fn']+=len(gs-ps)
  totals['association_tp']+=len(pa&ga);totals['association_fp']+=len(pa-ga);totals['association_fn']+=len(ga-pa)
  totals['entity_count']+=len(p['entities']);totals['association_count']+=len(pa)
 for task in ['entity','association']:
  tp,fp,fn=[totals[f'{task}_{k}'] for k in ['tp','fp','fn']]
  totals[f'{task}_f1']=round(2*tp/(2*tp+fp+fn),6)
 return dict(totals)
results={}
for variant in ['baseline','base_dictionary','full_boundary','longest_span','no_negation','no_blacklist','allow_two_chars','no_v10_new_abbr']:
 m.NEG=NEG.copy();m.FP_STOPWORDS=STOP.copy();m.dict_lookup=lookup;method=m.process_doc;aut=A
 if variant=='base_dictionary':aut=automaton(base,ABBREV)
 elif variant=='full_boundary':m.dict_lookup=full_boundary
 elif variant=='longest_span':m.dict_lookup=longest
 elif variant=='no_negation':m.NEG=[]
 elif variant=='no_blacklist':m.FP_STOPWORDS=set()
 elif variant=='allow_two_chars':
  mod=types.ModuleType('audit_short');exec((ROOT/'src/local_predict.py').read_text().replace('if len(stext) < 3:', 'if len(stext) < 2:'),mod.__dict__);method=mod.process_doc
 elif variant=='no_v10_new_abbr':aut=automaton(aug,{s:h for s,h in ABBREV.items() if s not in {'sp','mr','mh','oa','rp','rd','di'}})
 preds=[method(d,aut) for d in rows]
 (out/(variant+'.jsonl')).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in preds)+'\n')
 results[variant]=stats(preds)
print(json.dumps(results,ensure_ascii=False,indent=2))
