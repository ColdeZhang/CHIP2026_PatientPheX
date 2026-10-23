"""CPU-only reference: train-only distant supervision, never reads validation labels."""
import argparse,collections,hashlib,json,random,re
from pathlib import Path
WORD=re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
BOUNDARY=set('a an the of in on at for and or with without is was were be been has have had to from by as that this these those which whom who all any only no not yes both his her their its it we they he she our your while although because during after before then'.split())
SKIP={'REF','REFERENCES','BIBL','AUTH_CONT','COMP_INT','ACK_FUND'}
STOP_PUNCT=re.compile(r'[.,;:!?()\[\]{}\n]')
def norm(s):return ' '.join(s.lower().split())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hpo_aliases(path):
 assert 'data-version: hp/releases/2026-06-23' in Path(path).read_text().splitlines(), 'Use the designated HPO release'
 terms={};t=None
 for line in Path(path).read_text().splitlines():
  if line=='[Term]':t={'parents':[],'synonyms':[],'obsolete':False}
  elif line.startswith('['):t=None
  elif t is not None:
   if line.startswith('id: '):t['id']=line[4:];terms[line[4:]]=t
   elif line.startswith('name: '):t['name']=line[6:]
   elif line.startswith('is_a: '):t['parents'].append(line[6:].split()[0])
   elif line=='is_obsolete: true':t['obsolete']=True
   elif line.startswith('synonym: '):
    m=re.match(r'synonym: "((?:[^"\\]|\\.)*)" EXACT',line)
    if m:t['synonyms'].append(m.group(1).replace('\\"','"'))
 valid={'HP:0000118'}
 while True:
  nxt=valid|{h for h,v in terms.items() if not v['obsolete'] and any(p in valid for p in v['parents'])}
  if nxt==valid:break
  valid=nxt
 aliases=collections.defaultdict(set)
 for h in sorted(valid-{'HP:0000118'}):
  t=terms[h]
  if t['obsolete']:continue
  for s in [t.get('name','')]+t['synonyms']:
   z=norm(s)
   if 2<=len(z)<=200 and 1<=len(WORD.findall(z))<=16:aliases[z].add(h)
 return aliases

def windows(p):
 # No head condition, no learned rarity, no preemptive negation suppression.
 text=p['text'];tokens=list(WORD.finditer(text));base=p['offset']
 for i,m in enumerate(tokens):
  if m.group().lower() in BOUNDARY:continue
  for j in range(i,min(len(tokens),i+8)):
   if tokens[j].group().lower() in BOUNDARY:continue
   a,b=m.start(),tokens[j].end();s=text[a:b]
   if len(s)<3 or STOP_PUNCT.search(s):continue
   yield base+a,b-a,s

def local_context(p,start,length,radius=160):
 a=start-p['offset'];b=a+length
 # Left/right are always clipped relative to the candidate, so truncation can
 # retain the full candidate. No external sentence tokenizer is required.
 return p['text'][max(0,a-radius):min(len(p['text']),b+radius)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--out',required=True);ap.add_argument('--input');ap.add_argument('--base');ap.add_argument('--seed',type=int,default=20261023);ap.add_argument('--negative-per-doc',type=int,default=400);ap.add_argument('--materialize',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=random.Random(a.seed)
 docs=[json.loads(x) for x in Path(a.train).read_text().splitlines() if x.strip()]
 assert len({d['pmc_id'] for d in docs})==len(docs)
 aliases=hpo_aliases(a.hpo);vocab={w for s in aliases for w in WORD.findall(s) if w not in BOUNDARY}
 pos_surfaces={norm(e['text']) for d in docs for e in d['entities'] if str(e.get('note','')).upper()!='NO'}
 neg_surfaces={norm(e['text']) for d in docs for e in d['entities'] if str(e.get('note','')).upper()=='NO'}
 protected=set(aliases)|pos_surfaces|neg_surfaces
 rows=[];stats=collections.Counter();perdoc=[]
 for surface in sorted(aliases):
  rows.append({'source':'hpo_alias','span':surface,'context':'','lexical_y':1.,'occurrence_y':-1.,'weight':.2})
 stats['hpo_aliases']=len(aliases);stats['hpo_concepts']=len({h for hs in aliases.values() for h in hs})
 for d in docs:
  did=str(d['pmc_id']);counts=collections.Counter();passages=[p for p in d['full_text'] if str(p.get('section_type','')).upper() not in SKIP]
  forbidden=[(e['offset'],e['offset']+e['length']) for e in d['entities']]
  goldkeys={(e['offset'],e['length']) for e in d['entities'] if str(e.get('note','')).upper()!='NO'}
  observed=set();weak=[];dedup=set()
  for e in d['entities']:
   p=next((p for p in passages if p['offset']<=e['offset'] and e['offset']+e['length']<=p['offset']+len(p['text'])),None)
   if p is None:counts['gold_outside_kept_passages']+=1;continue
   is_no=str(e.get('note','')).upper()=='NO';key=(e['offset'],e['length'],is_no)
   if key in dedup:continue
   dedup.add(key);source='gold_no' if is_no else 'gold_positive';counts[source]+=1
   counts['unlinked_positive']+=not is_no and e.get('identifier')=='-1'
   rows.append({'source':source,'pmc_id':did,'offset':e['offset'],'length':e['length'],'span':e['text'],'context':local_context(p,e['offset'],e['length']),'lexical_y':1.,'occurrence_y':0. if is_no else 1.,'weight':1.})
  for p in passages:
   for off,length,s in windows(p):
    key=(off,length);observed.add(key);counts['raw_filtered_windows']+=1
    if any(off<b and a<off+length for a,b in forbidden):counts['excluded_any_gold_overlap']+=1;continue
    if norm(s) in protected:counts['excluded_known_phenotype_surface']+=1;continue
    words=[w for w in WORD.findall(norm(s)) if w not in BOUNDARY]
    support=sum(w in vocab for w in words)/max(1,len(words))
    weak.append({'source':'weak_hard' if support>=.5 else 'weak_random','pmc_id':did,'offset':off,'length':length,'span':s,'context':local_context(p,off,length),'lexical_y':0.,'occurrence_y':0.,'weight':.25})
  counts['positive_gold_boundaries_reachable']=len(goldkeys&observed);counts['positive_gold_boundaries_total']=len(goldkeys)
  # At most 50% lexical-hard; total bounded by negative-per-doc. These remain
  # weak negatives, never claimed to be human-verified negative spans.
  hard=[x for x in weak if x['source']=='weak_hard'];easy=[x for x in weak if x['source']=='weak_random'];rng.shuffle(hard);rng.shuffle(easy)
  selected=hard[:a.negative_per_doc//2]+easy[:a.negative_per_doc//2]
  remaining=hard[a.negative_per_doc//2:]+easy[a.negative_per_doc//2:];rng.shuffle(remaining);selected+=remaining[:a.negative_per_doc-len(selected)]
  counts['available_weak_negatives']=len(weak)
  for x in selected:counts[x['source']]+=1
  rows.extend(selected);stats.update(counts);perdoc.append({'pmc_id':did,**counts})
 stats['rows']=len(rows);stats['gold_pos_unique_surfaces']=len(pos_surfaces);stats['gold_no_unique_surfaces']=len(neg_surfaces);stats['surface_positive_NO_conflicts']=len(pos_surfaces&neg_surfaces);stats['surface_hpo_NO_conflicts']=len(set(aliases)&neg_surfaces)
 if a.materialize:
  target=out/'train-rows.jsonl'
  with target.open('w') as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 else:target=None
 report={'scope':'Only current training fold annotations and designated HPO; no heldout labels or input used. Weak negatives are incompletely annotated and downweighted.','source_sha256':sha(__file__),'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in [('train',a.train),('hpo',a.hpo)]},'configuration':vars(a),'summary':dict(stats),'per_document':perdoc,'source_counts':dict(collections.Counter(r['source'] for r in rows)),'output':{'path':str(target),'sha256':sha(target)} if target else None}
 (out/'data-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'summary':dict(stats),'source_counts':report['source_counts']},ensure_ascii=False))
 if a.input:
  assert a.base, 'Include existing base long spans without using input labels'
  inputs=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
  bases={str(d['pmc_id']):d for d in map(json.loads,Path(a.base).read_text().splitlines())}
  assert not {str(d['pmc_id']) for d in docs}&{str(d['pmc_id']) for d in inputs}
  assert all('entities' not in d and 'association' not in d for d in inputs)
  inference=[];counts={}
  for d in inputs:
   did=str(d['pmc_id']);cs={}
   for p in d['full_text']:
    if str(p.get('section_type','')).upper() in SKIP:continue
    for off,length,s in windows(p):
     cs[(off,length)]={'pmc_id':did,'offset':off,'length':length,'span':s,'context':local_context(p,off,length)}
   for e in bases[did]['entities']:
    p=next(p for p in d['full_text'] if p['offset']<=e['offset'] and e['offset']+e['length']<=p['offset']+len(p['text']))
    assert p['text'][e['offset']-p['offset']:e['offset']-p['offset']+e['length']]==e['text']
    cs[(e['offset'],e['length'])]={'pmc_id':did,'offset':e['offset'],'length':e['length'],'span':e['text'],'context':local_context(p,e['offset'],e['length'])}
   inference.extend(cs[k] for k in sorted(cs));counts[did]=len(cs)
  target=out/'input-rows.jsonl'
  target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inference))
  meta={'source_sha256':sha(__file__),'train_sha256':sha(a.train),'input_sha256':sha(a.input),'base_sha256':sha(a.base),'train_ids':sorted(str(d['pmc_id']) for d in docs),'input_ids':sorted(str(d['pmc_id']) for d in inputs),'counts':counts,'total':len(inference),'input_annotations_used':False,'output_sha256':sha(target),'candidate_policy':'unrestricted 1-8gram after function-word/punctuation filter, plus base long spans; no learned head/rarity/preemptive negation gate'}
  (out/'inference-summary.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'inference_total':len(inference)}))
if __name__=='__main__':main()
