"""Freeze train-only supervision occurrences from gold32 and fixed public cache."""
from collections import Counter,defaultdict
import argparse,gzip,hashlib,json,os,re,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(r):
 p=Path(r['path']);assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'];return p
def rows(r):return [json.loads(x) for x in checked(r).read_text().splitlines() if x.strip()]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main(inputs_path,policy_path,out):
 t=time.monotonic();sources=json.loads(Path(inputs_path).read_text());r=sources['refs'];policy=json.loads(Path(policy_path).read_text());out=Path(out);out.mkdir(exist_ok=False)
 assert policy['source']['sha256']==sha(__file__)
 mapping=json.loads(checked(r['column_mapping']).read_text());labels=checked(r['label_vocab']).read_text().splitlines();assert len(labels)==17266 and labels[17265]=='HP:None'
 inverse=defaultdict(list)
 for x in mapping['mapping']:
  assert labels[x['column']]==x['original_label']
  if x['current_label']:inverse[x['current_label']].append(x['column'])
 alias={};terms={};term=None
 for line in checked(r['hpo']).read_text().splitlines():
  if line=='[Term]':term={}
  elif line.startswith('['):term=None
  elif term is not None:
   if line.startswith('id: '):term['id']=line[4:];terms[term['id']]=term
   elif line.startswith('alt_id: '):alias[line[8:]]=term['id']
   elif line.startswith('is_a: '):term.setdefault('parents',[]).append(line[6:].split()[0])
   elif line=='is_obsolete: true':term['obsolete']=True
 valid={'HP:0000118'}
 while True:
  add={k for k,v in terms.items() if not v.get('obsolete') and any(p in valid for p in v.get('parents',[]))}
  if add<=valid:break
  valid|=add
 valid.remove('HP:0000118')
 docs=[]
 for f in [2,3]:
  gold=rows(r[f'fold{f}_gold']);inp=rows(r[f'fold{f}_input']);assert len(gold)==len(inp)==16
  assert [{k:v for k,v in x.items() if k not in ['entities','association']} for x in gold]==inp
  docs.extend(gold)
 ids={str(x['pmc_id']) for x in docs};assert len(ids)==32
 prior=json.loads(checked(r['train32_prior_preparation']).read_text());assert ids==set(prior['training_document_ids'])
 held=rows(r['held_F1_input']);reserved=rows(r['reserved_F0_input']);assert not ids&{str(x['pmc_id']) for x in held+reserved}
 manifest=json.loads(checked(r['public_manifest']).read_text());assert manifest['public_model_training_on_task'] is False
 cache=[x for x in rows(r['public_canonical']) if str(x['pmc_id']) in ids]
 provenance={(str(x['pmc_id']),int(x['offset']),int(x['length']),x['hpo_id']):x for x in rows(r['public_provenance']) if str(x['pmc_id']) in ids}
 groups=defaultdict(list)
 for c in cache:groups[str(c['pmc_id'])].append(c)
 events=[];counts=Counter();docout=[];goldspans={}
 def span(e):return(int(e['offset']),int(e['length']))
 def target(e):
  raw=str(e['identifier']);parts=raw.split(';')
  if raw=='-1':return None,'positive_NIL'
  if not all(re.fullmatch(r'HP:[0-9]{7}',v) for v in parts):return None,'positive_nonHPO_or_malformed'
  cs={alias.get(v,v) for v in parts}
  if not cs<=valid:return None,'positive_outside_valid_HPO_branch'
  if len(cs)!=1:return None,'positive_multiple_canonical_compound'
  c=next(iter(cs))
  if c not in inverse:return None,'positive_canonical_absent_original_head'
  return c,'positive_supported'
 for d in docs:
  did=str(d['pmc_id']);positive=[e for e in d['entities'] if str(e.get('note') or '').upper()!='NO'];no=[e for e in d['entities'] if str(e.get('note') or '').upper()=='NO'];ps={span(e) for e in positive};ns={span(e) for e in no};goldspans[did]=ps
  docout.append({k:d[k] for k in ['pmc_id','pmid','patient','full_text']})
  for e in d['entities']:
   isn=str(e.get('note') or '').upper()=='NO';c,kind=(None,'explicit_NO') if isn else target(e)
   columns=[17265] if isn else (inverse[c] if c else [])
   counts[kind]+=1;events.append(dict(event_id=len(events),pmc_id=did,offset=int(e['offset']),length=int(e['length']),text=e['text'],kind=kind,canonical_target=c,target_columns=columns,positive=not isn,source='official_train32_annotation'))
  seen=set()
  for c in groups[did]:
   k=span(c)
   if k in seen:continue
   same=[z for z in groups[did] if span(z)==k];seen.add(k);prov=[provenance[(did,int(z['offset']),int(z['length']),z['hpo_id'])] for z in same]
   if not any(x['raw_neural_match'] for x in prov):counts['public_dictionary_or_abbrev_only_span_excluded']+=1;continue
   if k in ps:counts['public_neural_exact_positive_span_protected']+=1;continue
   if k in ns:counts['public_neural_exact_NO_span_already_present']+=1;continue
   overlap=any(k[0]<o+l and o<k[0]+k[1] for o,l in ps);kind='boundary_negative' if overlap else 'candidate_nonentity_negative'
   counts[kind]+=1;events.append(dict(event_id=len(events),pmc_id=did,offset=k[0],length=k[1],text=c['text'],kind=kind,canonical_target=None,target_columns=[17265],positive=False,source='fixed_public_pubmed_native_cache',also_dictionary=any(x['raw_dictionary_match'] for x in prov),also_abbreviation=any(x['abbrev_inferred'] for x in prov)))
 request=dict(scope='train32_only_CPU_no_model',documents=docout,events=events,held_input_document_ids=[str(x['pmc_id']) for x in held],reserved_input_document_ids=[str(x['pmc_id']) for x in reserved],policy=policy,inverse_mapping={k:v for k,v in inverse.items()},label_order_sha256=mapping['original_order_sha256'])
 data=json.dumps(request,ensure_ascii=False,separators=(',',':')).encode();p=out/'request.json.gz'
 with p.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(data)
 write(out/'preparation.json',dict(status='LOCAL_TRAIN32_EVENTS_PREPARED_TOKEN_CENSUS_PENDING',source=ref(__file__),inputs=ref(inputs_path),policy=ref(policy_path),request=ref(p),documents=32,event_counts=dict(counts),events=len(events),public_native_canonical_records=len(cache),training_ids=sorted(ids),held_F1_all16_ids=[str(x['pmc_id']) for x in held],reserved_F0_all16_ids=[str(x['pmc_id']) for x in reserved],held_gold_read=False,trained_or_tokenized=False,seconds=time.monotonic()-t))
 print(json.dumps(dict(events=len(events),counts=counts,request=ref(p),seconds=time.monotonic()-t)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--policy',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.inputs,a.policy,a.out)
