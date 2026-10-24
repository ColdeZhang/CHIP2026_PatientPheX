"""Read-only old candidate geometry mapped to already encoded native full contexts.

Only train32 labels are read. Held rows are the existing public-candidate/pure-input
preparation. No new tokenizer/model/quality calls or external candidate resources.
"""
import gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/biomedical-long-context-readiness-20261024'
OLD=R/'stage8-document-relations'
OUT=D/'relation-readiness'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def shared_window(a,b,windows,marker_only=False):
 return any(lo<=i<(i+1 if marker_only else j)<=hi and lo<=k<(k+1 if marker_only else m)<=hi for i,j in a for k,m in b for lo,hi in windows)
def q(values):
 values=sorted(values);return dict(min=values[0],p50=values[len(values)//2],p95=values[int(.95*(len(values)-1))],max=values[-1])
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 payload=json.loads(gzip.decompress((D/'probe-v1/data.json.gz').read_bytes()))
 native={str(x['pmc_id']):x for x in payload['native_contracts']}
 rawtrain=read(R/'ontology_rules/nested-outer0-inner1/train32.jsonl')
 rawheld=read('/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl')
 assert all('entities' not in x and 'association' not in x for x in rawheld)
 orig={str(x['pmc_id']):x for x in rawtrain+rawheld}
 all_new={};summary={};perdocs=[]
 for split in ('train','held'):
  prepared=read(OLD/'prepared'/f'{split}.jsonl');old=json.loads(gzip.decompress((OLD/'marker-encoded-v2'/f'{split}.json.gz').read_bytes()))
  oldmap={x['pmc_id']:x for x in old};newrows=defaultdict(list)
  for row in payload[split]:newrows[row['pmc_id']].append(row)
  counts=Counter();newencoded=[];maxpairs=0;pairdist=[]
  for doc in prepared:
   pmc=doc['pmc_id'];nc=native[pmc];source=orig[pmc];od=oldmap[pmc]
   assert doc['text']==nc['text']=='\n\n'.join(p['text'] for p in source['full_text'])
   assert doc['pairs']==od['pairs'] and doc['keys']==od['keys']
   assert ('labels' in doc)==(split=='train')
   passage_starts=[];cursor=0
   for p in source['full_text']:passage_starts.append(cursor);cursor+=len(p['text'])+2
   offsets={};tokenpass={};ids=[None]*len(nc['ids'])
   ws=[]
   for row in newrows[pmc]:
    lo,hi=row['body_range'];ws.append((lo,hi));assert row['ids'][1:-1]==nc['ids'][lo:hi]
    for i,t in enumerate(row['ids'][1:-1],lo):assert ids[i] in (None,t);ids[i]=t
    for b in row['blocks']:
     pi=b['passage_index'];base=passage_starts[pi]
     for i,(a,z) in zip(b['positions'],b['offsets']):
      off=(base+a,base+z);assert offsets.get(i,off)==off
      offsets[i]=off;tokenpass[i]=pi
   assert ids==nc['ids']
   entities=[];exactends=Counter();physical=set()
   for ei,e in enumerate(doc['entities']):
    spans=[];pis=set()
    for a,z in e['spans']:
     inds=[i for i,(x,y) in offsets.items() if x<z and a<y];assert inds
     i,j=min(inds),max(inds)+1
     assert all(offsets[t][0]<z and a<offsets[t][1] for t in inds)
     assert len({tokenpass[t] for t in inds})==1
     pi=tokenpass[i];pis.add(pi)
     # Same overlap-to-subtoken policy as old DocRE, not an output-span label.
     exact=offsets[i][0]==a and offsets[j-1][1]==z;exactends[e['kind']+'_exact_endpoint' if exact else e['kind']+'_overlap_endpoint']+=1
     owners=[(min(i-lo,hi-j),-w,w) for w,(lo,hi) in enumerate(ws) if lo<=i<j<=hi];assert owners
     owner=max(owners)[2];spans.append((i,j,owner,pi));physical.add((ei,a,z))
    entities.append(dict(kind=e['kind'],id=e['id'],spans=spans,passages=sorted(pis)))
   assert len(physical)==doc['unique_physical_mentions']
   oldsp=defaultdict(list);oldowner=defaultdict(set)
   for ei,a,z,i,j,owner in od['ownership']:oldsp[ei].append((i,j));oldowner[ei].add(owner)
   ows=[(w['start'],w['end']) for w in od['windows']]
   c=Counter(documents=1,pairs=len(doc['pairs']),patients=sum(e['kind']=='patient' for e in entities),concepts=sum(e['kind']=='phenotype' for e in entities),mentions=len(physical),new_windows=len(ws),old_marker_windows=len(ows),native_tokens=len(ids))
   c.update(exactends);maxpairs=max(maxpairs,len(doc['pairs']))
   if split=='train':c['gold_relations_all']=doc['training_gold_pairs']
   for ix,(p,h) in enumerate(doc['pairs']):
    tag=('positive' if doc['labels'][ix] else 'negative') if split=='train' else 'unlabelled'
    pe,he=entities[p],entities[h];ps=[(i,j) for i,j,w,pi in pe['spans']];hs=[(i,j) for i,j,w,pi in he['spans']]
    stats={'pairs':True,'old_same_owner_window':bool(oldowner[p]&oldowner[h]),
      'old_any_encoder_window_covers_both_complete_mentions':shared_window(oldsp[p],oldsp[h],ows),
      'old_any_encoder_window_covers_both_start_markers':shared_window(oldsp[p],oldsp[h],ows,True),
      'new_any_encoder_window_covers_both_complete_mentions':shared_window(ps,hs,ws),
      'same_source_passage':bool(set(pe['passages'])&set(he['passages'])),
      'multiple_patient_document':sum(e['kind']=='patient' for e in entities)>1}
    for key,value in stats.items():c[tag+'_'+key]+=int(value)
    if stats['new_any_encoder_window_covers_both_complete_mentions'] and not stats['old_any_encoder_window_covers_both_start_markers']:c[tag+'_new_direct_interaction_not_available_in_any_old_window']+=1
    if not stats['same_source_passage']:c[tag+'_only_cross_passage']+=1
    pairdist.append(min(abs(i-k) for i,j in ps for k,m in hs))
   # Representational ambiguity: same exact physical span set cannot distinguish IDs.
   for kind in ('patient','phenotype'):
    groups=defaultdict(list)
    for ei,e in enumerate(doc['entities']):
     if e['kind']==kind:groups[tuple(map(tuple,e['spans']))].append(ei)
    for members in groups.values():
     if len(members)<2:continue
     c[kind+'_identical_span_groups']+=1;c[kind+'_entities_in_identical_span_groups']+=len(members)
     if split=='train':
      targets={tuple(doc['labels'][k] for k,(p,h) in enumerate(doc['pairs']) if (p if kind=='patient' else h)==ei) for ei in members}
      c[kind+'_identical_span_groups_with_conflicting_targets']+=int(len(targets)>1)
   item={k:doc[k] for k in ('pmc_id','pairs','keys')};item.update(entities=entities,windows=[dict(row_id=x['row_id'],body_range=x['body_range'],ids=x['ids']) for x in newrows[pmc]])
   if split=='train':item['labels']=doc['labels']
   newencoded.append(item);counts.update(c);perdocs.append(dict(split=split,pmc_id=pmc,**dict(c)))
  assert counts['pairs']==(4495 if split=='train' else 2874)
  dest=OUT/(split+'-geometry.json.gz');dest.write_bytes(gzip.compress(json.dumps(newencoded,separators=(',',':')).encode(),mtime=0))
  summary[split]=dict(counts=counts,max_pairs_per_document=maxpairs,nearest_token_start_gap=q(pairdist),output=ref(dest))
 sources=[OLD/'prepared/train.jsonl',OLD/'prepared/held.jsonl',OLD/'marker-encoded-v2/train.json.gz',OLD/'marker-encoded-v2/held.json.gz',D/'probe-v1/data.json.gz',Path(__file__)]
 report=dict(schema='bioclinical-fullcontext-relation-supply-v1',status='CPU_READONLY_GEOMETRY_NO_MODELS_OR_HELD_GOLD',splits=summary,sources=[ref(p) for p in sources],model_calls=0,quality_calls=0,new_tokenizer_calls=0,held_gold_reads=0,
  interpretation=['Covisibility is a representational supply difference, not observed patient-attribution error or evidence labels.','Old full512 overlap windows covered every token; old own-window pooling and shared-support context differ from native token interaction.','Overlap endpoints retain all original DocRE candidates; no candidate deletion or new label.','Train32 F2F3 only; public frozen candidate models not project-fitted. Original prepared goldall includes unreachable/NIL, not all candidates.','New held article with2windows has complete token/mention coverage; no invented cross-window attention.'])
 (OUT/'census.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'per-document.json').write_text(json.dumps(perdocs,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
