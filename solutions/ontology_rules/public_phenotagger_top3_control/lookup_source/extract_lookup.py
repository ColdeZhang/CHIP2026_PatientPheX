"""Bounded pure CPU extraction from pinned public-model raw caches, for all old-pool spans."""
import os,sys,json,hashlib,gzip,time,math,collections
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path(__file__).resolve().parent;started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
plan=json.loads((R/'plan.json').read_text());assert sha(__file__)==plan['extractor_sha256']
for key in ['input','labels','mapping','whitelist']:
 v=plan[key];assert sha(v['path'])==v['sha256'],key
assert plan['scope']=='train64_all_oldpool_spans_label_free_top3_lookup'
docs=[json.loads(x) for x in Path(plan['input']['path']).read_text().splitlines()];assert len(docs)==64 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs);byid={d['pmc_id']:d for d in docs};assert len(byid)==64
labels=Path(plan['labels']['path']).read_text().strip().splitlines();mapping=json.loads(Path(plan['mapping']['path']).read_text());assert len(labels)==mapping['columns']==17266 and labels[17265]=='HP:None';assert len(mapping['mapping'])==17266
bycol={}
for i,m in enumerate(mapping['mapping']):
 assert m['column']==i and m['original_label']==labels[i];bycol[i]=m['current_label']
keys=[];seen=set()
for line in Path(plan['whitelist']['path']).read_text().splitlines():
 r=json.loads(line);assert set(r)=={'pmc_id','offset','length'};key=(r['pmc_id'],r['offset'],r['length']);assert key not in seen and key[0] in byid and type(key[1]) is int and type(key[2]) is int and key[1]>=0 and key[2]>0;seen.add(key);keys.append(key)
assert len(keys)==plan['whitelist']['rows'] and len(keys)>0
pools={name:{} for name in plan['neural_raw']};summary={};full_span_sets={}
for model,files in plan['neural_raw'].items():
 allspans=set();linecounts=collections.Counter();kept_occurrences=0;parsed=0;verified=[]
 for source in files:
  digest=hashlib.sha256();n=0
  with Path(source['path']).open('rb') as f:
   for line in f:
    digest.update(line);r=json.loads(line);n+=1;parsed+=1
    key=(r['pmc_id'],r['offset'],r['length']);allspans.add(key);linecounts[r['pmc_id']]+=1
    d=byid[r['pmc_id']];p=d['full_text'][r['passage_index']];a,b=r['local_start'],r['local_end'];assert 0<=a<b<=len(p['text']) and r['text']==p['text'][a:b] and r['offset']==p['offset']+a and r['length']==b-a
    assert len(r['top3'])==3 and len({x['column'] for x in r['top3']})==3
    assert all(math.isfinite(x['probability']) and 0<=x['probability']<=1 and labels[x['column']]==x['hpo_id'] for x in r['top3'])
    assert all(r['top3'][i]['probability']>=r['top3'][i+1]['probability'] for i in range(2))
    if key not in seen:continue
    kept_occurrences+=1;entry=pools[model].setdefault(key,{'occurrences':0,'by_hpo':{},'passage_indices':set()});entry['occurrences']+=1;entry['passage_indices'].add(r['passage_index'])
    for rank,x in enumerate(r['top3'],1):
     current=bycol[x['column']]
     if current is None:continue
     target=entry['by_hpo'].setdefault(current,{'p':-1.0,'rank':4,'observations':set()});target['p']=max(target['p'],x['probability']);target['rank']=min(target['rank'],rank);target['observations'].add((x['column'],x['hpo_id'],x['probability'],rank))
  assert n==source['rows'] and digest.hexdigest()==source['sha256'] and Path(source['path']).stat().st_size==source['bytes'],source['path'];verified.append({'path':source['path'],'rows':n,'bytes':source['bytes'],'sha256':digest.hexdigest()})
 assert parsed==296984 and set(linecounts)==set(byid)
 full_span_sets[model]=allspans
 summary[model]={'raw_rows':parsed,'raw_distinct_spans':len(allspans),'whitelist_raw_present_spans':len(pools[model]),'whitelist_raw_missing_spans':len(keys)-len(pools[model]),'whitelist_kept_raw_occurrences':kept_occurrences,'current_id_entries':sum(len(v['by_hpo']) for v in pools[model].values()),'raw_present_all_top3_invalid_spans':sum(not v['by_hpo'] for v in pools[model].values()),'raw_files_verified':verified}
assert set(pools)=={'pubmed','bioformer'} and full_span_sets['pubmed']==full_span_sets['bioformer']
OUT=R/'output';OUT.mkdir(exist_ok=False);payload_digest=hashlib.sha256();uncompressed_bytes=0;linecount=0
with (OUT/'lookup.jsonl.gz').open('wb') as raw:
 with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as out:
  for key in sorted(keys):
   row={'pmc_id':key[0],'offset':key[1],'length':key[2]}
   for model in ['pubmed','bioformer']:
    entry=pools[model].get(key);row[model+'_raw_present']=entry is not None;row[model+'_raw_occurrences']=entry['occurrences'] if entry else 0;row[model]=[]
    if entry:
     for current,value in sorted(entry['by_hpo'].items(),key=lambda kv:(kv[1]['rank'],-kv[1]['p'],kv[0])):
      obs=[{'column':x[0],'raw_hpo_id':x[1],'p':x[2],'rank':x[3]} for x in sorted(value['observations'])]
      row[model].append({'hpo_id':current,'p':value['p'],'rank':value['rank'],'observed':True,'observations':obs})
   b=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write(b);payload_digest.update(b);uncompressed_bytes+=len(b);linecount+=1
assert linecount==len(keys);assert not any(name in sys.modules for name in ['torch','tensorflow','transformers'])
report={'status':'PASS_label_free_all_oldpool_span_top3_lookup','scope':plan['scope'],'plan_sha256':sha(R/'plan.json'),'source_sha256':sha(__file__),'input_sha256':plan['input']['sha256'],'whitelist_sha256':plan['whitelist']['sha256'],'whitelist_rows':len(keys),'whitelist_doc_count':len({k[0] for k in keys}),'model_statistics':summary,'full_raw_span_coordinate_sets_equal':True,'all_raw_coordinates_match_original_train64_input':True,'all_original_top3_columns_and_descending_actual_probabilities_checked':True,'current_mapping_sha256':plan['mapping']['sha256'],'label_order_sha256':plan['labels']['sha256'],'policy':'Two independent model lists; only observed top3 valid-current IDs; per same canonical ID take observed maxp and min original rank, retaining original column/rawID/probability/rank tuples. This is not a full canonical posterior. No threshold, candidate selection, labels or quality evaluation.','model_imported':False,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'seconds':time.monotonic()-started,'output':{'path':str(OUT/'lookup.jsonl.gz'),'sha256':sha(OUT/'lookup.jsonl.gz'),'bytes':(OUT/'lookup.jsonl.gz').stat().st_size,'jsonl_sha256':payload_digest.hexdigest(),'jsonl_bytes':uncompressed_bytes,'rows':linecount}}
dump(OUT/'summary.json',report);print(json.dumps(report),flush=True)
