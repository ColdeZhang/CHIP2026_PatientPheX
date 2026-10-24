"""Pure fixed Task1 candidate composition; no model loading, fit, predict or quality."""
import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

PUBLIC = [f'public_{s}' for s in ('present','score','neural','dictionary','abbreviation','origin_unknown')]
BIOFORMER = [s.replace('public_', 'bioformer_', 1) for s in PUBLIC]
BIOBERT = [s.replace('public_', 'biobert_', 1) for s in PUBLIC]
WEIGHTS = {
 'pubmed': 'd58af55a68840a24b052a775cf0cb32b8993a1fa4823c76aad10d3e6299d7ec3',
 'bioformer': '867819ac932e11250c42ca13c49eba3711a7f24b170e24bfb31b65589d0024a1',
 'biobert': '1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573',
}
SOURCES = {
 'single': ('dependencies/solutions/ontology_rules/calibrate.py', 'b7e18879673c889eec297c86a840349b23f8be5e6b3053c12ba58754091288d6'),
 'multi': ('dependencies/solutions/ontology_rules/calibrate_multi.py', '7301e9a7d3547e479dbdfb0dfe4475c7a26019e5d85e99f10d3ed0845858655d'),
 'pubmed': ('dependencies/legacy/select_public_candidates.py', '3378df59be8c13c45d07f9c17575523d58906837bba9c6b3f3cc814d3a2f7bd6'),
 'bioformer': ('dependencies/legacy/select_public_bioformer.py', '7cf106a6503cb515dd389b0664cee2bbf2591a61bbbe686d53ab1dccae73cb25'),
 'biobert': ('prepare_pool.py', '41872bde405defa212775ade97e394c3f4a097ececd110faa53e9d4f2cb2be37'),
 'hpo': ('dependencies/solutions/sequence_crf/run.py', '41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0'),
}
B_INPUT = '878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
B_CONTRACT = '2c1690660c58337235e9e11d96633109ac3ac3ffd62fa5e0600980c4b8a2ba21'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def ref(path):
 p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def checked(item):
 p=Path(item['path']);assert p.is_absolute() and sha(p)==item['sha256'],str(p)
 if 'bytes' in item:assert p.stat().st_size==item['bytes'],str(p)
 return p

def read(path):return json.loads(Path(path).read_text())
def rows(path):
 with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def keyed(data):
 result={r['pmc_id']:r for r in data};assert len(result)==len(data);return result

def functions(path,names,namespace):
 """Compile only selected original pure function definitions; never import model modules."""
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace

def load_pure_sources(root):
 root=Path(root);paths={}
 for role,(relative,digest) in SOURCES.items():
  path=root/relative;assert sha(path)==digest,(role,str(path));paths[role]=path
 single=functions(paths['single'],['pool'],{})
 multi=functions(paths['multi'],['pool'],dict(single=SimpleNamespace(pool=single['pool'])))
 pub=functions(paths['pubmed'],['key','merge_pool'],dict(PUBLIC_FEATURES=PUBLIC,math=math))
 bio=functions(paths['bioformer'],['merge_bioformer'],dict(key=pub['key'],PUBLIC_FEATURES=PUBLIC,BIOFORMER_FEATURES=BIOFORMER,math=math))
 bb=functions(paths['biobert'],['key','defaults','same_old','merge'],dict(FEATURES=BIOBERT,OLD_FEATURES=PUBLIC+BIOFORMER))
 hpo=functions(paths['hpo'],['parse_hpo'],dict(Path=Path,re=re))
 return multi['pool'],pub['merge_pool'],bio['merge_bioformer'],bb['merge'],hpo['parse_hpo'],{role:ref(p) for role,p in paths.items()}

def fullkey(c):return c['pmc_id'],c['offset'],c['length'],c.get('hpo_id',c.get('identifier'))

def public_candidates(spec,kind,docs,input_sha,allowed):
 cm=read(checked(spec['manifest']))
 assert cm['model_weight_sha256']==WEIGHTS[kind] and cm['public_model_training_on_task'] is False
 if 'model_type' in cm:assert cm['model_type']==kind
 assert cm['input_sha256']==input_sha and cm['counts']['documents']==len(docs)
 assert cm['input_ids']==list(docs)
 policy=cm['fixed_policy'];assert policy['ngram_per_sentence_min']==2 and policy['ngram_per_sentence_max']==8 and policy['threshold_strictly_greater_than']==.95
 for role,name in [('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]:
  checked(spec[role]);assert spec[role]['sha256']==cm['outputs'][name]['sha256']
  if 'bytes' in cm['outputs'][name]:assert Path(spec[role]['path']).stat().st_size==cm['outputs'][name]['bytes']
 provenance=rows(spec['provenance']['path']);ev={fullkey(r):r for r in provenance};assert len(ev)==len(provenance)
 for k,r in ev.items():
  assert k[0] in docs and k[3] in allowed
  flags=[r[n] for n in ('raw_neural_match','raw_dictionary_match','abbrev_inferred')]
  assert all(type(v) is bool for v in flags) and any(flags)
  assert all(type(v) is int and 0<=v<17266 for v in r['matched_neural_head_columns'])
  assert all(math.isfinite(v) and 0<=v<=1 for v in r['actual_hybrid_scores'])
 canonical=rows(spec['canonical']['path']);result={did:[] for did in docs};seen=set();group={};scores=defaultdict(set)
 for c in canonical:
  k=fullkey(c);e=ev[k];assert c['raw_hpo_id'] in e['raw_hpo_ids'] and c['score'] in e['actual_hybrid_scores']
  assert math.isfinite(c['score']) and 0<=c['score']<=1
  passage=docs[k[0]]['full_text'][c['passage_index']];lo,hi=c['local_start'],c['local_end']
  assert 0<=lo<hi<=len(passage['text']) and k[1]==passage['offset']+lo and k[2]==hi-lo and c['text']==passage['text'][lo:hi]
  seen.add(k);scores[k].add(c['score'])
  if kind=='biobert':
   if k not in group:group[k]=dict(pmc_id=k[0],offset=k[1],length=k[2],identifier=k[3],text=c['text'])
   assert group[k]['text']==c['text']
  else:
   prefix='public' if kind=='pubmed' else 'bioformer'
   row=dict(offset=c['offset'],length=c['length'],text=c['text'],identifier=c['hpo_id'])
   row.update({prefix+'_present':True,prefix+'_score':float(c['score']),prefix+'_neural':e['raw_neural_match'],prefix+'_dictionary':e['raw_dictionary_match'],prefix+'_abbreviation':e['abbrev_inferred'],prefix+'_origin_unknown':False,prefix+'_raw_hpo_ids':e['raw_hpo_ids'],prefix+'_matched_neural_head_columns':e['matched_neural_head_columns']})
   result[k[0]].append(row)
 assert seen==set(ev) and len(canonical)==cm['counts']['canonical_kept']
 for k,e in ev.items():assert sorted(scores[k])==e['actual_hybrid_scores']
 if kind=='biobert':
  for k,c in group.items():
   e=ev[k];c.update(biobert_present=True,biobert_score=float(max(scores[k])),biobert_neural=e['raw_neural_match'],biobert_dictionary=e['raw_dictionary_match'],biobert_abbreviation=e['abbrev_inferred'],biobert_origin_unknown=False);result[k[0]].append(c)
 return result,dict(canonical_rows=len(canonical),distinct_keys=len(ev),prepared_rows=sum(map(len,result.values())))

def raw_jsonl(data):return ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data).encode()
def write_json(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();p=read(a.plan)
 assert p['execution_ready'] is True and p['crf_confidence_policy']=='none_original_reference51c5'
 assert ref(__file__)['sha256']==p['source_sha256']
 for item in p.get('additional_sources',[]):checked(item)
 inputs=rows(checked(p['input']));docs=keyed(inputs);assert len(docs)==p['documents']
 assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in inputs)
 if p['mode']=='official_B':
  assert p['input']['sha256']==B_INPUT and p['documents']==100
  assert p['input_contract']['sha256']==B_CONTRACT;checked(p['input_contract'])
  assert p['train64_origin']['sha256']=='51c5b7199917aba56b893d45a04d7878e6df8454de0a1acb620372bc3a142813';checked(p['train64_origin'])
 else:assert p['mode']=='frozen_dev_input_only_reconstruction' and p['documents']==16 and p['input']['sha256']=='c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e'
 base=keyed(rows(checked(p['base'])));sap=keyed(rows(checked(p['sap_candidates'])));crf=keyed(rows(checked(p['crf'])))
 assert set(base)==set(sap)==set(crf)==set(docs)
 assert all(not {'crf_ner','crf_link'}&set(e) for d in crf.values() for e in d['entities'])
 oldpool,pubmerge,biomerge,bbmerge,parse_hpo,sources=load_pure_sources(p['fixed_source_root'])
 assert p['hpo']['sha256']=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b';_,allowed,_,_=parse_hpo(checked(p['hpo']));allowed.discard('HP:0000118')
 supply={};cache_counts={}
 for kind in ('pubmed','bioformer','biobert'):supply[kind],cache_counts[kind]=public_candidates(p['caches'][kind],kind,docs,p['input']['sha256'],allowed)
 stages={name:[] for name in ('base','pubmed','bioformer','input')}
 for did,doc in docs.items():
  original=oldpool(doc,base[did],sap[did]['candidates'],crf[did]);pub=pubmerge(original,supply['pubmed'][did]);bio=biomerge(pub,supply['bioformer'][did])
  for name,cs in [('base',original),('pubmed',pub),('bioformer',bio)]:stages[name].append(dict(pmc_id=did,input=doc,base=base[did],candidates=cs))
  stages['input'].append(bbmerge(stages['bioformer'][-1],supply['biobert'][did],doc))
 for values in stages.values():
  for row in values:
   cs=row['candidates'];assert len({(c['offset'],c['length'],c['identifier']) for c in cs})==len(cs)
   assert all('selector_probability' not in c for c in cs)
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
 outputs={}
 for name,values in stages.items():
  path=out/(name+'.pool.jsonl.gz');path.write_bytes(gzip.compress(raw_jsonl(values),mtime=0));outputs[path.name]=ref(path)
 for kind in ('pubmed','bioformer'):
  path=out/(kind+'.candidates.jsonl');path.write_bytes(raw_jsonl([dict(pmc_id=did,candidates=cs) for did,cs in supply[kind].items()]));outputs[path.name]=ref(path)
 comparisons={}
 for name,item in p.get('comparison_only',{}).items():
  expected=checked(item)
  if name=='final_pool':
   actual=out/'input.pool.jsonl.gz';actual_raw=gzip.decompress(actual.read_bytes());expected_raw=gzip.decompress(expected.read_bytes())
   comparisons[name]=dict(reference=item,compressed_byteexact=actual.read_bytes()==expected.read_bytes(),decompressed_byteexact=actual_raw==expected_raw,actual_uncompressed_sha256=hashlib.sha256(actual_raw).hexdigest(),expected_uncompressed_sha256=hashlib.sha256(expected_raw).hexdigest())
   assert comparisons[name]['decompressed_byteexact'],comparisons[name]
  elif name=='bioformer_scores':
   original=[dict(pmc_id=r['pmc_id'],candidates=[{k:v for k,v in c.items() if k!='selector_probability'} for c in r['candidates']]) for r in rows(expected)]
   actual=[dict(pmc_id=r['pmc_id'],candidates=r['candidates']) for r in stages['bioformer']]
   comparisons[name]=dict(reference=item,all_candidate_values_types_and_field_order_byteexact=raw_jsonl(actual)==raw_jsonl(original));assert comparisons[name]['all_candidate_values_types_and_field_order_byteexact']
  elif name in ('pubmed_prepared','bioformer_prepared'):
   kind=name.split('_')[0];old=[r for r in rows(expected) if r['pmc_id'] in docs];actual=[dict(pmc_id=did,candidates=cs) for did,cs in supply[kind].items()]
   comparisons[name]=dict(reference=item,dev_candidate_jsonl_byteexact=raw_jsonl(actual)==raw_jsonl(old));assert comparisons[name]['dev_candidate_jsonl_byteexact']
  else:raise AssertionError(name)
 result=dict(status='complete_pure_candidate_pool_no_model_fit_predict_or_quality',source=ref(__file__),plan=ref(a.plan),mode=p['mode'],documents=len(docs),input=p['input'],base=p['base'],sap_candidates=p['sap_candidates'],crf=p['crf'],crf_confidence_policy=p['crf_confidence_policy'],hpo=p['hpo'],caches=p['caches'],source_dependencies=sources,cache_counts=cache_counts,stage_candidate_counts={n:sum(len(r['candidates']) for r in v) for n,v in stages.items()},outputs=outputs,comparisons=comparisons,features='Frozen 85 recipe plus PubMed6/Bioformer6/BioBERT6; fixed vectorizer103 transform belongs to retrieval owner',candidate_order='Original Sap→base→CRF insertion order; then first observed canonical source order; merge exact(offset,length,HPO)',no_gold_read=True,no_model_loaded=True,no_fit=True,no_quality_evaluation=True,seconds=time.monotonic()-start)
 write_json(out/'manifest.json',result);print(json.dumps({k:result[k] for k in ('status','stage_candidate_counts','comparisons','seconds')}))

if __name__=='__main__':main()
