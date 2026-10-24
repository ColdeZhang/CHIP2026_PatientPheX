"""One fixed public PhenoTagger hybrid inference; no gold/training/quality selection."""
import os,sys,time,json,hashlib,io
from pathlib import Path
R=Path.cwd();SHARED=R.parent/'public-phenotagger-reuse';start=time.monotonic();prepare_only='--prepare-only' in sys.argv
OUT=R/('native-B901-census-cpu-preparation' if prepare_only else 'native-B901-census');OUT.mkdir(exist_ok=False)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==('' if prepare_only else 'GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1')
sys.path[:0]=[str(R),str(SHARED/'native-pipeline/src')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
plan=json.loads((R/'B901-census/plan.json').read_text())
for n,digest in plan['remote_source_sha256'].items():assert sha(R/n)==digest,(n,'source SHA mismatch')
assert sha(R/'B901-census/B.input.jsonl')=='878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
assert sha(R/'biobert_PT_v1.2.h5')=='1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573'
import nltk
nltk.data.path=[str(SHARED/'nltk_data')]
from ssplit_tokenzier import ssplit_token_pos_lemma
from ml_ner import ml_intext_fn,decode_tsv,output_result
from dic_ner import dic_ont
from restore_index import restore_index_nest_fn
from combine_result import combine_ml_dict
from abbre_resolution import postprocess_abbr
import ml_ner
ns=dict(ml_ner.__dict__);exec(compile((SHARED/'native-pipeline/fixed_candidate_function.py').read_text(),'fixed_candidate_function','exec'),ns);build=ns['build_ngram_testset_filted']
dic=dic_ont({'dic_file':str(SHARED/'native-pipeline/dict/noabb_lemma.dic'),'word_hpo_file':str(SHARED/'native-pipeline/dict/word_id_map.json'),'hpo_word_file':str(SHARED/'native-pipeline/dict/id_word_map.json')})
docs=[json.loads(x) for x in (R/'B901-census/B.input.jsonl').read_text().splitlines()];assert len(docs)==100
unique=[];uid={};passages=[];n_candidates=n_tokens=0
for doc in docs:
 assert set(doc)== {'pmc_id','pmid','patient','full_text'}
 for pi,p in enumerate(doc['full_text']):
  text=p['text']
  if not text.strip():continue
  conll=ssplit_token_pos_lemma(text)
  if not conll.strip():continue
  ml_input,ml_text,indices=build(conll);instances,_=ml_intext_fn(ml_input);instances=[x for x in instances if x];assert len(instances)==len(indices)
  sents=[x.splitlines() for x in conll.strip().split('\n\n')];annotation='';wanted=[]
  for lines in sents:
   words=[l.split('\t')[0] for l in lines];annotation+=' '.join(words)+'\n'
   for j,w in enumerate(words):annotation+=str(j)+'\t'+str(j)+'\t'+w+'\tHP:None\t0.0\n';wanted.append(w)
   annotation+='\n'
  locations=restore_index_nest_fn(text,annotation);assert len(locations)==len(wanted)
  norm=lambda s:s.lower().replace('``','"').replace("''",'"')
  normalized_source=text.lower().replace('``','" ').replace("''",'" ');assert len(normalized_source)==len(text)
  for w,loc in zip(wanted,locations):
   a,b=map(int,loc[:2]);assert 0<=a<b<=len(text) and normalized_source[a:b]==norm(w),(doc['pmc_id'],pi,a,b,w)
  bysentence=[];at=0
  for lines in sents:bysentence.append(locations[at:at+len(lines)]);at+=len(lines)
  candidates=[]
  for i,inst in enumerate(instances):
   key=tuple(x[0].lower() for x in inst)
   if key not in uid:uid[key]=len(unique);unique.append(key)
   si,a,b=indices[str(i+1)];left=int(bysentence[si][a][0]);right=int(bysentence[si][b][1]);assert 0<=left<right<=len(text)
   candidates.append({'unique_index':uid[key],'sentence':si,'word_start':a,'word_end_inclusive':b,'local_start':left,'local_end':right})
  passages.append({'pmc_id':doc['pmc_id'],'passage_index':pi,'passage_offset':p['offset'],'passage_type':p['type'],'text':text,'candidates':candidates,'ml_text':ml_text,'indices':indices,'dictionary_tsv':dic.matching(conll)})
  n_candidates+=len(candidates);n_tokens+=len(locations)
assert len(passages)>0 and n_tokens>0 and n_candidates>0 and len(unique)>0
# Reuse only exact lower-token tuples from this same BioBERT model and pinned representation.
parent_manifest_path=R/'native-train64-cache/manifest.json'
assert sha(parent_manifest_path)=='26ca138da95bf6dc1610ad2bb942c442556cf7ac770e066016bc0a022da6f695'
parent_manifest=json.loads(parent_manifest_path.read_text());assert parent_manifest['model_type']=='biobert' and parent_manifest['model_weight_sha256']==plan['model_weight_sha256'] and parent_manifest['fixed_policy']==plan['policy']
old_cache_path=R/'native-train64-cache/unique-raw-predictions.jsonl';assert sha(old_cache_path)=='6b41083473cc49016671b24d29b9b556899bda667c6ca0d04490455bfdc49139'
oldcache={}
with old_cache_path.open() as f:
 for line in f:
  row=json.loads(line);key=tuple(row['lower_tokens']);assert key not in oldcache;oldcache[key]=row
assert len(oldcache)==244938
cache=[];missing_indices=[]
for i,key in enumerate(unique):
 if key in oldcache:
  row=dict(oldcache[key]);row.update(unique_index=i,parent_unique_index=oldcache[key]['unique_index'],forward_origin='reused_same_BioBERT_train64_forward');cache.append(row)
 else:cache.append(None);missing_indices.append(i)
assert len(cache)==len(unique)
expected_dictionary=sum(sum(max(0,len(s.splitlines())-1) for s in p['dictionary_tsv'].strip().split('\n\n')) for p in passages)
prep={'documents':100,'passages':len(passages),'verified_token_offsets':n_tokens,'candidates':n_candidates,'unique_lower_token_inputs':len(unique),'reused_forward_keys':len(unique)-len(missing_indices),'new_forward_keys':len(missing_indices),'reused_forward_sha256':sha(old_cache_path),'reused_forward_parent_manifest_sha256':sha(parent_manifest_path),'dictionary_candidates':expected_dictionary,'seconds':time.monotonic()-start,'input_sha256':sha(R/'B901-census/B.input.jsonl')}
for key,value in plan['expected_input_only_counts'].items():assert prep[key]==value,(key,prep[key],value)
if not prepare_only:
 prior=json.loads((R/'native-B901-census-cpu-preparation/preparation.json').read_text())
 for key in ['documents','passages','verified_token_offsets','candidates','unique_lower_token_inputs','reused_forward_keys','new_forward_keys','dictionary_candidates','input_sha256']:assert prep[key]==prior[key],key
dump('preparation.json',prep)
if prepare_only:
 print((OUT/'preparation.json').read_text(),flush=True);sys.exit(0)
raise AssertionError('CPU census has no model path')
