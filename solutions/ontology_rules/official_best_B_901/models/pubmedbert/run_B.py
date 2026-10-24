"""One fixed public PhenoTagger hybrid inference; no gold/training/quality selection."""
import os,sys,time,json,hashlib,io
from pathlib import Path
R=Path.cwd();start=time.monotonic();prepare_only='--prepare-only' in sys.argv
OUT=R/('native-B901-cache-cpu-preparation' if prepare_only else 'native-B901-cache');OUT.mkdir(exist_ok=False)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==('' if prepare_only else 'GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1')
sys.path[:0]=[str(R),str(R/'native-pipeline/src')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
plan=json.loads((R/'B901-cache/plan.json').read_text())
for n,digest in plan['remote_source_sha256'].items():assert sha(R/n)==digest,(n,'source SHA mismatch')
assert sha(R/'B901-cache/B.input.jsonl')=='878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
assert sha(R/'pubmedbert_PT_v1.2.h5')=='d58af55a68840a24b052a775cf0cb32b8993a1fa4823c76aad10d3e6299d7ec3'
import nltk
nltk.data.path=[str(R/'nltk_data')]
from ssplit_tokenzier import ssplit_token_pos_lemma
from ml_ner import ml_intext_fn,decode_tsv,output_result
from dic_ner import dic_ont
from restore_index import restore_index_nest_fn
from combine_result import combine_ml_dict
from abbre_resolution import postprocess_abbr
import ml_ner
ns=dict(ml_ner.__dict__);exec(compile((R/'native-pipeline/fixed_candidate_function.py').read_text(),'fixed_candidate_function','exec'),ns);build=ns['build_ngram_testset_filted']
dic=dic_ont({'dic_file':str(R/'native-pipeline/dict/noabb_lemma.dic'),'word_hpo_file':str(R/'native-pipeline/dict/word_id_map.json'),'hpo_word_file':str(R/'native-pipeline/dict/id_word_map.json')})
docs=[json.loads(x) for x in (R/'B901-cache/B.input.jsonl').read_text().splitlines()];assert len(docs)==100
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
old_cache_path=R/'native-nested16/unique-raw-predictions.jsonl'
assert sha(old_cache_path)=='d3d813f40c17f637440c77f89803d7f9ae4f82175da070a9a26ba99e9dc7067f'
oldcache={}
with old_cache_path.open() as f:
 for line in f:
  row=json.loads(line);key=tuple(row['lower_tokens']);assert key not in oldcache;oldcache[key]=row
assert len(oldcache)==65406
additional_cache_path=R/'native-train48-cache/unique-raw-predictions.jsonl'
assert sha(additional_cache_path)=='270c1bd405c52ad24cfdeeca668962e96158dcfa0b50be4cd23dd86b80905e6f'
with additional_cache_path.open() as f:
 for line in f:
  row=json.loads(line);key=tuple(row['lower_tokens'])
  if key in oldcache:assert oldcache[key]['top3']==row['top3'] and oldcache[key]['native_line']==row['native_line']
  else:oldcache[key]=row
assert len(oldcache)==244938
cache=[];missing_indices=[]
for i,key in enumerate(unique):
 if key in oldcache:
  row=dict(oldcache[key]);row.update(unique_index=i,parent_unique_index=oldcache[key]['unique_index'],forward_origin='reused_frozen_public_forward');cache.append(row)
 else:cache.append(None);missing_indices.append(i)
expected_dictionary=sum(sum(max(0,len(s.splitlines())-1) for s in p['dictionary_tsv'].strip().split('\n\n')) for p in passages)
prep={'documents':100,'passages':len(passages),'verified_token_offsets':n_tokens,'candidates':n_candidates,'unique_lower_token_inputs':len(unique),'reused_forward_keys':len(unique)-len(missing_indices),'new_forward_keys':len(missing_indices),'dictionary_candidates':expected_dictionary,'seconds':time.monotonic()-start,'input_sha256':sha(R/'B901-cache/B.input.jsonl'),'reused_forward_sha256':sha(old_cache_path),'additional_reused_forward_sha256':sha(additional_cache_path)}
if not prepare_only:
 common_preparation_path=R.parent/'public-phenotagger-biobert/native-B901-census-cpu-preparation/preparation.json'
 assert sha(common_preparation_path)==plan['shared_CPU_census_sha256']
 prior=json.loads(common_preparation_path.read_text())
 for key in ['documents','passages','verified_token_offsets','candidates','unique_lower_token_inputs','reused_forward_keys','new_forward_keys','dictionary_candidates','input_sha256']:assert prep[key]==prior[key],key
dump('preparation.json',prep)
if prepare_only:
 print((OUT/'preparation.json').read_text(),flush=True);sys.exit(0)
# Original native head: the same complete H5 and tokenizer; no task-label optimization.
import numpy as np,tensorflow as tf
from transformers import AutoConfig,TFAutoModel
from nn_represent import BERT_RepresentationLayer
gpus=tf.config.list_physical_devices('GPU');assert len(gpus)==1;tf.config.experimental.set_memory_growth(gpus[0],True)
tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
assert str(R/'tf-gpu-overlay') in tf.__file__
labels=(R/'lable.vocab').read_text().strip().split('\n');assert len(labels)==17266
rep=BERT_RepresentationLayer(str(R/'pubmedbert-base'),str(R/'lable.vocab'),lowercase=True)
enc=TFAutoModel.from_config(AutoConfig.from_pretrained(R/'pubmedbert-base',local_files_only=True))
a=tf.keras.layers.Input((32,),dtype=tf.int32);b=tf.keras.layers.Input((32,),dtype=tf.int32);c=tf.keras.layers.Input((32,),dtype=tf.int32)
y=tf.keras.layers.Dense(17266,activation='softmax')(enc(a,token_type_ids=b,attention_mask=c)[0][:,0,:]);model=tf.keras.Model([a,b,c],y);model.load_weights(str(R/'pubmedbert_PT_v1.2.h5'));assert model.count_params()==122759794 and len(model.weights)==201
# Complete variable coverage was independently verified in the bound CPU/GPU strict-load reports.
infer=tf.function(lambda x,y,z:model([x,y,z],training=False));concrete=infer.get_concrete_function(*[tf.TensorSpec([None,32],tf.int32)]*3)
inferstart=time.monotonic()
with (OUT/'new-forward-predictions.jsonl').open('w') as out:
 for offset in range(0,len(missing_indices),128):
  batch_indices=missing_indices[offset:offset+128];batch=[unique[i] for i in batch_indices];inputs,_=rep.load_data([[tuple([w]) for w in key] for key in batch],[],word_max_len=32,training=False)
  probs=concrete(*inputs).numpy();assert probs.shape==(len(batch),17266) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
  native_lines=output_result(probs,rep.label_2_index,Top_N=3).strip().splitlines();assert len(native_lines)==len(batch)
  for j,(prob,line) in enumerate(zip(probs,native_lines)):
   raw=[]
   for cell in line.split('\t'):
    ident,rounded=cell.split('|');column=rep.label_2_index[ident];raw.append({'column':column,'hpo_id':ident,'probability':float(prob[column]),'native_probability_5dp':rounded})
   row={'unique_index':batch_indices[j],'lower_tokens':batch[j],'top3':raw,'native_line':line,'forward_origin':'new_public_forward'};cache[batch_indices[j]]=row;out.write(json.dumps(row,ensure_ascii=False)+'\n')
  if offset%4096==0:print('classified',min(offset+128,len(missing_indices)),'/',len(missing_indices),flush=True)
infer_seconds=time.monotonic()-inferstart;assert len(cache)==len(unique) and all(x is not None for x in cache)
with (OUT/'unique-raw-predictions.jsonl').open('w') as f:
 for row in cache:f.write(json.dumps(row,ensure_ascii=False)+'\n')
# Original hybrid composition (no longest-only filtering), then public abbreviation propagation.
headmap=json.loads((R/'original-column-to-current-HPO.json').read_text());dicmap=json.loads((R/'native-pipeline/dictionary-current-HPO.json').read_text())
canon={x['original_label']:x['current_label'] for x in headmap['mapping']}
for r in dicmap['mapping']:
 if r['original_id'] in canon:assert canon[r['original_id']]==r['current_id']
 canon[r['original_id']]=r['current_id']
counts={'raw_neural_candidates':0,'raw_dictionary_candidates':0,'hybrid_pre_abbreviation':0,'hybrid_after_abbreviation':0,'canonical_kept':0,'invalid_current_branch_dropped':0}
with (OUT/'raw-neural-candidates.jsonl').open('w') as rawout,(OUT/'raw-dictionary-candidates.jsonl').open('w') as dout,(OUT/'native-hybrid-stages.jsonl').open('w') as stages,(OUT/'canonical-entities.jsonl').open('w') as final:
 for p in passages:
  text=p['text'];scores=[]
  for x in p['candidates']:
   pred=cache[x['unique_index']];scores.append(pred['native_line']);left,right=x['local_start'],x['local_end']
   row={k:p[k] for k in ['pmc_id','passage_index','passage_offset','passage_type']};row.update(x);row.update(offset=p['passage_offset']+left,length=right-left,text=text[left:right],top3=pred['top3']);rawout.write(json.dumps(row,ensure_ascii=False)+'\n');counts['raw_neural_candidates']+=1
  ml_tsv=decode_tsv('\n'.join(scores),p['indices'],p['ml_text'],T=.95) if scores else '\n\n'.join(p['ml_text'].strip().splitlines())+'\n\n'
  dict_raw=restore_index_nest_fn(text,p['dictionary_tsv'])
  for x in dict_raw:
   left,right=int(x[0]),int(x[1]);assert 0<=left<right<=len(text)
   row={k:p[k] for k in ['pmc_id','passage_index','passage_offset','passage_type']};row.update(local_start=left,local_end=right,offset=p['passage_offset']+left,length=right-left,text=text[left:right],raw_hpo_id=x[2],score=float(x[3]));dout.write(json.dumps(row,ensure_ascii=False)+'\n');counts['raw_dictionary_candidates']+=1
  combined=combine_ml_dict(p['dictionary_tsv'],ml_tsv);before=restore_index_nest_fn(text,combined);after=postprocess_abbr(before,text)
  counts['hybrid_pre_abbreviation']+=len(before);counts['hybrid_after_abbreviation']+=len(after)
  stages.write(json.dumps({'pmc_id':p['pmc_id'],'passage_index':p['passage_index'],'before_abbreviation':before,'after_abbreviation':after},ensure_ascii=False)+'\n')
  for x in after:
   left,right=int(x[0]),int(x[1]);assert 0<=left<right<=len(text);target=canon.get(x[2])
   if target is None:counts['invalid_current_branch_dropped']+=1;continue
   row={k:p[k] for k in ['pmc_id','passage_index','passage_offset','passage_type']};row.update(local_start=left,local_end=right,offset=p['passage_offset']+left,length=right-left,text=text[left:right],raw_hpo_id=x[2],hpo_id=target,score=float(x[3]));final.write(json.dumps(row,ensure_ascii=False)+'\n');counts['canonical_kept']+=1
assert counts['raw_neural_candidates']==n_candidates and counts['raw_dictionary_candidates']==expected_dictionary
outputs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()}
manifest={'status':'complete_predictions_frozen_before_quality_evaluation','plan_sha256':sha(R/'B901-cache/plan.json'),'source_sha256':sha(__file__),'input_sha256':sha(R/'B901-cache/B.input.jsonl'),'model_weight_sha256':sha(R/'pubmedbert_PT_v1.2.h5'),'model_parameters':model.count_params(),'model_variables':len(model.weights),'strict_native_gpu_smoke_sha256':sha(R/'gpu-native-smoke.json'),'old_label_order_sha256':sha(R/'lable.vocab'),'public_model_training_on_task':False,'new_forward_keys':len(missing_indices),'reused_forward_keys':len(unique)-len(missing_indices),'cache_parent_sha256':sha(old_cache_path),'additional_cache_parent_sha256':sha(additional_cache_path),'fixed_policy':plan['policy'],'counts':counts,'inference_seconds':infer_seconds,'elapsed_seconds':time.monotonic()-start,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'memory_growth':True,'tf32_default_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'GPU_peak_memory':tf.config.experimental.get_memory_info('GPU:0'),'outputs':outputs}
dump('manifest.json',manifest);print(json.dumps(manifest),flush=True)
