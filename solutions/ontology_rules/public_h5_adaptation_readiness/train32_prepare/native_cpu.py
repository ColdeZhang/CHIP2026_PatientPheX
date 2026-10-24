"""Exact native train32 token/candidate census. CPU only, no model or fit."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,ast,gzip,hashlib,io,json,signal,sys,time
from pathlib import Path
from collections import Counter,defaultdict

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main(entry_path):
 t=time.monotonic();signal.alarm(180);e=json.loads(Path(entry_path).read_text());root=Path(e['native_root']);out=Path(e['output_directory']);out.mkdir(exist_ok=False)
 for item in e['assets'].values():assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
 request_path=Path(e['request']['path']);assert sha(request_path)==e['request']['sha256'];request=json.loads(gzip.decompress(request_path.read_bytes()));policy=request['policy']
 assert len(request['documents'])==32 and policy['head_columns']==17266 and policy['None_column']==17265
 # Only tokenizer and exact native padding; no NN model/weights loaded.
 import numpy as np,nltk,transformers,tokenizers,tensorflow as tf
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 assert not tf.config.list_physical_devices('GPU')
 assert transformers.__version__=='4.30.1' and tokenizers.__version__=='0.13.3' and nltk.__version__=='3.8.1' and tf.__version__=='2.12.0'
 from transformers import AutoTokenizer
 from tensorflow.keras.preprocessing.sequence import pad_sequences
 nltk.data.path=[str(root/'nltk_data')];sys.path[:0]=[str(root),str(root/'native-pipeline/src')]
 from ssplit_tokenzier import ssplit_token_pos_lemma
 from restore_index import restore_index_nest_fn
 import ml_ner
 env=dict(ml_ner.__dict__);exec(compile(Path(e['assets']['fixed_candidate_builder']['path']).read_text(),'frozen_candidate_builder','exec'),env);build=env['build_ngram_testset_filted']
 source=Path(e['assets']['native_representation']['path']);tree=ast.parse(source.read_text());cls=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BERT_RepresentationLayer'];assert len(cls)==1
 scope=dict(np=np,AutoTokenizer=AutoTokenizer,pad_sequences=pad_sequences);exec(compile(ast.Module(body=cls,type_ignores=[]),str(source),'exec'),scope)
 rep=scope['BERT_RepresentationLayer'](str(root/'pubmedbert-base'),str(root/'lable.vocab'),lowercase=True);assert rep.label_table_size==17266 and rep.label_2_index['HP:None']==17265
 bydoc=defaultdict(list)
 for event in request['events']:bydoc[str(event['pmc_id'])].append(event)
 occurrences=[];totals=Counter();unique={};lowerkeys=[]
 for doc in request['documents']:
  did=str(doc['pmc_id']);evs=bydoc[did];resolved=set()
  for pi,p in enumerate(doc['full_text']):
   text=p['text'];offset=int(p['offset']);events=[v for v in evs if offset<=v['offset'] and v['offset']+v['length']<=offset+len(text)]
   if not events:continue
   if not text.strip():continue
   conll=ssplit_token_pos_lemma(text)
   if not conll.strip():continue
   totals['processed_passages']+=1
   sents=[x.splitlines() for x in conll.strip().split('\n\n')];annotation='';wanted=[]
   for lines in sents:
    words=[l.split('\t')[0] for l in lines];annotation+=' '.join(words)+'\n'
    for j,w in enumerate(words):annotation+=str(j)+'\t'+str(j)+'\t'+w+'\tHP:None\t0.0\n';wanted.append(w)
    annotation+='\n'
   loc=restore_index_nest_fn(text,annotation);assert len(loc)==len(wanted)
   norm=lambda s:s.lower().replace('``','"').replace("''",'"')
   for w,x in zip(wanted,loc):
    a,b=map(int,x[:2]);assert 0<=a<b<=len(text) and norm(text[a:b])==norm(w)
   bysentence=[];at=0;starts={};ends={}
   for si,lines in enumerate(sents):
    ls=loc[at:at+len(lines)];at+=len(lines);bysentence.append(ls)
    for j,x in enumerate(ls):starts[int(x[0])]=(si,j);ends[int(x[1])]=(si,j)
   ml_input,ml_text,indices=build(conll);instances,_=ml_ner.ml_intext_fn(ml_input);instances=[x for x in instances if x];assert len(instances)==len(indices)
   native_spans={}
   for j,inst in enumerate(instances):
    si,a,b=indices[str(j+1)];left=int(bysentence[si][a][0]);right=int(bysentence[si][b][1]);native_spans[(offset+left,right-left)]=tuple(x[0].lower() for x in inst)
   totals['native_candidates_in_event_passages']+=len(instances)
   for event in events:
    assert event['event_id'] not in resolved,'passage overlap for training event';resolved.add(event['event_id']);x=dict(event);a=event['offset']-offset;b=a+event['length'];x['passage_index']=pi
    if text[a:b]!=event['text']:
     x['input_status']='annotation_text_offset_mismatch';occurrences.append(x);continue
    if a not in starts or b not in ends or starts[a][0]!=ends[b][0]:
     x['input_status']='not_exact_native_token_span';occurrences.append(x);continue
    si,first=starts[a];_,last=ends[b];assert last>=first
    words=tuple(sents[si][j].split('\t')[0].lower() for j in range(first,last+1));native=native_spans.get((event['offset'],event['length']))
    if native is not None:assert native==words
    x['native_candidate_eligible']=native is not None;x['input_status']='native_candidate' if native is not None else 'outside_native_2to8gram_or_POS_filter'
    if words not in unique:unique[words]=len(lowerkeys);lowerkeys.append(words)
    x['lower_query_index']=unique[words];occurrences.append(x)
  for event in evs:
   if event['event_id'] not in resolved:occurrences.append(dict(event,input_status='not_contained_in_one_passage'))
 assert len(occurrences)==len(request['events'])
 enc=[];full_lengths=[];tokens_started=time.monotonic()
 for off in range(0,len(lowerkeys),256):
  batch=lowerkeys[off:off+256];arrays,_=rep.load_data([[tuple([w]) for w in words] for words in batch],[],word_max_len=32,training=False)
  assert all(x.shape==(len(batch),32) for x in arrays)
  for j,words in enumerate(batch):
   full=rep.tokenizer(list(words),truncation=False,is_split_into_words=True);full_lengths.append(len(full['input_ids']))
   enc.append(tuple(tuple(int(v) for v in arr[j]) for arr in arrays))
 group_for={};groups=[]
 for j,z in enumerate(enc):
  if z not in group_for:group_for[z]=len(groups);groups.append(dict(query_index=len(groups),encoded=z,lower_query_indices=[],events=[]))
  groups[group_for[z]]['lower_query_indices'].append(j)
 for x in occurrences:
  if 'lower_query_index' in x:groups[group_for[enc[x['lower_query_index']]]]['events'].append(x)
 conflicts=Counter();eligible=[];group_status=Counter()
 for g in groups:
  evs=g['events'];positives=[x for x in evs if x['positive']];negatives=[x for x in evs if not x['positive']];supported={x['canonical_target'] for x in positives if x['kind']=='positive_supported'}
  if positives and negatives:status='excluded_positive_negative_same_actual_input';conflicts[status]+=1
  elif len(supported)>1:status='excluded_multiple_positive_canonical_same_actual_input';conflicts[status]+=1
  elif any(x['kind']!='positive_supported' for x in positives):status='excluded_unsupported_positive_protection';conflicts[status]+=1
  elif not any(x.get('native_candidate_eligible',False) for x in evs):status='excluded_no_native_candidate_occurrence'
  else:
   status='eligible_before_fixed_negative_cap';positive=bool(positives)
   if positive:
    assert len(supported)==1;target=next(iter(supported));columns=request['inverse_mapping'][target];category='positive'
   else:
    target=None;columns=[17265];kinds={x['kind'] for x in negatives};category=next(k for k in ['explicit_NO','boundary_negative','candidate_nonentity_negative'] if k in kinds)
   token_sha=hashlib.sha256(np.asarray(g['encoded'],dtype='<i4').tobytes()).hexdigest()
   eligible.append(dict(query_index=g['query_index'],input_ids=list(g['encoded'][0]),token_type_ids=list(g['encoded'][1]),attention_mask=list(g['encoded'][2]),target_columns=columns,canonical_target=target,category=category,token_sha256=token_sha,weight=1.,occurrence_count=len(evs),training_document_ids=sorted({x['pmc_id'] for x in evs}),truncated_any=any(full_lengths[j]>32 for j in g['lower_query_indices'])))
  g['status']=status;group_status[status]+=1
 positives=[x for x in eligible if x['category']=='positive'];n=len(positives);assert n>0
 selected=list(positives);selection_counts={}
 for category in ['explicit_NO','boundary_negative','candidate_nonentity_negative']:
  candidates=[x for x in eligible if x['category']==category]
  candidates.sort(key=lambda x:hashlib.sha256((str(policy['seed'])+'|'+category+'|'+x['token_sha256']).encode()).hexdigest())
  cap=len(candidates) if category=='explicit_NO' else n
  chosen=candidates[:cap];selected+=chosen;selection_counts[category]=dict(eligible=len(candidates),cap=cap,selected=len(chosen))
 selected.sort(key=lambda x:x['token_sha256']);assert len({x['token_sha256'] for x in selected})==len(selected)
 dataset=out/'training-queries.jsonl.gz'
 with dataset.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
   for x in selected:f.write((json.dumps(x,ensure_ascii=False)+'\n').encode())
 # Full input-only audit outside Git, with no held-label content.
 audit=out/'occurrence-audit.json.gz'
 with audit.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(json.dumps(dict(occurrences=occurrences,groups=[{k:v for k,v in g.items() if k not in ['encoded','events']} for g in groups],full_untruncated_lengths=full_lengths),ensure_ascii=False).encode())
 runtime=dict(python=sys.executable,python_version=sys.version.split()[0],transformers=transformers.__version__,tokenizers=tokenizers.__version__,nltk=nltk.__version__,tensorflow=tf.__version__,numpy=np.__version__,GPU_visible=[],CUDA_VISIBLE_DEVICES='',weights_loaded=False,model_instantiated=False,training_or_fit=False)
 result=dict(status='COMPLETE_NATIVE_CPU_TRAIN32_PREPARATION',source=ref(__file__),entry=ref(entry_path),request=e['request'],runtime=runtime,training_documents=32,held_F1_documents_preserved=16,held_gold_read=False,raw_event_counts=dict(Counter(x['kind'] for x in occurrences)),input_status_by_kind=[dict(kind=k,status=s,count=v) for (k,s),v in sorted(Counter((x['kind'],x['input_status']) for x in occurrences).items())],native_census=dict(totals),lower_queries=len(lowerkeys),actual_encoded32_queries=len(groups),different_lower_queries_collapsed_by_token32=len(lowerkeys)-len(groups),queries_truncated=sum(x>32 for x in full_lengths),max_untruncated_tokens=max(full_lengths),tokenization_seconds=time.monotonic()-tokens_started,group_status=dict(group_status),conflicts=dict(conflicts),eligible_positive_queries=n,positive_multi_original_column_queries=sum(len(x['target_columns'])>1 for x in positives),negative_selection=selection_counts,training_queries=len(selected),training_categories=dict(Counter(x['category'] for x in selected)),selected_truncated_queries=sum(x['truncated_any'] for x in selected),single_pass_batch64_steps=(len(selected)+63)//64,dataset=ref(dataset),audit=ref(audit),seconds=time.monotonic()-t)
 write(out/'result.json',result);signal.alarm(0);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);main(p.parse_args().entry)
