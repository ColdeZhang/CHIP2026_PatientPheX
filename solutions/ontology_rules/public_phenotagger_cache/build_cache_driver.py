from pathlib import Path
import ast,hashlib
R=Path(__file__).parent;old=R.parent/'public-phenotagger-reuse/native-pipeline/run_native_entities.py';s=old.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='e6dbdc720334786bbe1d97bd7cea321906cf5910ae8e999dda4f5ade812a1bbf'
s=s.replace("OUT=R/('native-nested16-cpu-preparation' if prepare_only else 'native-nested16')","OUT=R/('native-train48-cache-cpu-preparation' if prepare_only else 'native-train48-cache')")
s=s.replace("R/'native-pipeline/full-inference-plan.json'","R/'train64-cache/plan.json'")
s=s.replace("R/'nested16.input.jsonl'","R/'train64-cache/remaining48.input.jsonl'")
s=s.replace('962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415','b2256263e83810d4ed806d996cb3af1d5ff60c40c79e28bb6962472758e9e043')
s=s.replace('assert len(docs)==16','assert len(docs)==48')
a="assert (len(passages),n_tokens,n_candidates,len(unique))==(570,52185,75790,65406)\ndump('preparation.json',{'documents':16,'passages':len(passages),'verified_token_offsets':n_tokens,'candidates':n_candidates,'unique_lower_token_inputs':len(unique),'seconds':time.monotonic()-start,'input_sha256':sha(R/'train64-cache/remaining48.input.jsonl')})"
b="""assert len(passages)>0 and n_tokens>0 and n_candidates>0 and len(unique)>0
old_cache_path=R/'native-nested16/unique-raw-predictions.jsonl'
assert sha(old_cache_path)=='d3d813f40c17f637440c77f89803d7f9ae4f82175da070a9a26ba99e9dc7067f'
oldcache={}
with old_cache_path.open() as f:
 for line in f:
  row=json.loads(line);key=tuple(row['lower_tokens']);assert key not in oldcache;oldcache[key]=row
assert len(oldcache)==65406
cache=[];missing_indices=[]
for i,key in enumerate(unique):
 if key in oldcache:
  row=dict(oldcache[key]);row.update(unique_index=i,parent_unique_index=oldcache[key]['unique_index'],forward_origin='reused_frozen_public_forward');cache.append(row)
 else:cache.append(None);missing_indices.append(i)
expected_dictionary=sum(sum(max(0,len(s.splitlines())-1) for s in p['dictionary_tsv'].strip().split('\\n\\n')) for p in passages)
prep={'documents':48,'passages':len(passages),'verified_token_offsets':n_tokens,'candidates':n_candidates,'unique_lower_token_inputs':len(unique),'reused_forward_keys':len(unique)-len(missing_indices),'new_forward_keys':len(missing_indices),'dictionary_candidates':expected_dictionary,'seconds':time.monotonic()-start,'input_sha256':sha(R/'train64-cache/remaining48.input.jsonl'),'reused_forward_sha256':sha(old_cache_path)}
if not prepare_only:
 prior=json.loads((R/'native-train48-cache-cpu-preparation/preparation.json').read_text())
 for key in ['documents','passages','verified_token_offsets','candidates','unique_lower_token_inputs','reused_forward_keys','new_forward_keys','dictionary_candidates','input_sha256','reused_forward_sha256']:assert prep[key]==prior[key],key
dump('preparation.json',prep)"""
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace("cache=[];inferstart=time.monotonic()","inferstart=time.monotonic()")
s=s.replace("with (OUT/'unique-raw-predictions.jsonl').open('w') as out:","with (OUT/'new-forward-predictions.jsonl').open('w') as out:")
s=s.replace('for offset in range(0,len(unique),128):','for offset in range(0,len(missing_indices),128):')
s=s.replace("batch=unique[offset:offset+128];inputs,_=rep.load_data", "batch_indices=missing_indices[offset:offset+128];batch=[unique[i] for i in batch_indices];inputs,_=rep.load_data")
s=s.replace("row={'unique_index':offset+j,'lower_tokens':batch[j],'top3':raw,'native_line':line};cache.append(row);", "row={'unique_index':batch_indices[j],'lower_tokens':batch[j],'top3':raw,'native_line':line,'forward_origin':'new_public_forward'};cache[batch_indices[j]]=row;")
s=s.replace("min(offset+128,len(unique)),'/',len(unique)","min(offset+128,len(missing_indices)),'/',len(missing_indices)")
s=s.replace("infer_seconds=time.monotonic()-inferstart;assert len(cache)==65406", "infer_seconds=time.monotonic()-inferstart;assert len(cache)==len(unique) and all(x is not None for x in cache)\nwith (OUT/'unique-raw-predictions.jsonl').open('w') as f:\n for row in cache:f.write(json.dumps(row,ensure_ascii=False)+'\\n')")
s=s.replace("assert counts['raw_neural_candidates']==75790 and counts['raw_dictionary_candidates']==1174","assert counts['raw_neural_candidates']==n_candidates and counts['raw_dictionary_candidates']==expected_dictionary")
s=s.replace("'public_model_training_on_task':False,","'public_model_training_on_task':False,'new_forward_keys':len(missing_indices),'reused_forward_keys':len(unique)-len(missing_indices),'cache_parent_sha256':sha(old_cache_path),")
ast.parse(s);(R/'run_remaining48.py').write_text(s);print(hashlib.sha256(s.encode()).hexdigest())
