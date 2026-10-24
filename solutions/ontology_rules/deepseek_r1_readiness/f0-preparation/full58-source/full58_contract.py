"""Fixed F0 input/cache contract; no model, gold or native tokenizer at import."""
from pathlib import Path
import gzip,hashlib,json
from strict_final import parse_final
from sparse_final import parse_sparse
from normalized_final import parse_normalized

COMPLETE='COMPLETE_FIXED58_RESPONSES_NO_QUALITY_EVALUATION'
PROMPTS_SHA='d387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0'
CPU_SHA='e97b2ba4232bf4f4ff826d8740ef5c086e6747efce35dc0d3fd71b26a95459de'
AUDIT_SHA='7649cfc300ef0fbd116c43a6c71db4839674ca64192e345934c127dc70f1052d'
ENCODED_SHA='9a132bedb7edfa8817b8d7daecc1e1df5a96ec7f9a3b63bb8cd2bd4ed64adfcd'

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def checked(record):
 assert isinstance(record,dict) and set(record)=={'path','bytes','sha256'}, 'Pending immutable file ref'
 path=Path(record['path']);assert path.is_absolute();data=path.read_bytes()
 assert len(data)==record['bytes'] and sha_bytes(data)==record['sha256'];return data

def same_bytes(left,right):assert all(left[k]==right[k] for k in ('bytes','sha256'))

def validate_plan(plan):
 assert plan['execution_ready'] is True, 'Source-only template cannot run'
 assert isinstance(plan['full_inference_authorization'],str) and plan['full_inference_authorization'].strip(), 'Full58 authorization pending'
 assert type(plan['process_budget_seconds'])is int and plan['process_budget_seconds']>plan['serialization_reserve_seconds']==30
 assert type(plan['TERM_grace_seconds'])is int and plan['TERM_grace_seconds']>=0
 assert plan['seed']==20261024 and plan['RNG_policy']=='fresh_same_seed_per_row_not_continuous_RNG'
 assert plan['max_new_tokens']==4096 and plan['temperature']==.6 and plan['top_p']==.95 and plan['top_k']==0
 assert plan['cached_indices'] in ([],[30]), 'Only explicitly selected original longest cache is eligible'
 cache=plan['cached_indices'];assert type(cache)is list
 assert isinstance(plan['cached_rows'],dict) and isinstance(plan['cached_provenance'],dict)
 assert set(plan['cached_rows'])==set(plan['cached_provenance'])=={str(i) for i in cache}
 assert plan['new_indices']==[i for i in range(58) if i not in cache]
 assert isinstance(plan['target_copy_ACK'],dict) and set(plan['target_copy_ACK'])=={'path','bytes','sha256'}, 'Target full9 copy ACK pending'
 assert plan['complete_status']==COMPLETE
 return cache

def parse_item(original,row):
 item=dict(original);content=item['final_text_excluding_terminal_EOS'];kwargs=dict(eos_observed=item['native_EOS_observed'],stopped_at_cap=item['stopped_at_token_cap'] or item['stopped_at_wall_budget'],code_map=row['code_map'],patient_ids=row['patient_ids'])
 for name,fn in [('strict_final',parse_final),('legal_sparse',parse_sparse),('normalized',parse_normalized)]:
  try:
   result=fn(content,**kwargs);ok=True
  except (ValueError,TypeError) as e:
   result=None;ok=False;item[name+'_invalid_reason']=str(e)
  key=name+'_valid'
  if key in original:assert original[key]==ok,'Reused parser verdict differs: '+key
  item[key]=ok
  if ok:
   if name=='strict_final':item['parsed_final']=result
   elif name=='legal_sparse':item.update(result)
   else:
    if 'normalized_evidence' in original:assert original['normalized_evidence']==result
    item['normalized_evidence']=result
 if not item['normalized_valid']:item.pop('normalized_evidence',None)
 item['coverage']={'provided_CIDs':list(item.get('normalized_evidence',{}).get('provided_mapping',{})),'abstained_CIDs':item.get('normalized_evidence',{}).get('abstained_CIDs',list(row['code_map'])),'invalid_whole_row_abstention':not item['normalized_valid'],'missing_CIDs_are_negative':False,'fallback_used':False}
 return item


def load_inputs(plan):
 cache=validate_plan(plan)
 raw=gzip.decompress(checked(plan['prompts_gzip']));assert len(raw)==823917 and sha_bytes(raw)==plan['prompts_uncompressed_sha256']==PROMPTS_SHA
 assert plan['CPU_result']['sha256']==CPU_SHA and plan['all58_audit']['sha256']==AUDIT_SHA and plan['all58_encoded']['sha256']==ENCODED_SHA
 rows=[json.loads(x) for x in raw.decode().splitlines()]
 encoded=[json.loads(x) for x in gzip.decompress(checked(plan['all58_encoded'])).decode().splitlines()]
 audits=json.loads(checked(plan['all58_audit']));cpu=json.loads(checked(plan['CPU_result']))
 assert cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL58_F0_INPUT_CPU_CONTRACT' and cpu['all58_raw_generic_native_IDs_exact'] and cpu['all58_roundtrip_exact']
 same_bytes(plan['all58_encoded'],cpu['outputs']['all58-published-encoded.jsonl.gz']);same_bytes(plan['all58_audit'],cpu['outputs']['all58-input-audit.json'])
 assert len(rows)==len(encoded)==len(audits)==58 and len(plan['row_ids'])==58
 seen=set();concepts=set();patients={};tokens=pairs=0
 for i,(r,e,a) in enumerate(zip(rows,encoded,audits)):
  assert r['row_id']==e['row_id']==a['row_id']==plan['row_ids'][i] and a['index']==i and r['pmc_id']==a['pmc_id']
  assert r['row_id'] not in seen;seen.add(r['row_id']);assert set(r)=={'row_id','pmc_id','code_map','patient_ids','messages'}
  assert [m['role'] for m in r['messages']]==['system','user'] and all(set(m)=={'role','content'} and type(m['content'])is str for m in r['messages'])
  assert sha_bytes(json.dumps(r,ensure_ascii=False,separators=(',',':')).encode())==a['original_row_sha256']
  assert sha_bytes(json.dumps(r['messages'],ensure_ascii=False,separators=(',',':')).encode())==a['original_messages_sha256']
  assert sha_bytes(json.dumps(e['input_ids'],separators=(',',':')).encode())==a['native_ids_sha256'] and len(e['input_ids'])==a['native_input_tokens']
  assert all(type(v)is int and 0<=v<151936 for v in e['input_ids']);tokens+=len(e['input_ids'])
  assert len(r['code_map'])==len(set(r['code_map'].values())) and list(r['code_map'])==['C%03d'%j for j in range(len(r['code_map']))]
  assert all(type(v)is str for v in r['code_map'].values()) and 1<=len(r['code_map'])<=8
  assert all(type(v)is str for v in r['patient_ids']) and len(r['patient_ids'])==len(set(r['patient_ids']))
  pmc=r['pmc_id'];assert type(pmc)is str and (pmc not in patients or patients[pmc]==r['patient_ids']);patients[pmc]=r['patient_ids']
  pairs+=len(r['code_map'])*len(r['patient_ids'])
  for value in r['code_map'].values():assert (pmc,value) not in concepts;concepts.add((pmc,value))
 assert len(patients)==16 and len(concepts)==415 and sum(map(len,patients.values()))==47 and pairs==1166 and tokens==195157
 assert max(range(58),key=lambda i:audits[i]['native_input_tokens'])==30 and audits[30]['native_input_tokens']==6028
 cached={}
 for key,record in plan['cached_rows'].items():
  i=int(key);assert i==30;item=json.loads(checked(record))
  assert item['row_id']==rows[i]['row_id'] and item['input_tokens']==audits[i]['native_input_tokens'] and item['original_index']==i
  ids=item['generated_ids'];assert item['generated_tokens']==len(ids)<=4096 and all(type(v)is int and 0<=v<151936 for v in ids)
  assert type(item['native_EOS_observed'])is bool and item['native_EOS_observed']==bool(ids and ids[-1]==151645)
  assert item['stopped_at_token_cap']==(not item['native_EOS_observed'] and len(ids)>=4096) and item['stopped_at_wall_budget'] is False and item['quality_evaluated'] is False
  assert item['input_ids_sha256']==audits[i]['native_ids_sha256']
  parse_item(item,rows[i]) # Expected invalid/cap may be an explicit immutable abstention; unknown error still stops.
  proof=plan['cached_provenance'][key];assert set(proof)=={'plan','result','wrapper','source','launcher','raw','ids'}
  saved={name:checked(r) for name,r in proof.items()}
  evidence=json.loads(saved['result']);oldplan=json.loads(saved['plan']);wrapper=json.loads(saved['wrapper'])
  assert evidence['status'] in ('PASS_PUBLISHED_NORMALIZED_F0_LONGEST_FORMAT_COST_ONLY_NOT_QUALITY','STOPPED_FIRST_TRUNCATED_OR_INVALID_FINAL_THIS_BUDGET_ONLY')
  assert wrapper['status']=='owned_probe_process_completed' and wrapper['returncode']==0 and wrapper['result_sha256']==proof['result']['sha256'] and wrapper['result_status']==evidence['status']
  assert wrapper['plan_sha256']==proof['plan']['sha256'] and wrapper['source_sha256']==proof['launcher']['sha256']
  assert wrapper['target_UUID_has_no_compute_process'] is True
  release=wrapper['actual_release'];assert [v.strip() for v in release['gpu'].split(',')][1]==oldplan['gpu_uuid']
  assert not any(line.split(',')[0].strip()==oldplan['gpu_uuid'] for line in release['compute_apps'].splitlines())
  assert evidence['quality'] is False and len(evidence['rows'])==1 and any(x['sha256']==record['sha256'] and x['row_id']==item['row_id'] for x in evidence['rows'])
  same_bytes(evidence['source'],proof['source']);same_bytes(evidence['plan'],proof['plan']);same_bytes(item['raw_reference'],proof['raw']);same_bytes(evidence['ID_evidence'],proof['ids'])
  raw_item=json.loads(saved['raw']);ids_item=json.loads(saved['ids'])
  assert all(item[k]==v for k,v in raw_item.items()) and all(raw_item[k]==v for k,v in ids_item.items())
  assert oldplan['seed']==20261024 and oldplan['max_new_tokens']==4096 and oldplan['temperature']==.6 and oldplan['top_p']==.95 and oldplan['top_k']==0
  assert oldplan['MS_revision']==plan['MS_revision'] and oldplan['model_dir']==plan['model_dir'] and oldplan['gpu_uuid']==plan['gpu_uuid']
  for name in ('CPU_result','derived_config','assets_result','expected_files','target_copy_ACK'):same_bytes(oldplan[name],plan[name])
  assert oldplan['sources']['runner.py']==proof['source']['sha256'] and oldplan['sources']['launcher.py']==proof['launcher']['sha256']
  assert oldplan['sources']['normalized_final.py']==plan['sources']['normalized_final.py']
  assert evidence['parameters']['count']==8190735360 and evidence['parameters']['tensors']==399 and evidence['parameters']['all_actual_shapes_dtype_finite'] is True
  assert not any(evidence['loading_information'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
  assert evidence['actual_rotary']['attention_scaling']==1.0 and evidence['runtime']['precision']=='native_BF16_SDPA_no_quantization'
  cached[i]=item
 return rows,encoded,audits,cached
