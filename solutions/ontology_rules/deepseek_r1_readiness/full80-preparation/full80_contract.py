"""Actual inference input/cache/schema preflight, standard library only; no model/gold."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,time
from strict_final import parse_final
from sparse_final import parse_sparse
from normalized_final import parse_normalized,tests

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def checked(r):
 p=Path(r['path']);b=p.read_bytes();assert len(b)==r['bytes'] and sha_bytes(b)==r['sha256'];return b

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
 assert plan['seed']==20261024 and plan['RNG_policy']=='fresh_same_seed_per_row_not_continuous_RNG'
 assert plan['process_budget_seconds']==10800 and plan['TERM_grace_seconds']==30 and plan['max_new_tokens']==4096
 b=gzip.decompress(checked(plan['prompts_gzip']));assert sha_bytes(b)==plan['prompts_uncompressed_sha256']=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
 rows=[json.loads(x) for x in b.decode().splitlines()];enc=[json.loads(x) for x in gzip.decompress(checked(plan['all80_encoded'])).decode().splitlines()];audits=json.loads(checked(plan['all80_audit']));assert len(rows)==len(enc)==len(audits)==80
 cpu=json.loads(checked(plan['CPU_result']));assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact']
 assert plan['all80_encoded']==cpu['outputs']['all80-published-encoded.jsonl.gz'] and plan['all80_audit']==cpu['outputs']['all80-input-audit.json']
 seen=set();concepts=set();patients={};tokens=0
 for i,(r,e,a) in enumerate(zip(rows,enc,audits)):
  assert r['row_id']==e['row_id']==a['row_id']==plan['row_ids'][i] and a['index']==i
  assert r['row_id'] not in seen;seen.add(r['row_id']);assert set(r)=={'row_id','pmc_id','code_map','patient_ids','messages'}
  assert [m['role'] for m in r['messages']]==['system','user'] and all(set(m)=={'role','content'} for m in r['messages'])
  assert sha_bytes(json.dumps(r,ensure_ascii=False,separators=(',',':')).encode())==a['original_row_sha256']
  assert sha_bytes(json.dumps(e['input_ids'],separators=(',',':')).encode())==a['native_ids_sha256'] and len(e['input_ids'])==a['native_input_tokens']
  assert all(type(v)is int and 0<=v<151936 for v in e['input_ids']);tokens+=len(e['input_ids'])
  assert len(r['code_map'])==len(set(r['code_map'].values())) and list(r['code_map'])==['C%03d'%j for j in range(len(r['code_map']))]
  assert all(type(x)is str for x in r['code_map'].values()) and 1<=len(r['code_map'])<=8
  assert all(type(x)is str for x in r['patient_ids']) and len(r['patient_ids'])==len(set(r['patient_ids']))
  pmc=str(r['pmc_id']);assert pmc not in patients or patients[pmc]==r['patient_ids'];patients[pmc]=r['patient_ids']
  for value in r['code_map'].values():assert (pmc,value) not in concepts;concepts.add((pmc,value))
 assert len(patients)==16 and len(concepts)==587 and sum(map(len,patients.values()))==39 and tokens==249937
 assert set(plan['cached_rows'])=={'0','73'} and plan['new_indices']==[i for i in range(80) if i not in [0,73]]
 cached={}
 for key,ref in plan['cached_rows'].items():
  i=int(key);item=json.loads(checked(ref));assert item['row_id']==rows[i]['row_id'] and item['input_tokens']==audits[i]['native_input_tokens'];assert item['generated_tokens']==len(item['generated_ids'])
  assert item['native_EOS_observed'] and item['generated_ids'][-1]==151645 and not item['stopped_at_token_cap'] and not item['stopped_at_wall_budget']
  assert item['generated_tokens']=={0:746,73:2197}[i]
  parsed=parse_item(item,rows[i]);assert parsed['normalized_valid'];cached[i]=item
  evidence=json.loads(checked(plan['cached_provenance'][key]['result']));oldplan=json.loads(checked(plan['cached_provenance'][key]['plan']))
  assert any(x['sha256']==ref['sha256'] and x['row_id']==item['row_id'] for x in evidence['rows'])
  assert oldplan['seed']==20261024 and oldplan['max_new_tokens']==4096 and oldplan['temperature']==.6 and oldplan['top_p']==.95 and oldplan['top_k']==0
  assert oldplan['MS_revision']==plan['MS_revision'] and oldplan['model_dir']==plan['model_dir'] and oldplan['derived_config']==plan['derived_config']
  assert evidence['parameters']['count']==8190735360 and evidence['actual_rotary']['attention_scaling']==1.0 and evidence['runtime']['precision']=='native_BF16_SDPA_no_quantization'
 return rows,enc,audits,cached

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();t=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 plan=json.loads(Path(a.plan).read_text());rows,enc,audits,cached=load_inputs(plan);case=tests()
 result={'status':'ACTUAL_FORMAL_INPUT_CACHE_SCHEMA_PATH_CPU_PASS','source_sha256':sha_bytes(Path(__file__).read_bytes()),'plan_sha256':sha_bytes(Path(a.plan).read_bytes()),'rows':len(rows),'tokens':sum(len(e['input_ids']) for e in enc),'cached_indices':list(cached),'new_indices':plan['new_indices'],'codec_cases':case,'program_seconds':time.monotonic()-t,'no_model_tokenizer_GPU_gold_quality':True}
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(result))
if __name__=='__main__':main()
