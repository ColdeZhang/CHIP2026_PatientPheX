"""Truthful V1 failure and complete raw coverage; no model, parse or gold access."""
import hashlib,json
from pathlib import Path

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(collection_path,decision_path):
 decision=read(decision_path);assert decision['schema']=='copy-grounded-posthoc-whole-row-abstention-v2'
 assert ref(collection_path)==decision['original']['collection']
 col=read(collection_path);assert not col['inference_reported_success']
 assert col['all_actual_SHA_verified'] and col['all39_raw_and_ID_bytes_verified']
 assert col['files_and_ancestor_directories_fsynced'] and col['remote_archive_frozen']
 assert col['collection_mode']=='closed_child_byteexact_archive' and col['child_exit_before_freeze']['no_running_process']
 assert ref(col['archive']['path'])==col['archive'];files=col['files'];local=Path(col['local']).resolve();remote=Path(col['original_remote'])
 for name,item in files.items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert Path(item['path']).resolve()==local/name and ref(item['path'])==item
 wrapper=read(files['wrapper-complete.json']['path']);complete=read(files['output/complete.json']['path'])
 assert wrapper==col['wrapper'] and wrapper['returncode']==1 and not wrapper['timed_out']
 assert wrapper['target_has_no_process'] and wrapper['no_automatic_retry']
 for key,name in [('wrapper','wrapper-complete.json'),('complete','output/complete.json'),('responses','output/responses.jsonl'),('audit','output/constraint-audit.json')]:assert files[name]==decision['original'][key]
 assert complete['status']=='COMPLETE_FRESH39_CONSTRAINED_RAW_NO_QUALITY' and complete['rows']==39 and complete['documents']==16
 assert complete['no_regeneration_or_answer_repair']
 def mapped(item):
  key=Path(item['path']).relative_to(remote).as_posix();found=files[key]
  assert found['bytes']==item['bytes'] and found['sha256']==item['sha256'];return found
 assert mapped(complete['responses'])==files['output/responses.jsonl']
 records=[json.loads(x) for x in Path(files['output/responses.jsonl']['path']).read_text().splitlines()]
 rows={r['row_id']:r for r in records};audit=read(mapped(complete['constraint_audit'])['path']);checks={x['row_id']:x for x in audit}
 assert len(rows)==len(records)==len(checks)==len(audit)==39 and set(rows)==set(checks)
 for key in ['generation_refs','raw_refs']:
  members={read(mapped(i)['path'])['row_id']:(i,read(mapped(i)['path'])) for i in complete[key]}
  assert len(complete[key])==len(members)==39 and set(members)==set(rows)
  for rid,(item,value) in members.items():
   r=rows[rid]
   if key=='raw_refs':assert value==r and checks[rid]['raw']==item
   else:
    assert r['generation']==item and not value['decoded_or_parsed']
    assert all(r[k]==v for k,v in value.items() if k!='decoded_or_parsed')
 for r in records:
  assert r['raw_decoded'] and not r['parsed'] and r['raw_before_parse'] and r['HF_decode_equal']
  assert r['output_tokens']==len(r['generated_token_ids'])
  if checks[r['row_id']]['pass_strict']:
   assert r['actual_last_token_eos'] and not r['hit_token_limit'] and checks[r['row_id']]['complete_EOS_schema_pool_unique']
  else:assert checks[r['row_id']]['whole_row_rejected']
 stats=dict(rows=39,EOS=sum(r['actual_last_token_eos'] for r in records),cap=sum(r['hit_token_limit'] for r in records),strict=sum(x['pass_strict'] for x in audit),complete_EOS_parser_fail=sum(r['actual_last_token_eos'] and not r['hit_token_limit'] and not checks[r['row_id']]['pass_strict'] for r in records))
 assert stats==decision['observed_counts']
 assert complete['EOS_rows']==stats['EOS'] and complete['cap_rows']==stats['cap'] and complete['strict_rows']==stats['strict']
 return checks,stats

def interpret_row(record,doc,check_result,interpret):
 # Cap/non-EOS: original parser returns before json.loads; do not inspect prefixes.
 if record['hit_token_limit'] or not record['actual_last_token_eos']:
  d=interpret(record['response'],doc,record['actual_last_token_eos'],record['hit_token_limit'])
  assert d['status']=='abstain' and not d['anchors'] and d['reason']=='non_eos_or_token_cap';return d
 # Generic original strict failure: full-row abstention even if JSON might parse.
 if not check_result['pass_strict']:
  return dict(status='abstain',anchors=[],rejected=[],raw_not_repaired=True,reason='original_strict_failure_whole_row')
 d=interpret(record['response'],doc,True,False)
 assert d['status']=='complete' and not d['rejected']
 return d
