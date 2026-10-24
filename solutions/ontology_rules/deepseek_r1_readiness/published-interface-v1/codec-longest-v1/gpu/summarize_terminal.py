"""Summarize immutable published-interface pilot receipts without tokenizer/model/scoring."""
from pathlib import Path
import hashlib,json,math,datetime
G=Path(__file__).parent;C=G.parent;P=C.parent

def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):return json.loads(Path(p).read_text())
def main():
 plan=load(G/'plan.json');pr=ref(G/'plan.json');assert pr['sha256']=='d6f6bc5e3b84fc41c73d060f3e9dca8df01959e9ffabad51ba870cbede3c3604'
 sources={n:ref(G/n) for n in plan['sources']};assert all(sources[n]['sha256']==h for n,h in plan['sources'].items())
 R=G/'recovered';wrapper=load(R/'wrapper-result.json');assert wrapper['plan_sha256']==pr['sha256'] and wrapper['source_sha256']==plan['sources']['launch_probe.py'] and wrapper['cap_seconds']==240 and wrapper['TERM_grace_seconds']==30 and wrapper['only_published_longest_new_codec']
 resultpath=R/'terminal/result.json';result=load(resultpath) if resultpath.exists() else None
 if result:
  rr=ref(resultpath);assert rr['sha256']==wrapper['result_sha256'] and result['plan']['sha256']==pr['sha256'] and result['source']['sha256']==plan['sources']['run_probe.py'];assert result['quality'] is False and result['original80_not_inferred']
 rows=[];bound={};inputs=[json.loads(x) for x in (P/'cpu/recovered/selected-prompts.jsonl').read_text().splitlines()][1:2]
 for i,r in enumerate(result.get('rows',[]) if result else []):
  p=R/'terminal'/('row%d.json'%i);a=ref(p);assert a['sha256']==r['sha256'] and a['bytes']==r['bytes'];v=load(p);assert v['row_id']==plan['row_ids'][i]==inputs[i]['row_id'] and v['input_tokens']==plan['input_tokens'][i];ids=v['generated_ids'];assert 0<=len(ids)==v['generated_tokens']<=4096 and all(type(x) is int and x>=0 for x in ids);bound[i]=ids
  assert v['native_EOS_observed']==bool(ids and ids[-1]==result['generation_arguments']['eos_token_id']);assert v['stopped_at_token_cap']==(not v['native_EOS_observed'] and len(ids)>=4096);assert math.isclose(v['tokens_per_second_including_prefill'],len(ids)/v['generation_seconds'],rel_tol=1e-9)
  fields=['row_id','input_tokens','generated_tokens','native_EOS_observed','stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','tokens_per_second_including_prefill','strict_final_valid','legal_sparse_valid','normalized_valid'];assert all(v[k]==r[k] for k in fields)
  summary={k:v[k] for k in fields};summary.update(file=a,generated_IDs_SHA=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),think_open_count=v['think_open_count'],think_close_count=v['think_close_count'],first_think_close_character_offset=v['first_think_close_character_offset'],peak_allocated_bytes=v['peak_allocated_bytes'],peak_reserved_bytes=v['peak_reserved_bytes'])
  assert v['think_open_count']==v['raw_generated_text'].count('<think>') and v['think_close_count']==v['raw_generated_text'].count('</think>')
  if v['legal_sparse_valid']:
   provided=v['provided_mapping'];missing=sorted(set(inputs[i]['code_map'])-set(provided));assert set(provided).issubset(inputs[i]['code_map']) and missing==v['abstained_CIDs'];assert v['strict_every_CID']==v['strict_final_valid']==(not missing);assert v['missing_CIDs_are_negative'] is False and v['native_EOS_observed'] and not v['stopped_at_token_cap'] and not v['stopped_at_wall_budget']
   summary.update(provided_CIDs=sorted(provided),provided_CID_count=len(provided),explicit_empty_list_CIDs=sorted(k for k,x in provided.items() if not x),abstained_CIDs=missing,missing_CIDs_are_negative=False)
  else:summary['sparse_invalid_reason']=v.get('sparse_invalid_reason')
  if not v['strict_final_valid']:summary['strict_invalid_reason']=v.get('strict_invalid_reason')
  if v['normalized_valid']:
   evidence=v['normalized_evidence'];provided=evidence['provided_mapping'];missing=sorted(set(inputs[i]['code_map'])-set(provided));assert set(provided).issubset(inputs[i]['code_map']) and evidence['abstained_CIDs']==missing and evidence['normalized_every_CID']==(not missing) and evidence['missing_CIDs_are_negative'] is False and evidence['semantic_answer_repair_performed'] is False
   assert all(type(x) is str and x in inputs[i]['patient_ids'] for values in provided.values() for x in values) and all(len(values)==len(set(values)) for values in provided.values())
   for alias in evidence['integer_PID_aliases_applied']:assert type(alias['original_integer']) is int and alias['original_integer']>=0 and alias['exact_allowed_PID']=='P'+str(alias['original_integer']) and alias['exact_allowed_PID'] in inputs[i]['patient_ids'] and provided[alias['CID']][alias['position']]==alias['exact_allowed_PID']
   summary.update(normalized_provided_CIDs=sorted(provided),normalized_every_CID=evidence['normalized_every_CID'],normalized_abstained_CIDs=missing,whole_outer_fence_removed=evidence['whole_outer_fence_removed'],integer_PID_aliases_applied=evidence['integer_PID_aliases_applied'],normalized_explicit_empty_CIDs=sorted(k for k,x in provided.items() if not x),missing_CIDs_are_negative=False)
  else:summary['normalized_invalid_reason']=v.get('normalized_invalid_reason')
  rows.append(summary)
 assert len(rows)<=len(plan['row_ids'])
 if result and result['status'].startswith('PASS_'):assert len(rows)==len(plan['row_ids']) and all(x['normalized_valid'] for x in rows)
 if result and result['status'].startswith('STOPPED_'):assert rows and not rows[-1]['normalized_valid'] and all(x['normalized_valid'] for x in rows[:-1])
 partial=[]
 for p in sorted((R/'terminal').glob('row*.partial.json')):
  v=load(p);i=plan['row_ids'].index(v['row_id']);ids=v['generated_ids'];assert len(ids)==v['generated_tokens']<=4096
  if i in bound:assert ids==bound[i][:len(ids)]
  partial.append({'file':ref(p),'row_id':v['row_id'],'token_count':len(ids),'intermediate_only_not_final':True})
 release=wrapper.get('actual_release',{});target_clear=not any(x.split(',')[0].strip()==plan['gpu_uuid'] for x in release.get('compute_apps','').splitlines());assert wrapper.get('target_UUID_has_no_compute_process')==target_clear
 summary={'status':'FROZEN_NEW_CODEC_LONGEST_RECEIPT_SUMMARY_NO_QUALITY','source':ref(__file__),'plan':pr,'sources':sources,'wrapper':ref(R/'wrapper-result.json'),'result':ref(resultpath) if result else None,'terminal_status':result.get('status') if result else 'NO_RUNNER_RESULT_WRAPPER_TERMINAL_ONLY','wrapper_returncode':wrapper.get('returncode'),'wrapper_seconds':wrapper.get('GPU_main_seconds'),'program_seconds':result.get('program_seconds') if result else None,'load_seconds':result.get('load_seconds') if result else None,'actual_parameters':result.get('parameters') if result else None,'actual_rotary':result.get('actual_rotary') if result else None,'rows':rows,'partial_snapshots':partial,'release':release,'target_UUID_has_no_compute_process':target_clear,'quality_unknown':True,'no_full80_inference':True,'missing_CIDs_abstention_never_filled':True,'scope':'Newfixedwholefence/canonicalintegerPIDcodec onlongestonly;firstreusedCPUonly. Freshseed/prompt4096/240+30,oldrawpolicyverdictsretained;noqualityorbudgetexpansion.'}
 with (G/'terminal-summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
 print(json.dumps({'summary':ref(G/'terminal-summary.json'),'terminal_status':summary['terminal_status'],'rows':rows,'seconds':summary['wrapper_seconds']}))
if __name__=='__main__':main()
