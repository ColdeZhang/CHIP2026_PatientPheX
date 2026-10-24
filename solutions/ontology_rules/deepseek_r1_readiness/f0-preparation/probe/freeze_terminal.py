"""Freeze only existing original F0 probe terminal evidence; no model/inference/quality."""
from pathlib import Path
import json,hashlib,os,datetime
P=Path(__file__).parent;R=P/'recovered';F=P.parent

def read(p):return json.loads(Path(p).read_text())
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
collection=read(R/'collection-manifest.json');assert collection['terminal'] is True
for n,e in collection['files'].items():assert {k:ref(R/n)[k] for k in ['bytes','sha256']}==e
plan=read(P/'plan.json');wrapper=read(R/'wrapper-result.json');final_result_exists=(R/'terminal/result.json').exists();progress_exists=(R/'terminal/progress.json').exists();result=read(R/'terminal/result.json') if final_result_exists else read(R/'terminal/progress.json') if progress_exists else {'status':'NO_FINAL_RESULT_OR_PROGRESS_AFTER_PROCESS_EXIT','rows':[]}
assert wrapper['plan_sha256']==ref(P/'plan.json')['sha256']
if 'plan' in result:assert result['plan']['sha256']==ref(P/'plan.json')['sha256'] and result['source']['sha256']==plan['sources']['runner.py']
assert wrapper['source_sha256']==plan['sources']['launcher.py'] and wrapper['cap_seconds']==240 and wrapper['TERM_grace_seconds']==30
rows=[]
for itemref in result.get('rows',[]):
 item=read(R/'terminal'/Path(itemref['path']).name);ids=read(R/'terminal/row0.ids.json');raw=read(R/'terminal/row0.raw.json');assert item['original_index']==30 and item['input_tokens']==6028 and item['row_id']=='3563609:relations:0';assert item['generated_ids_sha256']==hashlib.sha256(json.dumps(item['generated_ids'],separators=(',',':')).encode()).hexdigest()
 assert all(item[k]==raw[k]==v for k,v in ids.items());assert all(item[k]==v for k,v in raw.items());assert item['raw_reference']['sha256']==ref(R/'terminal/row0.raw.json')['sha256'];assert itemref['sha256']==ref(R/'terminal'/Path(itemref['path']).name)['sha256']
 rows.append({k:item.get(k) for k in ['row_id','original_index','input_tokens','input_ids_sha256','generated_tokens','generated_ids_sha256','native_EOS_observed','stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','tokens_per_second_including_prefill','think_open_count','think_close_count','strict_final_valid','legal_sparse_valid','normalized_valid','normalized_invalid_reason','peak_allocated_bytes','peak_reserved_bytes'] }|{'raw':ref(R/'terminal/row0.raw.json'),'ids':ref(R/'terminal/row0.ids.json'),'parsed':ref(R/'terminal'/Path(itemref['path']).name),'normalized_coverage':item.get('normalized_evidence')})
release=wrapper.get('actual_release');released=wrapper.get('target_UUID_has_no_compute_process') is True
if release:
 parts=[x.strip() for x in release['gpu'].split(',')];assert parts[1]==plan['gpu_uuid'];released=released and not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in release['compute_apps'].splitlines())
passed=final_result_exists and result['status']=='PASS_PUBLISHED_NORMALIZED_F0_LONGEST_FORMAT_COST_ONLY_NOT_QUALITY' and wrapper.get('returncode')==0 and released and len(rows)==1 and rows[0]['normalized_valid'] is True
summary={'status':'PASS_FIXED_F0_LONGEST_FORMAT_COST_ONLY_NO_QUALITY' if passed else 'ORIGINAL_F0_LONGEST_TERMINAL_NOT_FORMAT_PASS_NO_QUALITY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final_result_exists':final_result_exists,'progress_only':not final_result_exists and progress_exists,'result_status':result['status'],'error':result.get('error'),'main_returncode':wrapper.get('returncode'),'wrapper_seconds':wrapper.get('GPU_main_seconds'),'program_seconds':result.get('program_seconds'),'load_seconds':result.get('load_seconds'),'runtime':result.get('runtime'),'parameters':result.get('parameters'),'actual_rotary':result.get('actual_rotary'),'TF32_end':result.get('TF32_end'),'peak_memory':result.get('peak_memory'),'rows':rows,'actual_release':release,'released':released,'quality_evaluated':False,'full58_inference_authorized':False,'asset_copy':ref(F/'asset-copy/formal/asset-copy-ack.json'),'source_assets':plan['assets_result'],'source_template':ref(F/'probe-source/plan.template.json'),'actual_plan':ref(P/'plan.json'),'original_F0_58_CPU':ref(F/'cpu/actual-result.json'),'raw_first_policy_verified_on_actual_files':bool(rows),'scope':'One actual longest original F0 row on A800; fresh same seed from original prompt, no KV/RNG continuation. Output may be reused only by a later explicitly identical formal configuration, not assumed cross-hardware F1 bitwise equivalent.'}
save(P/'summary.json',summary)
paths=[P/'entry-manifest.json',P/'plan.json',P/'summary.json',P/'start_once.py',P/'poll_original.py',P/'freeze_terminal.py',P/'start-intent.json',P/'job.json',P/'actual-prestart-check.json',R/'collection-manifest.json']+[R/n for n in collection['files'] if not n.endswith('.log')]+[F/'probe-source'/n for n in plan['sources']]
save(P/'terminal-manifest.json',{'status':summary['status'],'quality_evaluated':False,'files':[ref(p) for p in paths],'original_raw_full_text_and_IDs_not_for_Git':True})
print(json.dumps({'summary':ref(P/'summary.json'),'manifest':ref(P/'terminal-manifest.json'),'status':summary['status'],'wrapper_seconds':summary['wrapper_seconds'],'tokens':rows[0]['generated_tokens'] if rows else None,'released':released}))
