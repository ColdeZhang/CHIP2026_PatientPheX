from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
began=time.monotonic();b=Path(__file__).parents[1]
def ref(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
refs={}
for collection_name,directory in [('training-collection.json','training-terminal'),('inference-collection.json','inference-terminal')]:
 c=read(b/collection_name);refs[collection_name]=ref(b/collection_name)
 for n,v in c['files'].items():
  actual=ref(b/directory/n);assert all(actual[k]==v[k] for k in ['bytes','sha256']);refs[directory+'/'+n]=actual
train=read(b/'training-terminal/training_complete.json');inf=read(b/'inference-terminal/wrapper-complete.json')
assert train['epochs']==2 and train['global_step']==52 and inf['returncode']==1
assert inf['gpu_release']['target_has_no_process']
assert not (b/'inference-terminal/complete.json').exists() and not (b/'inference-terminal/responses.manifest.json').exists()
assert not any((b/n).exists() for n in ['assembled','secondary-union','evaluation','owner-terminal-manifest.json'])
plan=read(b/'endpoint-plan.json');r=plan['references'];prompts=rows(r['prompts']['path']);audit={x['row_id']:x['patient_first'] for x in read(r['input_token_audit']['path'])}
for key in ['prompts','input_token_audit','parser','schema']:
 refs[key]=ref(r[key]['path']);assert all(refs[key][k]==r[key][k] for k in ['bytes','sha256'])
ordered=sorted(prompts,key=lambda x:(audit[x['row_id']]['input_tokens'],x['row_id']))
saved=rows(b/'inference-terminal/responses.jsonl');assert len(saved)==23
assert [x['row_id'] for x in saved]==[x['row_id'] for x in ordered[:23]]
failed=ordered[23];source=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_first_f0/formal/infer.py');text=source.read_text();refs['infer_source']=ref(source)
assert text.index('            parse_final(record, row)') < text.index('            handle.write(json.dumps(record, ensure_ascii=False)')
trace=(b/'inference-terminal/stderr.log').read_text();assert trace.rstrip().endswith('ValueError: missing or unknown matrix key')
assert 'validate_matrix(new, patients, codes)' in trace
report=dict(status='original_strict_final58_stopped_on_first_format_failure_no_quality',at_utc=datetime.now(timezone.utc).isoformat(),training_success=True,epochs=2,optimizer_calls=52,
 inference_wrapper=inf,saved_raw_rows=23,expected_rows=58,failed_execution_index_1based=24,
 inferred_failed_input=dict(row_id=failed['row_id'],input_tokens=audit[failed['row_id']]['input_tokens'],allowed_patient_keys=failed['patient_ids'],allowed_candidate_values=list(failed['code_map'])),
 inference_of_identity='Actual saved row order equals first23 frozen length,row_id order; next input inferred from fixed runner, not from a persisted failed record.',
 failure=dict(stage='validate_matrix(new, patients, codes): non-dict JSON or top-level patient-key set differs',exception='ValueError: missing or unknown matrix key',raw_and_generated_ids_persisted=False,reason='Runner calls parse_final before handle.write/flush; failure exits without raw exception writer.',exact_missing_unknown_keys=None,actual_failed_text=None,actual_failed_EOS_ID=None,actual_failed_output_tokens=None,CID_values_checked=False),
 limited_control_flow_inference=dict(EOS_and_token_envelope_passed=True,hit_token_limit_flag_false=True,JSON_parse_passed=True,basis='Trace reached later matrix-key validation after exact frozen parse_final and parse_json; no failed-row raw to independently inspect.',not_a_persisted_token_receipt=True),
 no_new_quality=True,no_assembly_or_prediction=True,no_retry_or_fallback=True,original_state_and_source_unchanged=True,
 references=refs,diagnostic_source=ref(__file__),seconds=time.monotonic()-began)
out=b/'strict-failure/diagnostic.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(out)))
