from pathlib import Path
import ast, datetime, hashlib, importlib.util, json

R = Path('/workspace/patientphex-runs/round2-20261023')
B = R/'case_retrieval/qwen35-preparation/thinking-sampling-followup'
O = R/'validation/thinking-sampling-terminal-review'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
ref = lambda p: dict(path=str(Path(p)), bytes=Path(p).stat().st_size, sha256=sha(p))
plan = read(B/'run-plan.json')
assert sha(B/'run-plan.json') == '84d785833f232fcfd05827065684632ee385f85528aa99b452e64fe771ddf952'
for item in plan['references'].values():
    assert ref(item['path']) == item
for name, field in [('runner.py','runner_sha256'), ('presence.py','presence_sha256'), ('finalize.py','finalizer_sha256'), ('launch.py','driver_sha256')]:
    assert sha(B/'source-snapshots'/name) == plan[field]
prompt = json.loads((B/'prompts.jsonl').read_text())
old_first = json.loads(Path(plan['references']['original80']['path']).open().readline())
assert prompt == old_first
assert prompt['row_id'] == '10313222:relations:0'
assert set(prompt) == {'row_id','pmc_id','code_map','patient_ids','messages'}
assert [m['role'] for m in prompt['messages']] == ['system','user']
assert len(prompt['code_map']) == 8 and len(prompt['patient_ids']) == 1
cpu = read(B/'cpu-result.json')
assert not cpu['cuda_initialized'] and cpu['no_model_weights_loaded'] and cpu['no_targets']
assert len(cpu['cases']) == 9
assert cpu['source_sha256'] == sha(B/'source-snapshots/cpu_contract.py')
assert cpu['presence_sha256'] == plan['presence_sha256']
assert cpu['tokenizer']['True']['tokens'] == 2030

raw = [json.loads(s) for s in (B/'responses.jsonl').read_text().splitlines()]
assert len(raw) == 1
row = raw[0]
meta = read(B/'responses.manifest.json'); fp = meta['fingerprint']
frozen = read(B/'responses-freeze.json'); summary = read(B/'summary.json')
for f, key in [('run-plan.json','plan_sha256'),('responses.jsonl','raw_sha256'),('responses.manifest.json','manifest_sha256'),('gpu-release.json','gpu_release_sha256')]:
    assert sha(B/f) == frozen[key]
for item in summary['references'].values():
    assert ref(item['path']) == item
assert meta['expected_rows'] == 1 and meta['output_sha256'] == sha(B/'responses.jsonl')
assert not any(meta['load_information'].values())
assert fp['sampler'] == plan['sampler'] == dict(do_sample=True,temperature=1.0,top_p=.95,top_k=20,min_p=0.0,repetition_penalty=1.0,generated_suffix_presence_penalty=1.5,seed=20261023)
assert fp['source_sha256'] == plan['runner_sha256'] and fp['presence_sha256'] == plan['presence_sha256']
assert fp['adapter_files'] == plan['adapter_files']
assert fp['adapter_files']['adapter_model.safetensors'] == '4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'
model_source = read(plan['references']['model_source']['path'])
assert model_source['hf_revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
assert fp['model_files'] == {v['name']:dict(bytes=v['size'],sha256=v['sha256']) for v in model_source['files']}
assert fp['source_manifest_sha256'] == plan['references']['model_source']['sha256']
assert fp['thinking'] and fp['runtime']['transformers'] == '5.20.2' and fp['runtime']['peft'] == '0.21.2'
assert fp['prompts_sha256'] == sha(B/'prompts.jsonl')
assert row['row_id'] == prompt['row_id'] and row['input_tokens'] == 2030
assert row['input_ids_sha256'] == cpu['tokenizer']['True']['input_ids_sha256']
assert row['output_tokens'] == len(row['generated_token_ids']) == row['presence_calls'] == 4096
assert row['maximum_penalized_suffix_length'] == 4095
assert row['last_token_id'] == row['generated_token_ids'][-1]
assert row['last_token_id'] not in meta['eos_token_ids']
assert row['hit_token_limit'] and '</think>' not in row['response']
assert summary['strict_final_valid'] is False and summary['invalid_reason'] == 'token_limit'
assert summary['think_closed'] is False and summary['EOS_observed'] is False
assert summary['no_scores'] and summary['no_targets_opened'] and summary['no_retry']
assert summary['total_native_seconds'] == meta['elapsed_seconds'] == 209.493
assert summary['generated_seconds'] == row['batch_seconds'] == 186.896

# Independent format contract counterexamples; does not read a target or evaluate quality.
spec = importlib.util.spec_from_file_location('strict_contract', plan['references']['strict_final']['path'])
parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
codes = {'C0':'HP:0000001'}; patients = {'P0'}
cases = [
    ('{"C0":["P0"]}',False,'missing_think_close'),
    ('{"C0":["P0"]}</think>',False,'invalid_complete_final_JSON'),
    ('x</think>{"C0":["P0"]} extra',False,'invalid_complete_final_JSON'),
    ('x</think>{"C0":[],"C0":[]}',False,'invalid_complete_final_JSON'),
    ('x</think>{"C1":[]}',False,'candidate_keys'),
    ('x</think>{"C0":["P1"]}',False,'patient_values'),
    ('x</think>{"C0":["P0","P0"]}',False,'patient_values'),
    ('x</think>{"C0":["P0"]}',True,'token_limit'),
]
for text, hit, reason in cases:
    assert parser.strict_final(text,codes,patients,hit) == (None,reason)
assert parser.strict_final('x</think>{"C0":["P0"]}',codes,patients,False) == ({'C0':['P0']},None)
assert parser.strict_final(row['response'],prompt['code_map'],prompt['patient_ids'],row['hit_token_limit']) == (None,'token_limit')

job = read(B/'job.json')['job']; monitor = read(B/'monitor.json')
assert job['id'] == monitor['job'] == 'proc_260680cf-7024-405d-a752-3f00e09f8d80'
assert not monitor['running']
release = json.loads(read(B/'gpu-release.json')['stdout'])
assert release['empty'] and release['gpu_uuid'] == plan['gpu_uuid']
assert not any(l.split(',')[0].strip()==plan['gpu_uuid'] for l in release['apps'].splitlines())
started = datetime.datetime.fromisoformat(job['started_at'].replace('Z','+00:00'))
released = datetime.datetime.fromtimestamp(release['server_epoch'],datetime.timezone.utc)
assert (released-started).total_seconds() < 300
assert datetime.datetime.fromisoformat(frozen['at_utc']) < datetime.datetime.fromisoformat(summary['at_utc'])
recovery = read(B/'recovery-plan.json')
assert recovery['existing_job'] == job['id'] and recovery['original_plan_sha256'] == sha(B/'run-plan.json')
assert recovery['recovery_sha256'] == sha(B/'recover_existing.py')
tree = ast.parse((B/'recover_existing.py').read_text())
assert not any(isinstance(n,ast.Constant) and n.value=='start_process' for n in ast.walk(tree))

report = dict(
    status='PASS_SINGLE_SAMPLED_THINKING_FORMAT_DIAGNOSTIC_INVALID_FINAL_NO_QUALITY',
    at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Incremental source/prompt/sampler/strict-final/chronology review; no reviewer GPU, generation, target parsing or quality evaluation.',
    result=dict(input_tokens=2030,output_tokens=4096,presence_calls=4096,maximum_generated_suffix_length=4095,think_closed=False,EOS=False,token_limit=True,strict_final=False,reason='token_limit'),
    actual=dict(generation_seconds=186.896,native_total_seconds=209.493,peak_allocated_bytes=18724500992,cap_seconds=300,TERM_grace_seconds=30,GPU_release_utc=released.isoformat(),managed_running_false=True,managed_exit_code_not_returned=True),
    checks=dict(original_first_input_whole_object_exact=True,input_contains_only_system_user=True,actual_true_tokenizer_fingerprint_exact=True,old32_adapter_and_public_model_fingerprint_exact=True,source_bound_generated_suffix_distinct_token_additive_presence=True,actual_CPU_nine_cases_verified_receipts=True,actual_generate_4096_processor_calls=True,strict_final_nine_reviewer_counterexamples_pass=True,one_original_job_no_NN_retry=True,recovery_only_monitor_and_collection=True,complete_raw_frozen_before_summary=True,no_targets_or_scores=True),
    limitations=[
        'Single predetermined input and multiple differences from previous full64/greedy1600 attempt; no causal or quality comparison.',
        'Reaching 4096 without a closing think/final invalidates this bounded inference recipe, not every thinking model or larger budget.',
        'Presence runtime math is checked from frozen source and actual CPU receipts; no second generation or numeric GPU rerun.',
        'MCP returns running=false but no inference exit code. The rc0 in GPU-release.json belongs only to the resource check.',
    ],
    references=[ref(B/n) for n in ['run-plan.json','cpu-result.json','cpu-execution.json','source-snapshots/runner.py','source-snapshots/presence.py','source-snapshots/cpu_contract.py','source-snapshots/finalize.py','runner-source.diff','responses-freeze.json','responses.manifest.json','summary.json','gpu-release.json','monitor.json','job.json','recovery-plan.json','recover_existing.py']],
    raw_response_locator=ref(B/'responses.jsonl'),
    reviewer_source=ref(__file__),
)
(O/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],review=ref(O/'review.json'),raw_sha256=sha(B/'responses.jsonl'))))
