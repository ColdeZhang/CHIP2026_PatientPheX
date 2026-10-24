"""Freeze one format/cost diagnostic; never open labels or compute quality."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True, type=Path)
    a = p.parse_args()
    b = a.directory
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
    plan = json.loads((b/'run-plan.json').read_text())
    assert sha(__file__) == plan['finalizer_sha256']
    for item in plan['references'].values():
        assert ref(item['path']) == item
    frozen = json.loads((b/'responses-freeze.json').read_text())
    assert frozen['plan_sha256'] == sha(b/'run-plan.json')
    assert frozen['raw_sha256'] == sha(b/'responses.jsonl')
    assert frozen['manifest_sha256'] == sha(b/'responses.manifest.json')
    rows = [json.loads(l) for l in (b/'responses.jsonl').read_text().splitlines()]
    prompts = [json.loads(l) for l in (b/'prompts.jsonl').read_text().splitlines()]
    assert len(rows) == len(prompts) == 1
    row, prompt = rows[0], prompts[0]
    assert row['row_id'] == prompt['row_id'] == '10313222:relations:0'
    assert 'answer' not in prompt and all(m['role'] in ['system','user'] for m in prompt['messages'])
    meta = json.loads((b/'responses.manifest.json').read_text())
    assert meta['fingerprint']['adapter_files'] == plan['adapter_files']
    assert meta['fingerprint']['sampler'] == plan['sampler']
    assert row['presence_calls'] == row['output_tokens'] == len(row['generated_token_ids'])
    assert row['last_token_id'] == row['generated_token_ids'][-1]
    assert row['hit_token_limit'] == (row['output_tokens'] >= 4096 and row['last_token_id'] not in meta['eos_token_ids'])
    spec = importlib.util.spec_from_file_location('strict_final_only',plan['references']['strict_final']['path'])
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)  # Its main/target-reading function is never called.
    obj, reason = parser.strict_final(row['response'],prompt['code_map'],prompt['patient_ids'],row['hit_token_limit'])
    result = dict(status='complete_single_input_format_cost_diagnostic_no_targets_no_quality',
                  at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),row_id=row['row_id'],
                  strict_final_valid=obj is not None,invalid_reason=reason,
                  think_closed='</think>' in row['response'],hit_token_limit=row['hit_token_limit'],
                  EOS_observed=row['last_token_id'] in meta['eos_token_ids'],
                  input_tokens=row['input_tokens'],output_tokens=row['output_tokens'],
                  presence_calls=row['presence_calls'],generated_seconds=row['batch_seconds'],
                  total_native_seconds=meta['elapsed_seconds'],peak_allocated_bytes=meta['max_gpu_allocated_bytes'],
                  legal_CID_count=len(obj) if obj is not None else None,
                  references={n:ref(b/n) for n in ['run-plan.json','cpu-result.json','responses-freeze.json','responses.jsonl','responses.manifest.json','gpu-release.json']},
                  limitation='One fixed input, old32 adapter, sampled4096; multiple changes vs old full64 greedy1600 failure. Neither success nor failure establishes quality or generalization.',
                  no_targets_opened=True,no_scores=True,no_retry=True)
    with (b/'summary.json').open('x') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
