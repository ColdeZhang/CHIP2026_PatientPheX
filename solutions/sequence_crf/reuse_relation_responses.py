#!/usr/bin/env python3
"""Reuse only completely identical prompt rows, with model/decoder/source identity."""
import argparse
import hashlib
import json
from pathlib import Path
from run import read_jsonl, sha


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def index(rows):
    result = {row['row_id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def check_runtime(old, new, original_cwd):
    a, b = old['fingerprint'], new['fingerprint']
    for key in ['code_sha256', 'model_config_sha256', 'adapter_files', 'base_weight_files', 'inference_precision']:
        assert a[key] == b[key], 'Inference fingerprint mismatch: '+key
    for key in ['max_input_tokens', 'max_new_tokens', 'batch_size', 'model_revision', 'model']:
        assert a['configuration'][key] == b['configuration'][key], 'Decoder/model mismatch: '+key
    old_adapter = Path(a['configuration']['adapter'])
    if not old_adapter.is_absolute():
        old_adapter = Path(original_cwd)/old_adapter
    assert str(old_adapter) == b['configuration']['adapter']
    for key in ['torch', 'transformers', 'gpu']:
        assert old[key] == new[key], 'Runtime mismatch: '+key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['original-prompts', 'prompts', 'original-responses', 'original-manifest', 'runner', 'output-dir']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--new-responses')
    p.add_argument('--new-manifest')
    p.add_argument('--original-cwd')
    args = p.parse_args()
    assert bool(args.new_responses) == bool(args.new_manifest)
    rows = read_jsonl(args.prompts)
    new, old = index(rows), index(read_jsonl(args.original_prompts))
    responses = index(read_jsonl(args.original_responses))
    m = json.loads(Path(args.original_manifest).read_text())
    assert set(new) == set(old) == set(responses)
    assert m['fingerprint']['prompts_sha256'] == sha(args.original_prompts)
    assert m['output_sha256'] == sha(args.original_responses)
    assert m['fingerprint']['code_sha256'] == sha(args.runner)
    changed, cached = [], []
    for row in rows:
        key = row['row_id']
        before = old[key]
        assert {k: v for k, v in row.items() if k != 'messages'} == {k: v for k, v in before.items() if k != 'messages'}
        assert row['messages'][0] == before['messages'][0]
        assert str(responses[key]['pmc_id']) == str(row['pmc_id'])
        if row == before:
            cached.append(responses[key])
        else:
            changed.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    changed_path, cached_path = out/'changed.prompts.jsonl', out/'cached.responses.jsonl'
    changed_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in changed))
    cached_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in cached))
    manifest = {'configuration': vars(args), 'source_sha256': sha(__file__),
                'files': {k: sha(getattr(args, k)) for k in ['original_prompts', 'prompts', 'original_responses', 'original_manifest', 'runner']},
                'full_rows': len(rows), 'changed_rows': len(changed), 'cached_rows': len(cached),
                'changed_prompt_sha256': sha(changed_path), 'cached_responses_sha256': sha(cached_path),
                'original_runtime_fingerprint': m['fingerprint'],
                'scope': 'cache complete identical original row including ID/map/patient IDs/messages; fresh changed responses must also match actual model weights/source/decoder/runtime'}
    if args.new_responses:
        assert args.original_cwd
        fresh = index(read_jsonl(args.new_responses))
        nm = json.loads(Path(args.new_manifest).read_text())
        assert set(fresh) == {row['row_id'] for row in changed}
        assert nm['fingerprint']['prompts_sha256'] == sha(changed_path)
        assert nm['output_sha256'] == sha(args.new_responses)
        check_runtime(m, nm, args.original_cwd)
        all_responses = index(cached)
        assert not set(all_responses) & set(fresh)
        all_responses.update(fresh)
        provenance = []
        for row in rows:
            key = row['row_id']
            response = all_responses[key]
            assert str(response['pmc_id']) == str(row['pmc_id'])
            source = args.new_responses if key in fresh else args.original_responses
            provenance.append({'row_id': key, 'pmc_id': row['pmc_id'],
                               'source': 'fresh_changed' if key in fresh else 'cached_identical',
                               'prompt_object_sha256': object_sha(row),
                               'response_object_sha256': object_sha(response),
                               'source_response_file': source, 'source_response_sha256': sha(source)})
        path = out/'combined.responses.jsonl'
        path.write_text(''.join(json.dumps(all_responses[row['row_id']], ensure_ascii=False)+'\n' for row in rows))
        (out/'row-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        manifest['combined_response_sha256'] = sha(path)
        manifest['new_manifest_sha256'] = sha(args.new_manifest)
        manifest['all_runtime_fingerprints_equal'] = True
    (out/'reuse.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['configuration', 'files', 'original_runtime_fingerprint']}))


if __name__ == '__main__':
    main()
