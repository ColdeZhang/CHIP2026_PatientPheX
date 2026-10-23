#!/usr/bin/env python3
"""One final-epoch inference arm on frozen label-free encoded patient pairs."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
import torch
import transformers
from transformers import AutoModelForSequenceClassification


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    root = args.plan.resolve().parent
    out = args.output.resolve()
    require(not out.exists(), 'Existing inference must be recovered, never overwritten')
    plan = json.loads(args.plan.read_text())
    require(plan['source_sha256'] == sha(__file__), 'Inference source drift')
    require(plan['precision'] == 'bf16_autocast_fp32_parameters_adam_no_scaler', 'Precision drift')
    require(plan['arm'] in ['baseline', 'bridge'] and plan['hard_budget_seconds'] == 300, 'Wrong inference arm/cap')
    require(plan['runtime_versions'] == dict(torch=torch.__version__, cuda=torch.version.cuda,
                                             transformers=transformers.__version__), 'Runtime drift')
    for name in ['training_complete', 'training_manifest']:
        require(sha(root / plan[name]['path']) == plan[name]['sha256'], 'Training terminal changed')
    trained = json.loads((root / plan['training_complete']['path']).read_text())
    manifest = json.loads((root / plan['training_manifest']['path']).read_text())
    require(trained['status'] == 'complete_four_epochs' and trained['updates'] == 1032 and
            [x['epoch'] for x in trained['epochs']] == [1, 2, 3, 4], 'Not complete unique final4epochs')
    require(manifest['arm'] == plan['arm'] and manifest['fresh_public_base'] is True and
            manifest['old_gold_CE_or_probe_state_reused'] is False, 'Wrong training arm/provenance')
    require(trained['source_sha256'] == plan['trainer_source_sha256'] and
            trained['plan_sha256'] == plan['training_plan_sha256'], 'Wrong trained recipe')
    model_path = root / plan['model_directory']
    require(set(plan['model_files']) == {'config.json', 'model.safetensors'}, 'Unexpected checkpoint files')
    for name, expected in plan['model_files'].items():
        path = model_path / name
        require(dict(bytes=path.stat().st_size, sha256=sha(path)) == expected, 'Final model SHA mismatch')
    encoded = root / plan['encoded_file']
    require(sha(encoded) == plan['encoded_sha256'], 'Encoded inference package changed')
    with gzip.open(encoded, 'rt') as handle:
        package = json.load(handle)
    require(package['schema'] == 'patient-identity-encoded-holdout-v1', 'Not label-free holdout schema')
    rows = package['rows']
    require(len(rows) == 1166 and len({tuple(r['key']) for r in rows}) == 1166, 'Changed pair coverage')
    require(all('label' not in row and 'association' not in row for row in rows), 'Labels in inference package')
    require(len({row['key'][0] for row in rows}) == 16, 'Wrong document coverage')
    for row in rows:
        old, extra = row['old_active_tokens'], row['bridge_active_tokens']
        require(0 < old <= 384 and 0 <= extra <= 128 and old + extra <= 512, 'Input budget changed')
        for arm in ['baseline', 'bridge']:
            active = old + (extra if arm == 'bridge' else 0)
            encoded_row = row['arms'][arm]
            require(set(encoded_row) == {'input_ids', 'token_type_ids', 'attention_mask'} and
                    all(len(v) == 512 for v in encoded_row.values()), 'Invalid tensors')
            require(encoded_row['attention_mask'] == [1] * active + [0] * (512 - active), 'Invalid mask')
            require(encoded_row['input_ids'][old - 1] == encoded_row['input_ids'][active - 1] == 3, 'Original/final SEP moved')
            require(all(x == 0 for x in encoded_row['input_ids'][active:]), 'Nonzero padding')
            require(all(0 <= x < 30522 for x in encoded_row['input_ids']), 'Token vocabulary changed')
        for field in ['input_ids', 'token_type_ids', 'attention_mask']:
            require(row['arms']['baseline'][field][:old] == row['arms']['bridge'][field][:old], 'Old evidence prefix changed')
        if extra == 0:
            require(row['arms']['baseline'] == row['arms']['bridge'], 'Empty bridge differs')
    uuid = plan['gpu_uuid']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'Wrong GPU UUID')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader'], text=True)
    require(not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()), 'Assigned GPU occupied')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    model, loading = AutoModelForSequenceClassification.from_pretrained(model_path, output_loading_info=True)
    require(all(not loading[key] for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), 'Incomplete final checkpoint load')
    require(sum(p.numel() for p in model.parameters()) == 109483778, 'Wrong model architecture')
    require(manifest['precision'] == plan['precision'] and trained['precision'] == plan['precision'], 'Train/infer precision mismatch')
    require(torch.cuda.is_bf16_supported(), 'BF16 unavailable')
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Parameters must remain FP32')
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    output = []
    with torch.inference_mode():
        for first in range(0, len(rows), 16):
            group = rows[first:first + 16]
            batch = {field: torch.tensor([row['arms'][plan['arm']][field] for row in group], dtype=torch.long, device='cuda')
                     for field in ['input_ids', 'token_type_ids', 'attention_mask']}
            batch['position_ids'] = torch.arange(512, device='cuda').unsqueeze(0).expand(len(group), -1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
            for row, probability in zip(group, probs):
                require(0 <= probability <= 1, 'Nonfinite probability')
                output.append(dict(key=row['key'], probability=probability))
    path = out / 'pair-scores.jsonl'
    with path.open('x') as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    complete = dict(status='complete_unique_final_epoch_inference', arm=plan['arm'], rows=len(output),
                    source_sha256=sha(__file__), plan_sha256=sha(args.plan), encoded_sha256=sha(encoded),
                    final_epoch=4, training_updates=1032, model_files=plan['model_files'],
                    output_sha256=sha(path), no_gold_access=True, no_retokenization=True,
                    explicit_positions512=True, elapsed_seconds=time.monotonic() - start,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved(),
                    runtime=plan['runtime_versions'], loading_information=loading, precision=plan['precision'])
    save(out / 'inference-complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
