#!/usr/bin/env python3
"""Bind frozen patient-first train124 to the real CPU Transformers tokenizer.

CLI: --plan PLAN --model LOCAL_SMALL_CACHE --output NEW_DIRECTORY
PLAN has data_contract_sha256 and references with exactly these six keys:
train, train_token_audit, summary, preparation_manifest, encode_helper, transform.
Each reference has path, bytes, sha256; relative paths resolve beside PLAN.
Optional model_files maps small filenames to {bytes, sha256}; these must match
the hard bindings below. No F1 file, old train file, model weights, or CUDA is read.
The current helper.encode is called once per new row; old encode_all and old
verify_sources_and_data are not called. Synthetic parser tests are not repeated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time


FIXED = {
    'train': ('14af86cffe1a8b87af022d13e42f9b910281f0da478e3cc731bc2fbc9ba68c3b', 1720127),
    'train_token_audit': ('36f6faaee683fcaff319714a3b08535208848b45185ab72b7958829417d03586', 163235),
    'summary': ('68d28bafb7b463b68857f44957d7cf33d05116ff2e49872183f7e363458cb254', 10469),
    'preparation_manifest': ('356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47', 9323),
    'encode_helper': ('9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3', 26050),
    'transform': ('25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08', 7505),
}
MODEL_SMALL = {
    'config.json': (3126, 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'),
    'tokenizer_config.json': (16710, '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'),
    'tokenizer.json': (12807982, '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
    'chat_template.jinja': (7756, 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'),
    'vocab.json': (6722759, 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'),
    'merges.txt': (3353259, 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d'),
}
REQUIRED_SMALL = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja'}
OLD_TRAIN_SHA = 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
SELECTED_ROW_ID = '11235872:relations:8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha_bytes(raw)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(raw):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def audit_list_sha(values):
    # Existing train-token-audit uses json.dumps default separators/ensure_ascii.
    # This is intentionally NOT helpers.json_sha (compact+sorted+ensure_ascii=False).
    return sha_bytes(json.dumps(values).encode())


def import_frozen(name, path):
    require(name not in sys.modules, 'Preloaded module is not permitted: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load fixed source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_new(path, value):
    with path.open('x', encoding='utf-8') as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def token_record(row, encoded):
    return {
        'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']),
        'full_tokens': len(encoded['input_ids']), 'prefix_tokens': encoded['prefix_tokens'],
        'supervised_tokens': encoded['supervised_tokens'], 'eos_token_id': encoded['eos_token_id'],
        'rendered_prefix_sha256': encoded['rendered_prefix_sha256'],
        'rendered_full_sha256': encoded['rendered_full_sha256'],
        'input_ids_sha256': audit_list_sha(encoded['input_ids']),
        'labels_sha256': audit_list_sha(encoded['labels']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require explicit CUDA_VISIBLE_DEVICES empty')
    require(not any(x in sys.modules for x in ('torch', 'transformers', 'tensorflow', 'jax', 'peft')),
            'Run in a fresh CPU-only process')
    # Disable all model backends before importing Transformers. Tokenization is native.
    os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    require(not args.output.exists(), 'Output directory must be new; refusing overwrite')
    plan_ref = ref(args.plan)
    plan = parse(args.plan.read_bytes())
    source_ref = ref(__file__)
    require(plan.get('data_contract_sha256') == source_ref['sha256'], 'Contract source SHA differs')
    references = plan.get('references')
    require(isinstance(references, dict) and set(references) == set(FIXED), 'Require exactly six fixed input references')
    paths, actual_sources = {}, {}
    for name, (expected_sha, expected_bytes) in FIXED.items():
        item = references[name]
        require(isinstance(item, dict) and {'path', 'sha256', 'bytes'} <= set(item), 'Malformed source reference: ' + name)
        require(item['sha256'] == expected_sha and item['bytes'] == expected_bytes, 'Plan cannot change frozen input: ' + name)
        path = Path(item['path'])
        if not path.is_absolute():
            path = args.plan.resolve().parent / path
        actual = ref(path)
        require(actual['sha256'] == expected_sha and actual['bytes'] == expected_bytes, 'Actual input differs: ' + name)
        paths[name], actual_sources[name] = path.resolve(), actual

    summary = parse(paths['summary'].read_bytes())
    preparation = parse(paths['preparation_manifest'].read_bytes())
    audit = parse(paths['train_token_audit'].read_bytes())
    require(summary['status'] == 'PASS_CPU_patient_first_closed_set_preparation_no_training_or_quality', 'Preparation not complete')
    require(summary['sources']['preparation_manifest']['sha256'] == FIXED['preparation_manifest'][0], 'Preparation lineage differs')
    require(summary['sources']['encode_source']['sha256'] == FIXED['encode_helper'][0], 'Encoder lineage differs')
    require(summary['transform_source']['sha256'] == FIXED['transform'][0], 'Transform lineage differs')
    require(summary['sources']['train']['sha256'] == preparation['training_sha256'] == OLD_TRAIN_SHA, 'Original training lineage differs')
    artifacts = {Path(x['path']).name: x for x in summary['artifacts']}
    for name, artifact_name in [('train', 'train124.patient-first.jsonl'), ('train_token_audit', 'train-token-audit.json')]:
        require(artifacts[artifact_name]['sha256'] == FIXED[name][0] and artifacts[artifact_name]['bytes'] == FIXED[name][1], 'Summary artifact binding differs')
    train_docs = summary['training']['document_ids']
    f1_docs = summary['F1_input_only']['document_ids']
    excluded_docs = preparation['excluded_document_ids']
    require(len(train_docs) == len(set(train_docs)) == 32 and train_docs == sorted(train_docs), 'Invalid train ID manifest')
    require(len(f1_docs) == len(set(f1_docs)) == 16 and f1_docs == sorted(f1_docs), 'Invalid F1 ID manifest')
    require(len(excluded_docs) == len(set(excluded_docs)) == 32, 'Invalid excluded ID manifest')
    require(train_docs == preparation['training_document_ids'] and not set(train_docs) & set(excluded_docs), 'Training/excluded isolation differs')
    require(set(f1_docs) <= set(excluded_docs) and not set(train_docs) & set(f1_docs), 'Training/F1 isolation differs')
    require(preparation['all_target_documents_excluded_from_own_generators'] is True
            and preparation['no_F1_or_outer0_in_any_generator_or_SFT_training'] is True
            and preparation['fresh_base_required'] is True and preparation['old_adapter_continuation_forbidden'] is True,
            'Original source isolation/fresh-base contract differs')
    require(summary['training']['rows'] == preparation['training_rows'] == 124, 'Training row manifest differs')
    require(summary['F1_input_only']['targets_read'] is False and summary['F1_input_only']['answer_fields'] == 0,
            'F1 manifest no longer input-only')

    declared_small = plan.get('model_files', {})
    require(isinstance(declared_small, dict) and set(declared_small) <= set(MODEL_SMALL), 'Only fixed small tokenizer files may be declared')
    actual_small = {}
    for name, (size, digest) in MODEL_SMALL.items():
        path = args.model / name
        if name in REQUIRED_SMALL or path.exists() or name in declared_small:
            actual = ref(path)
            require(actual['bytes'] == size and actual['sha256'] == digest, 'Tokenizer small file differs: ' + name)
            if name in declared_small:
                require(declared_small[name]['bytes'] == size and declared_small[name]['sha256'] == digest, 'Plan small file binding differs')
            actual_small[name] = actual
    require(actual_small['tokenizer.json']['sha256'] == summary['sources']['tokenizer']['sha256'], 'Tokenizer provenance differs')
    require(actual_small['chat_template.jinja']['sha256'] == summary['sources']['template']['sha256'], 'Template provenance differs')

    helper = import_frozen('qwen35_smoke', paths['encode_helper'])
    transform = import_frozen('_patient_first_fixed_transform', paths['transform'])
    require(ref(inspect.getsourcefile(helper.encode))['sha256'] == FIXED['encode_helper'][0], 'Wrong encode callable source')
    require(ref(inspect.getsourcefile(transform.validate_input))['sha256'] == FIXED['transform'][0], 'Wrong schema callable source')
    import transformers
    from transformers import AutoTokenizer
    import tokenizers
    require(transformers.__version__ == '5.20.2', 'Require actual Transformers 5.20.2')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast and tokenizer.eos_token_id == 248046, 'Unexpected native tokenizer/EOS')
    require(isinstance(tokenizer.chat_template, str)
            and sha_bytes(tokenizer.chat_template.encode()) == MODEL_SMALL['chat_template.jinja'][1], 'Active native chat template differs')
    require(not any(x in sys.modules for x in ('torch', 'tensorflow', 'jax', 'peft')), 'Unexpected model backend imported')

    raw_rows = paths['train'].read_text(encoding='utf-8').splitlines()
    require(len(raw_rows) == 124 and all(raw_rows), 'Require exact 124 nonempty training rows')
    rows = [transform.parse_json(line) for line in raw_rows]
    require(len(audit) == 124 and len({x['row_id'] for x in audit}) == 124, 'Token audit row count/uniqueness differs')
    row_ids = [row.get('row_id') for row in rows]
    require(all(isinstance(x, str) and x for x in row_ids) and len(set(row_ids)) == 124, 'Invalid training row IDs')
    require(row_ids == [x['row_id'] for x in audit], 'Frozen row order differs')
    require(sorted({str(row['pmc_id']) for row in rows}) == train_docs, 'Actual training documents differ')
    encoded_rows, records = [], []
    for row, expected in zip(rows, audit):
        transform.validate_input(row)
        require(set(row) == transform.INPUT_KEYS | {'answer'}, 'Require training schema with answer')
        system_lines = row['messages'][0]['content'].split('\n')
        require(system_lines[:2] == [transform.NEW_TASK, transform.NEW_SCHEMA] and len(system_lines) >= 3, 'Wrong patient-first system schema')
        answer = transform.parse_json(row['answer'])
        transform.validate_matrix(answer, row['patient_ids'], list(row['code_map']))
        canonical_answer = {pid: [cid for cid in row['code_map'] if cid in answer[pid]] for pid in row['patient_ids']}
        require(row['answer'] == transform.compact(canonical_answer), 'Patient/CID ordering or compact answer serialization differs')
        encoded = helper.encode(row, tokenizer)
        actual_record = token_record(row, encoded)
        require(actual_record == expected['patient_first'], 'Native token audit mismatch: ' + row['row_id'])
        require(len(encoded['labels']) == len(encoded['input_ids']) and encoded['prefix_tokens'] + encoded['supervised_tokens'] == len(encoded['input_ids']), 'Encoded shape differs')
        require(encoded['eos_token_id'] == 248046 and encoded['input_ids'][encoded['eos_supervised_position']] == 248046
                and encoded['labels'][encoded['eos_supervised_position']] == 248046, 'EOS supervision differs')
        encoded_rows.append(encoded)
        records.append(actual_record)

    lengths = [len(x['input_ids']) for x in encoded_rows]
    supervised = [x['supervised_tokens'] for x in encoded_rows]
    require(len(encoded_rows) == 124 and max(lengths) == 5556 and sum(lengths) == 400614 and sum(supervised) == 4694, 'New frozen token aggregate guard failed')
    stat = summary['training']['token_statistics']['patient_first']
    require(stat['full_tokens']['maximum'] == 5556 and stat['full_tokens']['sum'] == 400614
            and stat['supervised_tokens']['sum'] == 4694 and stat['longest_row'] == SELECTED_ROW_ID, 'Summary aggregate binding differs')
    longest = [x for x in encoded_rows if len(x['input_ids']) == 5556]
    require(len(longest) == 1 and longest[0]['row_id'] == SELECTED_ROW_ID and longest[0]['supervised_tokens'] == 135, 'Longest-row selection differs')
    selected = longest[0]
    require(selected['prefix_tokens'] == 5421, 'Longest prefix length differs')
    selected_record = next(x for x in records if x['row_id'] == SELECTED_ROW_ID)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not any(x in sys.modules for x in ('torch', 'tensorflow', 'jax', 'peft')), 'CPU-only isolation changed')
    args.output.mkdir(parents=True, exist_ok=False)
    selected_path = args.output / 'selected-encoded.json'
    save_new(selected_path, selected)
    receipt = {
        'status': 'PASS_native_TF5202_CPU_patient_first_train124_token_contract',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
        'plan': plan_ref, 'data_contract_source': source_ref, 'sources': actual_sources,
        'tokenizer_small_files': actual_small,
        'runtime': {'python': sys.version.split()[0], 'transformers': transformers.__version__,
                    'tokenizers': tokenizers.__version__, 'native_tokenizer_class': type(tokenizer).__name__,
                    'auto_tokenizer_source': ref(inspect.getsourcefile(AutoTokenizer)),
                    'tokenizer_class_source': ref(inspect.getsourcefile(type(tokenizer))),
                    'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'model_backends_disabled': True,
                    'torch_imported': 'torch' in sys.modules, 'CUDA_calls': 0},
        'training_document_ids': train_docs, 'F1_document_ids_from_summary_only': f1_docs,
        'training_F1_overlap': 0, 'excluded_document_count': 32, 'rows': 124,
        'total_full_tokens': 400614, 'max_full_tokens': 5556, 'supervised_tokens': 4694,
        'all_rows_patient_first_token_audit_exact': True, 'matched_audit_fields': sorted(records[0]),
        'per_row_id_and_label_SHA_serialization': 'sha256(json.dumps(list).encode()); default separators, distinct from helper.json_sha',
        'records_helper_json_sha256': helper.json_sha(records),
        'encoded_helper_json_sha256': helper.json_sha(encoded_rows),
        'helper_json_sha_serialization': 'json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(comma,colon)).encode()',
        'selected_row': selected_record, 'selected_encoded': ref(selected_path),
        'selected_encoded_helper_json_sha256': helper.json_sha(selected),
        'truncation': False, 'assistant_only_mask_and_single_supervised_EOS': True,
        'no_old_encode_all_or_verify_sources_called': True, 'F1_files_read': 0,
        'old_training_rows_read': 0, 'no_weights_loaded': True, 'no_model_forward_training_or_quality': True,
        'synthetic_parser_tests_repeated': False,
    }
    save_new(args.output / 'cpu-receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'rows': 124, 'selected_row_id': SELECTED_ROW_ID,
                      'selected_tokens': 5556, 'selected_supervised_tokens': 135,
                      'selected_encoded': ref(selected_path), 'receipt': ref(args.output / 'cpu-receipt.json')}), flush=True)


if __name__ == '__main__':
    main()
