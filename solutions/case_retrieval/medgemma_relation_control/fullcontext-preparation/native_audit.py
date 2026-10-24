#!/usr/bin/env python3
"""One native Med tokenizer CPU audit; no model, training, truncation or score."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
import construct
import medgemma_interface as interface

INTERFACE_SHA = 'aecef7eaab0565ab98ce76c034f01b543be3342b00a84d4bae71c36d614d55be'
ASSET_SHA = '690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1'
PINS = {'original_train': 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8',
        'original_infer': 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe',
        'train_input': 'a2c2baecb86ca425ae978688014073b69a415dbc9d3526d24be96f82b89e3e6d',
        'infer_input': '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'}
CAPS = (6144, 8192, 12288, 14000, 16384, 32768)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(directory, name, value):
    with (directory / name).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def aggregate(rows, field):
    values = [row[field] for row in rows]
    return {'rows': len(values), 'total': sum(values), 'min': min(values), 'median': statistics.median(values),
            'max': max(values), 'longest_row_id': max(rows, key=lambda row: row[field])['row_id'],
            'over_caps': {str(cap): sum(value > cap for value in values) for cap in CAPS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output-directory', required=True, type=Path)
    args = parser.parse_args()
    out, started, stage = args.output_directory, time.monotonic(), 'source_and_input_contract'
    out.mkdir(parents=True, exist_ok=False)
    try:
        assert sha(__file__) and sha(interface.__file__) == INTERFACE_SHA
        assert sha(args.source_manifest) == ASSET_SHA and sha(args.bundle) == args.bundle_sha256
        bundle = json.loads(gzip.decompress(Path(args.bundle).read_bytes()))
        assert bundle['schema'] == 'medgemma-same12480-fullcontext-budget-v1'
        for name, expected in PINS.items():
            assert construct.digest(bundle[name].encode()) == expected == bundle['sources'][name]['sha256'], name
        originals = {split: construct.rows(bundle['original_' + split]) for split in ('train', 'infer')}
        documents = {split: construct.rows(bundle[split + '_input']) for split in ('train', 'infer')}
        extended, coordinate_audits = {}, {}
        for split in ('train', 'infer'):
            generated, coordinate_audits[split], _ = construct.append_reports(originals[split], documents[split], split == 'train')
            extended[split] = construct.rows(bundle['extended_' + split])
            assert extended[split] == generated
        assert len(extended['train']) == 124 and len(extended['infer']) == 80
        assert len(documents['train']) == 32 and len(documents['infer']) == 16
        assert not {str(d['pmc_id']) for d in documents['train']} & {str(d['pmc_id']) for d in documents['infer']}
        baseline = bundle['baseline_train_audit']
        assert baseline['rows_count'] == 124 and baseline['total_full_tokens'] == 382000 and baseline['supervised_tokens'] == 7150
        assert baseline['interface_sha256'] == INTERFACE_SHA and baseline['model_source_manifest_sha256'] == ASSET_SHA
        assert baseline['train_sha256'] == PINS['original_train'] and baseline['truncated_rows'] == 0
        old_lengths = {row['row_id']: row for row in baseline['rows']}
        assert set(old_lengths) == {row['row_id'] for row in extended['train']}
        stage = 'verify_eight_tokenizer_assets'
        actual_assets = interface.model_files(args.model, args.source_manifest, False)
        assert len(actual_assets['actual_files']) == 8 and actual_assets['all19_files_verified'] is False
        stage = 'native_tokenizer_only'
        import torch
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
        assert transformers.__version__ == '5.20.2'
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, padding_side='right')
        train_records, infer_records = [], []
        for index, row in enumerate(extended['train']):
            encoded = interface.encode(row, tokenizer)
            prefix, ids, labels = encoded['prefix_tokens'], encoded['input_ids'], encoded['labels']
            assert labels[:prefix] == [-100] * prefix and labels[prefix:] == ids[prefix:]
            assert encoded['supervised_tokens'] == old_lengths[row['row_id']]['supervised_tokens']
            assert labels[encoded['eos_supervised_position']] == 106 and labels[prefix:].count(106) == 1
            train_records.append({'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']),
                                  'full_tokens': len(ids), 'prefix_tokens': prefix, 'supervised_tokens': encoded['supervised_tokens'],
                                  'token_ids_sha256': construct.jsha(ids), 'supervised_ids_sha256': construct.jsha(ids[prefix:]),
                                  'prefix_mask_eos_verified': True, 'original_answer_unchanged': True,
                                  'rendered_prefix_sha256': encoded['rendered_prefix_sha256'],
                                  'rendered_full_sha256': encoded['rendered_full_sha256']})
            if (index + 1) % 32 == 0:
                print(json.dumps({'stage': 'CPU_tokenizer', 'training_rows_completed': index + 1, 'elapsed_seconds': time.monotonic() - started}), flush=True)
        for row in extended['infer']:
            text, ids = interface.render_prefix(row, tokenizer)
            infer_records.append({'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']), 'input_tokens': len(ids),
                                  'input_plus_fixed1600': len(ids) + 1600, 'token_ids_sha256': construct.jsha(ids),
                                  'rendered_prefix_sha256': construct.digest(text.encode()), 'no_answer': 'answer' not in row})
        assert sum(row['supervised_tokens'] for row in train_records) == 7150
        assert all(row['no_answer'] for row in infer_records) and not torch.cuda.is_initialized()
        train_full = aggregate(train_records, 'full_tokens')
        infer_input = aggregate(infer_records, 'input_tokens')
        ratio = train_full['total'] / 382000
        old_epoch = bundle['measured_original_epoch1']
        assert old_epoch['completed_epochs'] == 1 and old_epoch['global_step'] == 16 and old_epoch['training_only'] is True
        epoch_loop_seconds = old_epoch['epoch_seconds']
        config = json.loads((Path(args.model) / 'config.json').read_text())
        summary = {'status': 'complete_native_CPU_fullcontext_budget_only', 'source_sha256': sha(__file__),
                   'construction_source_sha256': sha(construct.__file__), 'legacy_source_sha256': construct.LEGACY_SHA,
                   'interface_sha256': INTERFACE_SHA, 'bundle_sha256': args.bundle_sha256,
                   'model_asset_manifest_sha256': ASSET_SHA, 'actual_tokenizer_files': actual_assets['actual_files'],
                   'native_runtime': {'python': sys.version, 'transformers': transformers.__version__, 'torch': torch.__version__,
                                      'tokenizers': tokenizers.__version__, 'tokenizer_class': type(tokenizer).__name__},
                   'train_full_tokens': train_full, 'train_prefix_tokens': aggregate(train_records, 'prefix_tokens'),
                   'train_supervised_tokens': aggregate(train_records, 'supervised_tokens'),
                   'inference_input_tokens': infer_input, 'inference_input_plus_fixed1600': aggregate(infer_records, 'input_plus_fixed1600'),
                   'baseline': {'train_full_tokens': 382000, 'train_max': 5380, 'supervised_tokens': 7150,
                                'infer_input_tokens': bundle['baseline_infer_audit']['total_tokens'],
                                'infer_input_max': bundle['baseline_infer_audit']['max_tokens'],
                                'actual_original_epoch1_loop_seconds': epoch_loop_seconds,
                                'actual_original_epoch1_reference': bundle['sources']['measured_original_epoch1']},
                   'training_token_ratio_vs_382000': ratio,
                   'inference_input_token_ratio': infer_input['total'] / bundle['baseline_infer_audit']['total_tokens'],
                   'cost_proxy_not_measured': {'two_epoch_full_tokens': 2 * train_full['total'],
                                             'epoch_loop_seconds_linear_token_proxy': epoch_loop_seconds * ratio,
                                             'two_epoch_loop_seconds_linear_token_proxy': 2 * epoch_loop_seconds * ratio,
                                             'not_actual_fullcontext_training_time': True,
                                             'limitations': 'Attention length and allocator can be nonlinear; no memory-fit or latency guarantee; excludes model loads, checkpoints, durable export and inference. Do not multiply the 118.342s export by token ratio.'},
                   'existing_caps_unchanged': {'training_sequence_cap': 8192, 'inference_input_cap': 14000, 'inference_new_tokens': 1600,
                                             'over_training_cap': train_full['over_caps']['8192'],
                                             'over_inference_cap': infer_input['over_caps']['14000'],
                                             'config_text_max_position_embeddings': config['text_config']['max_position_embeddings']},
                   'coordinate_summary': {split: {'documents': len(values), 'patients': sum(len(d['patient_ids']) for d in values),
                                                 'passages': sum(len(d['passages']) for d in values),
                                                 'given_mentions': sum(d['given_mentions'] for d in values),
                                                 'all_original_text_reconstructed': all(d['original_text_reconstruction_exact'] for d in values),
                                                 'all_given_patients_marked': all(d['all_patients_marked'] for d in values)}
                                          for split, values in coordinate_audits.items()},
                   'supervision_boundary_passed_all124': True, 'all_answers_CID_PID_candidates_original_messages_preserved': True,
                   'no_truncation': True, 'model_weights_loaded': False, 'cuda_initialized': torch.cuda.is_initialized(),
                   'F1_gold_opened': False, 'training_or_prediction_or_score_performed': False,
                   'elapsed_CPU_seconds': time.monotonic() - started}
        save(out, 'row-token-audit.json', {'train': train_records, 'infer': infer_records})
        summary['row_token_audit_sha256'] = sha(out / 'row-token-audit.json')
        save(out, 'summary.json', summary)
        print(json.dumps({'complete': True, 'summary_sha256': sha(out / 'summary.json'), 'train_full_tokens': train_full['total'],
                          'max_train_tokens': train_full['max'], 'infer_input_tokens': infer_input['total'],
                          'max_infer_tokens': infer_input['max'], 'elapsed_CPU_seconds': summary['elapsed_CPU_seconds']}), flush=True)
    except BaseException as error:
        save(out, 'failure.json', {'stage': stage, 'error_type': type(error).__name__, 'error_text': str(error),
                                   'elapsed_seconds': time.monotonic() - started, 'source_sha256': sha(__file__),
                                   'not_a_quality_result': True, 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    main()
