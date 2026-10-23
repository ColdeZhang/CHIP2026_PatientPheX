#!/usr/bin/env python3
"""Strict CPU checkpoint load and full context input interface gate; no scores."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import ner_context
import ner_context_runtime as runtime
from ner_context_runtime import sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['raw-train', 'train', 'input', 'model-dir', 'tokenizer-dir', 'reference-trainer', 'output']:
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    destination = Path(args.output); assert not destination.exists()
    assert sha(args.train) == runtime.PREPARED_TRAIN_SHA
    assert sha(args.input) == runtime.HELD_INPUT_SHA
    assert sha(args.reference_trainer) == runtime.REFERENCE_TRAINER_SHA
    sys.path.insert(0, str(Path(args.reference_trainer).parent))
    spec = importlib.util.spec_from_file_location('reference_context_training', args.reference_trainer)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    inputs = runtime.training_inputs(args.raw_train)
    rows = runtime.read_rows(args.train)
    held = runtime.read_rows(args.input)
    assert all('entities' not in d and 'association' not in d for d in held)
    assert len(held) == len({str(d['pmc_id']) for d in held}) == 16
    assert not {d['pmc_id'] for d in inputs} & {str(d['pmc_id']) for d in held}
    model = runtime.load_model(args.model_dir, args.tokenizer_dir, runtime.BIO_BASE_SHA)
    _, _, nominal, files = runtime.native_processor(args.model_dir, args.tokenizer_dir)
    assert model.config.to_dict() == nominal.config.to_dict()
    processor = model.data_processor
    builder = ner_context.ContextBuilder(inputs, processor.transformer_tokenizer, model.config)
    rebuilt = builder.input_rows()
    assert len(rows) == len(rebuilt) == 1150
    for row, unlabelled in zip(rows, rebuilt):
        assert all(row[key] == value for key, value in unlabelled.items())
    counts = Counter()
    for index in range(0, len(rows), 4):
        batch = rows[index:index+4]
        old = reference.collate(processor, batch)
        off = ner_context.collate(processor, batch, builder, context=False)
        on = ner_context.collate(processor, batch, builder)
        infer = ner_context.collate(processor, rebuilt[index:index+4], builder, prepare_labels=False)
        assert set(old) == set(off) == set(on)
        for key in old:
            assert torch.equal(old[key], off[key]), key
        for key in ['labels', 'span_idx', 'span_mask', 'text_lengths']:
            assert torch.equal(old[key], on[key]), key
        for key in infer:
            assert torch.equal(on[key], infer[key]), key
        counts['rows'] += len(batch)
        counts['positive_instances'] += int((on['labels'] > 0).sum())
        counts['valid_target_spans'] += int(on['span_mask'].sum())
        for i, row in enumerate(batch):
            counts['NO_instances'] += len(row['no_span_indices'])
            assert (on['labels'].reshape(len(batch), -1)[i, row['no_span_indices']] == 0).all()
            counts['rows_with_context'] += int(on['attention_mask'][i].sum()) > row['tokens']
    assert dict(counts) == {'rows':1150, 'positive_instances':2236,
        'valid_target_spans':1125538, 'NO_instances':87, 'rows_with_context':1121}
    validation_builder = ner_context.ContextBuilder(held, processor.transformer_tokenizer, model.config)
    validation_rows = validation_builder.input_rows()
    input_stats = Counter(rows=len(validation_rows), documents=16)
    for index in range(0, len(validation_rows), 4):
        batch = validation_rows[index:index+4]
        tensors = ner_context.collate(processor, batch, validation_builder, prepare_labels=False)
        assert 'labels' not in tensors
        for i, row in enumerate(batch):
            total = int(tensors['attention_mask'][i].sum())
            assert row['tokens'] <= total <= 384
            assert int(tensors['words_mask'][i].max()) == len(row['tokenized_text'])
            input_stats['target_tokens'] += row['tokens']
            input_stats['combined_tokens'] += total
            input_stats['rows_with_context'] += total > row['tokens']
    selection = runtime.smoke_selection(rows, builder)
    selected = [rows[item['index']] for item in selection]
    tensors = ner_context.collate(processor, selected, builder)
    assert tensors['attention_mask'].sum(-1).tolist() == [384]*4
    probe_positive = int((tensors['labels'] > 0).sum())
    probe_valid = int(tensors['span_mask'].sum())
    assert 0 < probe_positive < probe_valid
    assert not torch.cuda.is_initialized()
    report = {'status':'PASS_actual_CPU_strict_model_and_full_context_inputs',
        'source_dependencies': runtime.source_fingerprint(__file__),
        'reference_trainer_sha256': sha(args.reference_trainer), 'model_config_tokenizer': files,
        'data_sha256': {k:sha(getattr(args,k)) for k in ['raw_train','train','input']},
        'weight_sha256': runtime.BIO_BASE_SHA, 'strict_state_load':True,
        'model_parameters':sum(p.numel() for p in model.parameters()), 'counts':dict(counts),
        'label_free_held_input':dict(input_stats), 'smoke_selection':selection,
        'smoke_supervision': {'positive':probe_positive, 'valid_negative':probe_valid-probe_positive,
                             'valid_total':probe_valid},
        'checks': {'all1150_original_off_tensors_exact':True,'all1150_on_labels_NO_geometry_unchanged':True,
            'all1150_input_only_reconstruction_and_infer_tensors_exact':True,
            'all594_label_free_held_windows_actual_collated':len(validation_rows)==594,
            'selected_longest_batch384_has_nonempty_context':True},
        'cuda_initialized':False,'full_model_forward':False,'heldout_gold_read':False,
        'prediction_or_score_generated':False,'seconds':time.monotonic()-started}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':dict(counts),'parameters':report['model_parameters'],
        'seconds':report['seconds'],'sha256':sha(destination)}),flush=True)


if __name__ == '__main__':
    main()
