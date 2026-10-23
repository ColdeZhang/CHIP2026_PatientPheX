#!/usr/bin/env python3
"""CPU-only strict checkpoint and frozen training-window compatibility check."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import run_gliner_pilot
import train_gliner_supervised
from run_gliner_pilot import sha


def load(model_dir, tokenizer_dir):
    """Load the saved current-format vocabulary without legacy FLERT migration."""
    import torch
    from transformers import AutoTokenizer
    from gliner import GLiNER, GLiNERConfig
    raw = json.loads((model_dir/"gliner_config.json").read_text())
    assert raw["encoder_config"]["vocab_size"] == raw["vocab_size"] == 128003
    assert raw["class_token_index"] == 128001
    config = GLiNERConfig(**raw)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.model_max_length = config.max_len
    assert len(tokenizer) == config.vocab_size
    assert tokenizer.convert_tokens_to_ids(config.ent_token) == config.class_token_index
    assert tokenizer.convert_tokens_to_ids(config.sep_token) == 128002
    assert "[FLERT]" not in tokenizer.get_vocab()
    model = GLiNER(config, tokenizer=tokenizer, encoder_from_pretrained=False)
    state = torch.load(model_dir/"pytorch_model.bin", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(t.numel() for t in state.values())
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir', 'tokenizer-dir', 'expected-weight-sha',
                 'data-manifest', 'train', 'excluded-input', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    model_dir, tokenizer_dir = Path(args.model_dir), Path(args.tokenizer_dir)
    assert sha(model_dir/'pytorch_model.bin') == args.expected_weight_sha
    data = json.loads(Path(args.data_manifest).read_text())
    expected = data['artifacts']['train48']
    assert sha(args.train) == expected['sha256']
    assert sha(args.excluded_input) == data['files']['excluded_input']
    rows = [json.loads(x) for x in Path(args.train).read_text().splitlines()]
    excluded = [json.loads(x) for x in Path(args.excluded_input).read_text().splitlines()]
    assert all('entities' not in d and 'association' not in d for d in excluded)
    ids = {str(r['pmc_id']) for r in rows}
    assert ids == set(expected['pmc_ids']) and len(ids) == 48
    assert not ids & {str(d['pmc_id']) for d in excluded}
    model = load(model_dir, tokenizer_dir)
    parameters = sum(p.numel() for p in model.parameters())
    assert 0 < parameters < 10_000_000_000
    assert model.config.max_width == 12 and model.config.max_len >= 384
    counts = {'rows': len(rows), 'documents': len(ids), 'positive_instances': 0,
              'valid_spans': 0, 'max_tokens': 0}
    for offset in range(0, len(rows), 4):
        # This is the actual trainer collator, including exact positive indices,
        # NO targets, word masks and the complete non-truncated token budget.
        batch = train_gliner_supervised.collate(model.data_processor, rows[offset:offset+4])
        counts['positive_instances'] += int((batch['labels'] > 0).sum())
        counts['valid_spans'] += int(batch['span_mask'].sum())
        counts['max_tokens'] = max(counts['max_tokens'], int(batch['attention_mask'].sum(-1).max()))
    assert counts['max_tokens'] <= 384
    assert not torch.cuda.is_initialized()
    record = {
        'strict_state_load': True, 'parameters': parameters, 'cuda_initialized': False,
        'vocabulary': {'size': len(model.data_processor.transformer_tokenizer), 'class_token_index': model.config.class_token_index, 'legacy_resize_applied': False},
        'source_max_len': model.config.max_len, 'prepared_window_token_cap': 384,
        'actual_all_training_collation_pass': True, 'counts': counts,
        'seconds': time.monotonic()-started, 'training_fit': False,
        'excluded_labels_read': False, 'arguments': vars(args),
        'data_origin_note': 'Prepared data records the medium checkpoint used to define windows; '
                            'this check validates all those frozen windows with the new actual checkpoint/tokenizer.',
        'sources': {'checker': sha(__file__), 'loader': sha(__file__), 'hash_helper': sha(run_gliner_pilot.__file__),
                    'trainer_collator': sha(train_gliner_supervised.__file__)},
        'artifacts': {role: {'path': str(path), 'sha256': sha(path)} for role, path in {
            'train': args.train, 'data_manifest': args.data_manifest, 'excluded_input': args.excluded_input,
            'weight': model_dir/'pytorch_model.bin', 'gliner_config': model_dir/'gliner_config.json',
            'added_tokens': tokenizer_dir/'added_tokens.json',
            'special_tokens': tokenizer_dir/'special_tokens_map.json',
            'tokenizer_json': tokenizer_dir/'tokenizer.json', 'spm': tokenizer_dir/'spm.model',
            'tokenizer_config': tokenizer_dir/'tokenizer_config.json'}.items()},
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch', 'transformers', 'gliner', 'tokenizers', 'sentencepiece']},
    }
    Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'strict_state_load': True, 'parameters': parameters,
                      'all_training_collation_pass': True, 'counts': counts,
                      'seconds': record['seconds'], 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
