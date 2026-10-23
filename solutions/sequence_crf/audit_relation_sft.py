#!/usr/bin/env python3
"""Audit every untruncated SFT row with the real model chat tokenizer on CPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    for name in ['trainer', 'train', 'excluded-input', 'inference-prompts', 'model', 'output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('trainer', a.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    rows, ids = trainer.load_training_rows(a.train, a.excluded_input)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    lengths, prefixes, supervised = [], [], []
    longest = None
    for row in rows:
        answer = json.loads(row['answer'])
        assert set(answer) == set(row['code_map'])
        assert all(isinstance(value, list) and set(value) <= set(row['patient_ids']) for value in answer.values())
        encoded = trainer.encode_example(row, tokenizer, 32768)
        prefix = encoded['prefix_tokens']
        assert all(label == -100 for label in encoded['labels'][:prefix])
        assert encoded['labels'][prefix:] == encoded['input_ids'][prefix:]
        decoded = tokenizer.decode(encoded['input_ids'][prefix:], skip_special_tokens=True).strip()
        assert json.loads(decoded) == answer
        lengths.append(len(encoded['input_ids']))
        prefixes.append(prefix)
        supervised.append(encoded['supervised_tokens'])
        if longest is None or len(encoded['input_ids']) > longest['tokens']:
            longest = {'row_id': row['row_id'], 'tokens': len(encoded['input_ids'])}
    inf_lengths = []
    for row in trainer.read_jsonl(a.inference_prompts):
        tokens = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        inf_lengths.append(len(tokens))
    result = {'rows': len(rows), 'documents': len(ids), 'all_prompt_tokens_masked': True,
              'all_completion_tokens_supervised': True, 'all_answers_roundtrip_exactly': True,
              'truncation_count': 0, 'max_train_tokens': max(lengths), 'mean_train_tokens': statistics.mean(lengths),
              'max_prefix_tokens': max(prefixes), 'max_completion_tokens': max(supervised),
              'min_completion_tokens': min(supervised), 'total_train_tokens': sum(lengths),
              'longest_sample': longest, 'inference_prompts': len(inf_lengths),
              'max_inference_tokens': max(inf_lengths), 'source_sha256': trainer.sha256(__file__),
              'files': {name: trainer.sha256(getattr(a, name)) for name in ['trainer', 'train', 'excluded_input', 'inference_prompts']},
              'model_config_sha256': trainer.sha256(Path(a.model) / 'config.json')}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
