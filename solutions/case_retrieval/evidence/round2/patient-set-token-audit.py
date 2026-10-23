"""CPU-only exact trainer encoding and held-out-input token budget audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--trainer', required=True)
parser.add_argument('--train', required=True)
parser.add_argument('--excluded-input', required=True)
parser.add_argument('--prompts', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path.insert(0, str(Path(args.trainer).parent))
import train_lora
from transformers import AutoTokenizer

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summary(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values),
                mean=statistics.mean(values), total=sum(values))

tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
rows, documents = train_lora.load_training_rows(args.train, args.excluded_input)
records = []
failures = []
for row in rows:
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': row['answer']}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False)
    assert full[:len(prefix)] == prefix
    try:
        encoded = train_lora.encode_example(row, tokenizer, 12288)
        assert len(encoded['input_ids']) == len(full)
        assert all(x == -100 for x in encoded['labels'][:len(prefix)])
        assert encoded['labels'][len(prefix):] == full[len(prefix):]
    except ValueError as error:
        failures.append({'row_id': row['row_id'], 'error': str(error)})
    records.append(dict(row_id=row['row_id'], pmc_id=row['pmc_id'], patient_id=row['patient_id'],
                        prefix_tokens=len(prefix), answer_tokens=len(full)-len(prefix), full_tokens=len(full)))
inference = []
for row in train_lora.read_jsonl(args.prompts):
    assert 'answer' not in row
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    inference.append(dict(row_id=row['row_id'], pmc_id=row['pmc_id'], patient_id=row['patient_id'],
                          prefix_tokens=len(prefix)))
max_answer = max(r['answer_tokens'] for r in records)
max_new = max(1024, math.ceil(1.5 * max_answer / 256) * 256)
context_limit = json.loads((Path(args.model)/'config.json').read_text()).get('max_position_embeddings')
result = dict(scope='Actual quoted patient-set targets; train48 labels and held-out input only; no model/GPU loaded',
    configuration=vars(args), source_sha256=sha(__file__), trainer_sha256=sha(args.trainer),
    files={k: sha(getattr(args, k)) for k in ['train', 'excluded_input', 'prompts']},
    tokenizer_sha256=sha(Path(args.model)/'tokenizer.json'),
    model_config_sha256=sha(Path(args.model)/'config.json'),
    train=dict(rows=len(rows), documents=len(documents),
               **{key: summary([r[key+'_tokens'] for r in records]) for key in ['prefix','answer','full']},
               all_prefixes_match=True, encode_example_maxseq12288_failures=failures,
               records=records),
    inference=dict(rows=len(inference), documents=len(set(r['pmc_id'] for r in inference)),
                   prefix=summary([r['prefix_tokens'] for r in inference]), records=inference),
    recommended_max_new_tokens=max_new,
    generation_budget_rule='max(1024, ceil(1.5 * max_train_answer_tokens / 256) * 256)',
    inference_max_input_plus_generation=max(r['prefix_tokens'] for r in inference)+max_new,
    model_context_limit=context_limit, truncation=False, cuda_visible_devices='')
Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: ({a:b for a,b in v.items() if a != 'records'} if k in ['train','inference'] else v)
                  for k,v in result.items()}), flush=True)
