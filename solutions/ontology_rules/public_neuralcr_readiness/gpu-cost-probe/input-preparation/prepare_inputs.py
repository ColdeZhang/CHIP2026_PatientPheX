"""CPU-only deterministic prefix selection from the frozen native token-key file."""
from pathlib import Path
import ast
import collections
import copy
import datetime
import hashlib
import json
import re
import sys
import time

R = Path(__file__).resolve().parent
N = R.parent.parent
C = N / 'input-only-census'
Q = C / 'unique-normalized-queries.jsonl'
SOURCE = N / 'postdownload-readiness/ncrmodel-upstream.py'
sha = lambda b: hashlib.sha256(b).hexdigest()
ref = lambda p: {'path': str(p), 'bytes': p.stat().st_size,
                 'sha256': sha(p.read_bytes())}


def save(name, value):
    (R / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


assert not (R / 'result.json').exists(), 'Preserve an existing preparation result'
started = time.monotonic()
plan = {
    'status': 'prepared_CPU_only',
    'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Select exactly first 1536 existing rows in frozen file order; 3 consecutive batches of 512. No filtering, resorting, replacement, labels, model or quality.',
    'source': ref(Path(__file__)),
    'frozen_keys': ref(Q),
    'census_source': ref(C / 'census.py'),
    'census_plan': ref(C / 'plan.json'),
    'census_report': ref(C / 'report.json'),
    'native_source': ref(SOURCE),
    'expected_key_rows': 408272,
    'selection_zero_based': [0, 1536],
    'batches': 3,
    'batch_size': 512,
    'max_sequence_length': 50,
    'no_model_or_TensorFlow_import': True,
    'no_MCP_GPU_NN_fit_quality': True,
}
save('plan.json', plan)
assert plan['frozen_keys']['bytes'] == 16445155
assert plan['frozen_keys']['sha256'] == '1924f3691da766a81de963690d9a72bbd4f030ff34c3e191c082c64bc6ff7b96'
assert plan['native_source']['sha256'] == '7d196049361dd19d9e88601c2e91c21c055b63ce3f29afa17d8bdb85259e9aef'
census_report = json.loads((C / 'report.json').read_text())
census_plan = json.loads((C / 'plan.json').read_text())
assert census_report['status'] == 'pass_input_only_native_enumeration_no_model_or_quality'
assert census_report['artifact'] == plan['frozen_keys']
assert census_report['source_sha256'] == plan['census_source']['sha256']
assert census_report['plan_sha256'] == plan['census_plan']['sha256']
assert census_plan['native_source'] == plan['native_source']
assert census_report['unique_phrase2vec_token_keys'] == plan['expected_key_rows']
assert census_report['competition_labels_read'] is False

# Execute only the exact two pure token functions, never the model module.
tree = ast.parse(SOURCE.read_text())
functions = [copy.deepcopy(n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in ('is_number', 'tokenize')]
assert [n.name for n in functions] == ['is_number', 'tokenize']
module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
scope = {'re': re}
exec(compile(module, '<fixed NCR pure token functions>', 'exec'), scope)
tokenize = scope['tokenize']

keys = []
marker_counts = collections.Counter()
roundtrip_failures = []
length_counts = collections.Counter()
with Q.open() as f:
    for row_index, line in enumerate(f):
        row = json.loads(line)
        assert set(row) == {'query'} and isinstance(row['query'], str)
        key = row['query']
        assert 1 <= len(key.split(' ')) <= 7
        assert not keys or keys[-1] < key, 'Frozen census order must be strictly sorted'
        tokens = tokenize(key)[:49]
        if ' '.join(tokens) != key:
            roundtrip_failures.append({'row': row_index, 'key': key, 'actual_tokens': tokens})
        marker_counts.update(token for token in key.split(' ') if token in ('INT', 'FLOAT'))
        length_counts[len(tokens)] += 1
        keys.append(key)
assert len(keys) == plan['expected_key_rows']

general_marker_examples = []
for raw in ['123', 'nan', 'inf', 'INT', 'FLOAT']:
    first = tokenize(raw)
    second = tokenize(' '.join(first))
    general_marker_examples.append({'input': raw, 'first_tokens': first,
                                    'joined_then_retokenized': second,
                                    'idempotent': first == second})

# Any marker problem stops preparation; do not silently rewrite or skip entries.
if roundtrip_failures:
    save('result.json', {
        'status': 'blocked_actual_frozen_key_not_idempotent_no_inputs_written',
        'plan_sha256': sha((R / 'plan.json').read_bytes()),
        'actual_roundtrip_failure_count': len(roundtrip_failures),
        'examples': roundtrip_failures[:10],
        'general_marker_examples': general_marker_examples,
        'elapsed_seconds': time.monotonic() - started,
    })
    raise RuntimeError('Frozen normalized token key cannot be safely re-tokenized')

selected = keys[:1536]
assert len(selected) == len(set(selected)) == 1536
batches = [selected[i:i+512] for i in range(0, 1536, 512)]
assert [len(b) for b in batches] == [512, 512, 512]
assert [q for b in batches for q in b] == keys[:1536]
assert all(tokenize(q)[:49] == q.split(' ') for q in selected)
payload = {'selection': 'frozen file original rows [0,1536), no filtering or resorting',
           'batches': batches}
(R / 'probe-inputs.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
batch_refs = []
for i, batch in enumerate(batches):
    name = f'batch-{i:03d}.json'
    (R / name).write_text(json.dumps({'queries': batch}, ensure_ascii=False, separators=(',', ':')) + '\n')
    batch_refs.append({'batch_index': i, 'source_rows_zero_based': [i*512, (i+1)*512],
                       'query_count': 512, 'first_query': batch[0], 'last_query': batch[-1],
                       'token_length_counts': dict(sorted(collections.Counter(len(q.split(' ')) for q in batch).items())),
                       'artifact': ref(R / name)})
assert not any(n == 'tensorflow' or n.startswith('tensorflow.') for n in sys.modules)
assert not any(n == 'fasttext' or n.startswith('fasttext.') for n in sys.modules)

result = {
    'status': 'pass_CPU_only_three_frozen_512_batches',
    'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256': plan['source']['sha256'],
    'plan_sha256': sha((R / 'plan.json').read_bytes()),
    'pure_functions_AST_sha256': sha(ast.dump(module, include_attributes=False).encode()),
    'entire_frozen_file_rows_checked': len(keys),
    'entire_frozen_file_strict_sorted_unique': True,
    'entire_frozen_file_native_token_roundtrip_failures': 0,
    'entire_frozen_file_INT_FLOAT_marker_counts': dict(marker_counts),
    'entire_frozen_file_token_length_counts': dict(sorted(length_counts.items())),
    'selected_rows_zero_based': [0, 1536],
    'selected_count': 1536,
    'selected_native_token_keys_exact': True,
    'selected_keys_reordered_filtered_replaced': False,
    'probe_inputs': ref(R / 'probe-inputs.json'),
    'batch_artifacts': batch_refs,
    'marker_semantics': {
        'general_join_then_retokenize_is_not_idempotent': True,
        'examples': general_marker_examples,
        'observed_actual_corpus_has_no_INT_FLOAT_markers': not marker_counts,
        'actual_full_corpus_and_selected_prefix_safe': True,
        'reason': 'The actual native candidate generator splits on non-ASCII-letter/non-space characters, so numeric text is not directly enumerated; FLOAT can still arise from literal nan/inf, so all 408272 frozen keys were checked rather than assuming safety.',
        'representative_mapping': 'Census persists normalized keys only, not a key-to-original-query map. Its original four-field input manifest remains available for exact pre-NN re-enumeration if ever needed. No representative reconstruction was needed or performed here.',
    },
    'input_only_provenance': {
        'census_input': census_plan['input'],
        'original_input_opened_by_this_preparation': False,
        'gold_or_FN_files_read': False,
        'selection_uses_only_frozen_file_order': True,
        'no_MCP_GPU_NN_fit_quality': True,
        'no_TensorFlow_or_fasttext_import': True,
    },
    'elapsed_seconds': time.monotonic() - started,
}
save('result.json', result)
print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                  'rows_checked': len(keys), 'selected': len(selected),
                  'probe_inputs': result['probe_inputs'], 'actual_marker_counts': dict(marker_counts)}))
