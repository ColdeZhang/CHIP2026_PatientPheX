"""Exact final-adapter row73 cache and durable full80 output; no ML or gold imports."""
import gzip
import json
import os
from pathlib import Path
from probe_contract import checked, ids_sha, ref
from raw_first import atomic


def fsync_chain(directory, root):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    assert directory == root or root in directory.parents
    while True:
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if directory == root:
            break
        directory = directory.parent


def check_cache(plan, rows, encoded, binding):
    assert plan['cached_indices'] == [73]
    assert plan['new_indices'] == [i for i in range(80) if i != 73]
    assert plan['RNG_policy'] == 'fresh_same_seed_per_row_not_continuous_RNG'
    cache = plan['final_adapter_cache']
    assert set(cache) == {'terminal', 'plan', 'result', 'ids', 'raw', 'row'}
    values = {key: json.loads(checked(value).read_text()) for key, value in cache.items()}
    terminal, oldplan, result = values['terminal'], values['plan'], values['result']
    assert terminal['exit_code'] == 0 and terminal['timeout_triggered'] is False
    assert terminal['plan'] == cache['plan']
    for key, relative in [('result', 'actual/result.json'), ('ids', 'actual/row73.ids.json'),
                          ('raw', 'actual/row73.raw.json'), ('row', 'actual/row73.json')]:
        assert terminal['files'][relative] == {k: cache[key][k] for k in ('bytes', 'sha256')}
    assert result['status'] == 'COMPLETE_SINGLE_FINAL_ADAPTER_LONGEST_FORMAT_COST_PROBE_NO_QUALITY'
    assert result['new_generations'] == 1 and result['quality_calls'] == 0 and result['original_index'] == 73
    assert result['format_pass'] and result['native_EOS_observed'] and result['normalized_valid']
    assert result['source']['sha256'] == plan['prior_probe_source_sha256']
    assert result['plan'] == cache['plan']
    assert oldplan['adapter_binding'] == result['adapter_binding'] == binding
    for key in ('seed', 'max_new_tokens', 'temperature', 'top_p', 'top_k', 'model_dir',
                'derived_config', 'derived_shapes', 'rotary', 'inputs', 'model_files'):
        assert oldplan[key] == plan[key]
    for name in ('full80_contract.py', 'normalized_final.py', 'sparse_final.py', 'strict_final.py', 'raw_first.py'):
        assert oldplan['sources'][name] == plan['sources'][name]
    item, raw, ids = values['row'], values['raw'], values['ids']
    assert item['original_index'] == raw['original_index'] == ids['original_index'] == 73
    assert item['row_id'] == raw['row_id'] == ids['row_id'] == rows[73]['row_id']
    assert item['input_ids_sha256'] == raw['input_ids_sha256'] == ids_sha(encoded[73]['input_ids'])
    assert ids['input_ids_sha256'] == ids_sha(encoded[73]['input_ids'])
    assert item['adapter_binding'] == raw['adapter_binding'] == ids['adapter_binding'] == binding
    assert item['seed'] == raw['seed'] == ids['seed'] == plan['seed']
    assert item['generated_ids'] == raw['generated_ids'] == ids['generated_ids']
    assert item['generated_tokens'] == raw['generated_tokens'] == ids['generated_tokens'] == len(ids['generated_ids'])
    assert ids['generated_ids'][-1] == 151645
    assert item['input_tokens'] == raw['input_tokens'] == ids['input_tokens'] == len(encoded[73]['input_ids'])
    assert item['native_EOS_observed'] and item['normalized_valid']
    assert not item['stopped_at_token_cap'] and not item['stopped_at_wall_budget']
    assert item['ids_reference'] == cache['ids'] and item['raw_reference'] == cache['raw']
    assert result['row'] == cache['row'] and result['ids'] == cache['ids'] and result['raw'] == cache['raw']
    return values


def copy_cached(plan, output, cached, decode):
    """Preserve original cache bytes; identify this run's reuse in separate receipt."""
    values = cached['ids']['generated_ids']
    assert decode(values) == cached['raw']['raw_generated_text']
    assert decode(values[:-1]) == cached['raw']['final_text_excluding_terminal_EOS']
    for name, suffix in [('ids', '.ids.json'), ('raw', '.raw.json'), ('row', '.json')]:
        source = checked(plan['final_adapter_cache'][name])
        target = Path(output) / ('row73' + suffix)
        with target.open('xb') as handle:
            handle.write(source.read_bytes()); handle.flush(); os.fsync(handle.fileno())
        assert ref(target)['sha256'] == plan['final_adapter_cache'][name]['sha256']
    fsync_chain(output, Path(plan['project_root']))
    atomic(Path(output) / 'row73.cache-use.json', dict(
        original_index=73, fresh_NN_in_this_run=False, original_row_bytes_preserved=True,
        provenance=plan['final_adapter_cache'], no_old_base_model_cache=True))
    return cached['row']


def write_responses(path, items):
    """Write one complete ordered freeze only after all80 immutable row files exist."""
    assert len(items) == 80 and [item['original_index'] for item in items] == list(range(80))
    path = Path(path)
    with path.open('xb') as handle:
        for item in items:
            handle.write((json.dumps(item, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode())
        handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ref(path)


def compress_responses(path):
    path = Path(path)
    destination = path.with_suffix(path.suffix + '.gz')
    with destination.open('xb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as compressed:
            compressed.write(path.read_bytes())
        handle.flush(); os.fsync(handle.fileno())
    assert gzip.decompress(destination.read_bytes()) == path.read_bytes()
    descriptor = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ref(destination)
