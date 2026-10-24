"""Bind only the completed new reverse probe cache; preserve its raw bytes."""
import gzip
import json
import os
from pathlib import Path
from common import checked, ref, fsync_chain


def same(actual, expected):
    assert all(actual[key] == expected[key] for key in ('bytes', 'sha256'))


def check_cache(plan, rows, audit):
    refs = plan['probe_references']
    value = {name: json.loads(checked(record).read_text()) for name, record in refs.items()}
    terminal, old, result = value['terminal'], value['plan'], value['summary']
    assert terminal['exit_code'] == 0 and not terminal['timeout_triggered']
    assert terminal['target_UUID_has_no_compute_process']
    assert result['status'] == 'PASS_NATIVE_LONGEST_STRICT_COMPLETE' and result['fresh_rows'] == 1
    assert result['original_index'] == old['probe_index'] == 19 and result['quality_calls'] == 0
    assert result['source']['sha256'] == plan['new_reverse_probe_source_sha256']
    assert old['sources']['runner.py']['sha256'] == plan['new_reverse_probe_source_sha256']
    for name in ('model', 'adapter', 'adapter_files', 'CPU_result', 'CPU_audit', 'full80_prompts',
                 'CPU_source_sha256', 'CPU_contract_sha256', 'runtime_sha256', 'generation', 'input_limit', 'generation_limit'):
        assert old[name] == plan[name]
    for name in ('codec.py', 'legacy_runner.py'):
        assert old['sources'][name] == plan['sources'][name]
    same(refs['plan'], terminal['plan']); same(refs['plan'], result['plan'])
    for name, relative in [('summary','actual/summary.json'), ('assets','actual/actual-input-and-assets.json'),
                           ('generation','actual/generation-config.json'), ('ids','actual/generated-ids.json'),
                           ('raw','actual/raw-response.json'), ('format','actual/format-result.json')]:
        same(refs[name], terminal['files'][relative])
    ids, raw, form = value['ids'], value['raw'], value['format']
    assert ids['row_id'] == raw['row_id'] == rows[19]['row_id']
    assert ids['original_index'] == raw['original_index'] == 19
    assert ids['input_ids_sha256'] == audit[19]['reverse']['input_ids_sha256']
    assert raw['input_tokens'] == result['input_tokens'] == audit[19]['reverse']['tokens']
    assert ids['natural_eos'] and raw['natural_eos'] and result['natural_eos']
    assert not any(x['hit_token_limit'] or x['wall_stop'] for x in (ids,raw,result))
    assert ids['configured_eos_ids'] == [248044,248046] and ids['generated_ids'][-1] in ids['configured_eos_ids']
    assert len(ids['generated_ids']) == raw['output_tokens'] == result['generated_tokens']
    assert form['strict_complete'] and form['error'] is None and form['fallback'] is False
    assert set(form['parsed']) == set(rows[19]['code_map'])
    assert all(type(pid) is str and pid in rows[19]['patient_ids']
               for values in form['parsed'].values() for pid in values)
    assert value['assets']['adapter_files'] == {name: {'bytes': value['assets']['adapter_files'][name]['bytes'], 'sha256': digest}
                                                for name,digest in plan['adapter_files'].items()}
    assert value['assets']['runtime']['matmul_TF32'] is False and value['assets']['runtime']['cudnn_TF32'] is True
    assert value['generation']['configured_eos_ids'] == [248044,248046]
    return value


def copy_exact(source, target, root):
    source = checked(source); target = Path(target)
    with target.open('xb') as handle:
        handle.write(source.read_bytes()); handle.flush(); os.fsync(handle.fileno())
    same(ref(target), ref(source)); fsync_chain(target.parent, root)
    return ref(target)


def jsonl_with_gzip(path, values):
    path = Path(path)
    data = ''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in values).encode()
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    archive = path.with_suffix(path.suffix+'.gz')
    with archive.open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as zipped: zipped.write(data)
        handle.flush(); os.fsync(handle.fileno())
    assert gzip.decompress(archive.read_bytes()) == data
    descriptor=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return {'plain':ref(path),'gzip':ref(archive)}
