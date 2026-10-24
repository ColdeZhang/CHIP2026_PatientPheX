"""Small durable raw-before-parser boundary; no model, tokenizer or data at import."""
from pathlib import Path
import hashlib
import json
import os
import traceback


def atomic(path, value, *, replace=False):
    path = Path(path)
    assert replace or not path.exists(), 'Immutable row evidence already exists'
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    assert replace or not path.exists()
    os.replace(temporary, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def persist_fresh(output, index, original, decode, parse_item, prompt):
    """IDs always precede decode; immutable decoded raw always precedes any parser."""
    output = Path(output)
    ids_path = output / ('row%d.ids.json' % index)
    atomic(ids_path, original)
    ids_ref = ref(ids_path)
    assert original['input_prefix_exact'] and original['generated_length_within_cap']
    item = dict(original)
    try:
        ids = item['generated_ids']
        raw = decode(ids)
        content = decode(ids[:-1] if item['native_EOS_observed'] else ids)
    except BaseException as error:
        atomic(output / ('row%d.decode-error.json' % index), {
            'status': 'UNEXPECTED_DECODE_ERROR_COMPLETE_IDS_PRESERVED', 'ids_reference': ids_ref,
            'error': repr(error), 'traceback': traceback.format_exc(), 'quality_evaluated': False})
        raise
    item.update(raw_generated_text=raw, final_text_excluding_terminal_EOS=content,
                think_open_count=raw.count('<think>'), think_close_count=raw.count('</think>'),
                first_think_close_character_offset=raw.find('</think>'))
    raw_path = output / ('row%d.raw.json' % index)
    atomic(raw_path, item)
    raw_ref = ref(raw_path)
    try:
        parsed = parse_item(dict(item), prompt)
    except BaseException as error:
        atomic(output / ('row%d.parse-error.json' % index), {
            'status': 'UNEXPECTED_PARSER_ERROR_COMPLETE_RAW_PRESERVED', 'raw_reference': raw_ref,
            'ids_reference': ids_ref, 'error': repr(error), 'traceback': traceback.format_exc(),
            'quality_evaluated': False})
        assert ref(raw_path) == raw_ref
        raise
    assert ref(raw_path) == raw_ref and ref(ids_path) == ids_ref
    parsed['raw_reference'] = raw_ref
    parsed['ids_reference'] = ids_ref
    atomic(output / ('row%d.json' % index), parsed)
    return parsed
