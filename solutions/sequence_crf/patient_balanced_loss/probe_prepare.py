"""Encode exactly the twelve authorized probe rows; reuse frozen row metadata."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

INDICES = [62, 108, 84, 87, 63, 93, 106, 78, 65, 85, 80, 6]


def ref(path):
    p = Path(path).resolve()
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sync(path):
    while True:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if path.parent == path:
            break
        path = path.parent


def publish(path, data):
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    sync(path.parent)


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def run(plan_path, plan, output):
    started = time.monotonic()
    assert plan['schema'] == 'patient-balanced-alpha05-probe12-preparation-v1'
    assert plan['source'] == ref(__file__)
    assert plan['indices'] == INDICES and len(set(INDICES)) == 12
    assert plan['P'] == 87 and plan['S'] == 16 and plan['group_size'] == 8
    assert 'torch' not in sys.modules
    refs = plan['references']
    allowed_jsonl = {str(Path(refs[k]['path']).resolve()) for k in ['train', 'row_weights']}
    def audit(event, values):
        if event == 'import' and values[0].split('.')[0] in {'torch', 'transformers', 'peft'}:
            raise PermissionError('Probe input encoding must not import Torch or models')
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(values[0])).resolve())
            if path.endswith('.jsonl') and (values[1] is None or 'r' in values[1]) and path not in allowed_jsonl:
                raise PermissionError('Only original train124 and existing row metadata may be read')
    sys.addaudithook(audit)
    for item in refs.values():
        assert ref(item['path']) == item
    preparation = json.loads(Path(refs['preparation_manifest']['path']).read_text())
    assert preparation['rows'] == 124 and preparation['patients'] == 87 and preparation['groups_per_epoch'] == 16
    assert preparation['full_tokens'] == 400614 and preparation['supervised_tokens'] == 4694
    for key, filename in [('row_weights', 'row-weights.jsonl'), ('epoch_groups', 'epoch-groups.json')]:
        assert preparation['outputs'][filename] == refs[key]
    train = [json.loads(line) for line in Path(refs['train']['path']).read_text().splitlines()]
    audits = json.loads(Path(refs['token_audit']['path']).read_text())
    metadata_lines = Path(refs['row_weights']['path']).read_bytes().splitlines(keepends=True)
    metadata = [json.loads(line) for line in metadata_lines]
    assert len(train) == len(audits) == len(metadata) == 124
    assert [r['row_id'] for r in train] == [r['row_id'] for r in audits] == [r['row_id'] for r in metadata]
    groups = json.loads(Path(refs['epoch_groups']['path']).read_text())
    first_epoch = groups['epochs'][0]
    assert first_epoch['epoch'] == 1 and first_epoch['seed'] == 20261023
    assert first_epoch['groups'][0]['row_indices'] == INDICES[:8]
    assert first_epoch['groups'][1]['row_indices'][:3] == INDICES[8:11]
    assert first_epoch['groups'][0]['group_size'] == first_epoch['groups'][1]['group_size'] == 8
    assert max(range(124), key=lambda i: metadata[i]['sequence_length']) == 6
    assert metadata[6]['sequence_length'] == 5556 and metadata[6]['prefix_length'] == 5421
    assert len(metadata[6]['supervised_label_positions']) == 135
    sys.path.insert(0, plan['cpu_libraries'])
    tree = ast.parse(Path(refs['tokenizer_helper']['path']).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FrozenTokenizer']
    assert len(nodes) == 1
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original-FrozenTokenizer-only', 'exec'), namespace)
    tokenizer = namespace['FrozenTokenizer'](refs['tokenizer']['path'], refs['template']['path'])
    encoded, evidence = [], []
    for index in INDICES:
        row, expected, meta = train[index], audits[index]['patient_first'], metadata[index]
        assert meta['row_index'] == index and meta['row_id'] == row['row_id']
        prefix = tokenizer.apply_chat_template(row['messages'], False, True, False)
        full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': row['answer']}], False, False, False)
        assert full.startswith(prefix) and full[len(prefix):] == row['answer'] + '<|im_end|>\n'
        prefix_ids = tokenizer.backend.encode(prefix, add_special_tokens=False).ids
        ids = tokenizer.backend.encode(full, add_special_tokens=False).ids
        assert ids[:len(prefix_ids)] == prefix_ids
        labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
        actual = dict(row_id=row['row_id'], pmc_id=str(row['pmc_id']), full_tokens=len(ids),
            prefix_tokens=len(prefix_ids), supervised_tokens=len(ids) - len(prefix_ids), eos_token_id=tokenizer.eos_token_id,
            rendered_prefix_sha256=text_sha(prefix), rendered_full_sha256=text_sha(full),
            input_ids_sha256=digest(ids), labels_sha256=digest(labels))
        assert actual == expected == meta['native_audit']
        q = [i for i, label in enumerate(labels) if label != -100]
        assert q == meta['supervised_label_positions']
        assert [i - 1 for i in q] == meta['supervised_logit_positions']
        assert len(ids) == meta['sequence_length'] <= 6144 and len(prefix_ids) == meta['prefix_length']
        assert all(set(p['value_label_positions']) <= set(q) and p['n_p'] > 0 for p in meta['patients'])
        encoded.append(dict(row_metadata=meta, original_index=index, row_id=row['row_id'],
            prefix_tokens=len(prefix_ids), supervised_tokens=len(q), input_ids=ids, labels=labels, group_size=8))
        evidence.append(dict(original_index=index, row_id=row['row_id'], native_audit=actual,
            original_row_metadata_line_sha256=hashlib.sha256(metadata_lines[index]).hexdigest(),
            original_row_metadata_object_unchanged=True))
    assert len(encoded) == 12 and [r['original_index'] for r in encoded] == INDICES
    assert sum(len(r['input_ids']) for r in encoded) == plan['expected_probe_full_tokens']
    assert sum(r['supervised_tokens'] for r in encoded) == plan['expected_probe_supervised_tokens']
    payload = dict(schema='patient-balanced-probe12-inputs-v1', P=87, S=16, group_size=8, rows=encoded)
    data = (json.dumps(payload, ensure_ascii=False) + '\n').encode()
    compressed = gzip.compress(data, compresslevel=6, mtime=0)
    destination = output / 'probe-inputs.json.gz'
    publish(destination, compressed)
    assert gzip.decompress(destination.read_bytes()) == data
    assert json.loads(gzip.decompress(destination.read_bytes())) == payload
    assert 'torch' not in sys.modules
    for item in refs.values():
        assert ref(item['path']) == item
    save(output / 'manifest.json', dict(status='PASS_ORIGINAL_TWELVE_PROBE_ROWS_NATIVE_EXACT',
        plan=ref(plan_path), source=ref(__file__), references=refs, payload=ref(destination),
        payload_uncompressed_sha256=hashlib.sha256(data).hexdigest(), rows=12, indices=INDICES, P=87, S=16, group_size=8,
        native_audits=evidence, row_metadata_preserved_whole=True,
        full_tokens=sum(len(r['input_ids']) for r in encoded), supervised_tokens=sum(r['supervised_tokens'] for r in encoded),
        only_twelve_rows_encoded=True, no_retokenization_of_other112_rows=True, no_recount_of_n_p=True,
        original_first_group8_then_next3_then_longest6=True, formal_training_order_unchanged=True,
        no_Torch_model_NN_fit_or_score=True, no_heldout_gold_or_predictions=True,
        files_and_ancestor_directories_fsynced=True, gzip_readback_exact=True, seconds=time.monotonic() - started))
    print(json.dumps(ref(output / 'manifest.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    assert __debug__
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    assert plan['cpu_timeout_seconds'] == 120
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'launch.json', dict(plan=ref(args.plan), source=ref(__file__), only_twelve_rows=True, automatic_retry=False))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe12 CPU encoding budget exceeded')))
    signal.alarm(120)
    try:
        run(args.plan, plan, output)
    except BaseException:
        save(output / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
