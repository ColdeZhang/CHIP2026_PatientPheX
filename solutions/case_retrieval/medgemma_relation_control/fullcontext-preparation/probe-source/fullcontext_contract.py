"""Input-only gate for one fresh-base full-context capacity probe, never training."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import construct

ORIGINAL_PROBE_SHA = '0a9828c41e6748d1dabddca620bc7745ce807fbc8507048a55f325e161b07d60'
TRAINER_SHA = '3daa85b011d80a371589fbd2b4cbe1f6dda28885ddd8d4d27089d283d16d965f'
CONSTRUCT_SHA = 'b201fb7f4b1208a6bdad6a4178aea04045cdb9d3f59e64fb0d666fdc891b5c44'
FULL_TRAIN_SHA = '9e80e54125d3d46479acb9a3720e8b5f4c13ffacb5f65e82d3d9d10ac839d25e'
TRAIN_INPUT_SHA = 'a2c2baecb86ca425ae978688014073b69a415dbc9d3526d24be96f82b89e3e6d'
PREPARATION_SHA = '811ef287e28735c859ec0547db3a820535ac30041ce1fde11ae332604539ac9b'
CPU_SHA = '5d295ec580cd0c862a0fa1ded99ccfe93bb68f92631aec8dbbc5eb94868d34d9'
ROW_AUDIT_SHA = '80f5bbce3f2f4d1537bfef9e9dbee5bd61c77d17bb7430c3810fc6bfbdedd1e0'
ORIGINAL_TRAINING_CAP = 8192
PROBE_SEQUENCE_CAP = 14000
LONGEST_ROW_ID = '12628480:relations:8'
LONGEST_TOKENS = 12824


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError('Frozen input/source SHA differs: ' + str(path))
    return json.loads(Path(path).read_text())


def verify_sources_and_data(args, training, interface):
    if sha(inspect.getfile(training)) != TRAINER_SHA or sha(construct.__file__) != CONSTRUCT_SHA:
        raise ValueError('Require unchanged original trainer and append constructor')
    if training.CONFIG['max_seq_length'] != ORIGINAL_TRAINING_CAP:
        raise ValueError('Original trainer cap must remain8192')
    original_args = SimpleNamespace(**vars(args))
    original_args.train = args.original_train
    originals, train_ids, excluded = training.verify_sources_and_data(original_args)
    if sha(args.train) != FULL_TRAIN_SHA or sha(args.train_input) != TRAIN_INPUT_SHA:
        raise ValueError('Require exact fullcontext124 and pure strict32 source')
    preparation = checked_json(args.fullcontext_manifest, PREPARATION_SHA)
    summary = checked_json(args.cpu_reference, CPU_SHA)
    row_audit = checked_json(args.cpu_row_audit, ROW_AUDIT_SHA)
    if (preparation['artifacts']['train.fullcontext.jsonl']['sha256'] != FULL_TRAIN_SHA
            or not preparation['training_answers_unchanged']
            or not preparation['all_nonmessage_fields_unchanged']):
        raise ValueError('Append-only preparation contract differs')
    if (summary['row_token_audit_sha256'] != ROW_AUDIT_SHA
            or summary['interface_sha256'] != training.INTERFACE_SHA
            or summary['model_asset_manifest_sha256'] != training.SOURCE_MANIFEST_SHA
            or summary['construction_source_sha256'] != CONSTRUCT_SHA
            or summary['cuda_initialized'] or summary['model_weights_loaded']
            or summary['F1_gold_opened'] or summary['training_or_prediction_or_score_performed']
            or not summary['no_truncation'] or not summary['supervision_boundary_passed_all124']):
        raise ValueError('Require the completed native CPU-only audit')
    documents = training.helpers.rows(args.train_input)
    rows = training.helpers.rows(args.train)
    reconstructed, _, _ = construct.append_reports(originals, documents, True)
    if rows != reconstructed or len(rows) != 124:
        raise ValueError('Full context does not reversibly preserve original124')
    if [r['row_id'] for r in row_audit['train']] != [r['row_id'] for r in rows]:
        raise ValueError('Native CPU row ordering differs')
    if (summary['train_full_tokens']['total'] != 965436
            or summary['train_full_tokens']['max'] != LONGEST_TOKENS
            or summary['train_full_tokens']['longest_row_id'] != LONGEST_ROW_ID
            or summary['train_supervised_tokens']['total'] != 7150):
        raise ValueError('Native CPU full-context totals differ')
    return rows, train_ids, excluded, summary, row_audit['train']


def verify_one_encoding(encoded, expected):
    ids, labels = encoded['input_ids'], encoded['labels']
    prefix = encoded['prefix_tokens']
    if (encoded['row_id'] != expected['row_id']
            or len(ids) != expected['full_tokens'] or prefix != expected['prefix_tokens']
            or encoded['supervised_tokens'] != expected['supervised_tokens']
            or len(ids) - prefix != encoded['supervised_tokens']
            or labels != [-100] * prefix + ids[prefix:]
            or construct.jsha(ids) != expected['token_ids_sha256']
            or construct.jsha(ids[prefix:]) != expected['supervised_ids_sha256']
            or encoded['rendered_prefix_sha256'] != expected['rendered_prefix_sha256']
            or encoded['rendered_full_sha256'] != expected['rendered_full_sha256']
            or ids[prefix:].count(106) != 1 or 1 in ids[prefix:]
            or encoded['eos_supervised_position'] != prefix + ids[prefix:].index(106)
            or encoded['eos_token_id'] != 106
            or not expected['prefix_mask_eos_verified'] or not expected['original_answer_unchanged']):
        raise ValueError('Actual native encoding differs from frozen CPU row: ' + encoded['row_id'])


def verify_encoded(encoded, audited, summary):
    if len(encoded) != len(audited) or len(encoded) != 124:
        raise ValueError('Require exactly all124 native encodings')
    for actual, expected in zip(encoded, audited):
        verify_one_encoding(actual, expected)
    selected = max(range(len(encoded)), key=lambda i: len(encoded[i]['input_ids']))
    if (encoded[selected]['row_id'] != LONGEST_ROW_ID
            or len(encoded[selected]['input_ids']) != LONGEST_TOKENS
            or sum(len(e['input_ids']) for e in encoded) != summary['train_full_tokens']['total']
            or sum(e['supervised_tokens'] for e in encoded) != 7150
            or any(len(e['input_ids']) > PROBE_SEQUENCE_CAP for e in encoded)):
        raise ValueError('Longest native row/totals/new14000 rejection cap differs')
    return selected
