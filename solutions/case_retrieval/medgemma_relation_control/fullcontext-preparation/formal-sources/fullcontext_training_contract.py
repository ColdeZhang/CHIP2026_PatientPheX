"""Bind the existing native full-context CPU audit, original labels and append-only input."""
from pathlib import Path
import hashlib
import inspect
import json
import fullcontext_contract as proven
import construct

PROVEN_CONTRACT_SHA = '9b4d7505224240b8408ada3d7c8e92a40981985a71215bbe5d848f85f92b54b5'
INTERFACE_SHA = 'aecef7eaab0565ab98ce76c034f01b543be3342b00a84d4bae71c36d614d55be'
MODEL_SOURCE_SHA = '690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1'
ORIGINAL_INFER_SHA = 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
FULL_INFER_SHA = '2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085'
INFER_INPUT_SHA = '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
ORIGINAL_TRAIN_SHA = 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
INFERENCE_CAP = 14000
NEW_TOKENS = 1600


def source_contract(interface):
    if (proven.sha(proven.__file__) != PROVEN_CONTRACT_SHA
            or proven.sha(construct.__file__) != proven.CONSTRUCT_SHA
            or proven.sha(interface.__file__) != INTERFACE_SHA):
        raise ValueError('Frozen append/native input contracts differ')


def references(args):
    summary = proven.checked_json(args.cpu_reference, proven.CPU_SHA)
    audited = proven.checked_json(args.cpu_row_audit, proven.ROW_AUDIT_SHA)
    preparation = proven.checked_json(args.fullcontext_manifest, proven.PREPARATION_SHA)
    if (summary['row_token_audit_sha256'] != proven.ROW_AUDIT_SHA
            or summary['interface_sha256'] != INTERFACE_SHA
            or summary['model_asset_manifest_sha256'] != MODEL_SOURCE_SHA
            or summary['cuda_initialized'] or summary['model_weights_loaded']
            or summary['F1_gold_opened'] or summary['training_or_prediction_or_score_performed']
            or not summary['no_truncation'] or not summary['supervision_boundary_passed_all124']
            or not summary['all_answers_CID_PID_candidates_original_messages_preserved']):
        raise ValueError('Require the completed native CPU-only append audit')
    return summary, audited, preparation


def verify_inference_sources(args, interface, prompts, original_prompts, model_source_manifest):
    source_contract(interface)
    summary, audit, preparation = references(args)
    for path, expected in [(prompts,FULL_INFER_SHA),(original_prompts,ORIGINAL_INFER_SHA),
                           (args.inference_input,INFER_INPUT_SHA),(model_source_manifest,MODEL_SOURCE_SHA)]:
        if proven.sha(path) != expected:
            raise ValueError('Frozen full80 input/provenance differs: '+str(path))
    if (preparation['artifacts']['infer.fullcontext.jsonl']['sha256'] != FULL_INFER_SHA
            or preparation['sources']['original_infer']['sha256'] != ORIGINAL_INFER_SHA
            or preparation['sources']['infer_input']['sha256'] != INFER_INPUT_SHA):
        raise ValueError('Full80 provenance must preserve the original80 task')
    originals = construct.rows(Path(original_prompts).read_text())
    documents = construct.rows(Path(args.inference_input).read_text())
    rows = construct.rows(Path(prompts).read_text())
    generated, _, _ = construct.append_reports(originals,documents,False)
    if rows != generated or len(rows) != 80 or len(documents) != 16:
        raise ValueError('Full80 does not reversibly preserve original80')
    if (any({'answer','entities','association'} & row.keys() for row in rows)
            or [r['row_id'] for r in rows] != [r['row_id'] for r in audit['infer']]
            or summary['inference_input_tokens']['total'] != 645000
            or summary['inference_input_tokens']['max'] != 13805
            or summary['inference_input_tokens']['over_caps']['14000'] != 0):
        raise ValueError('Label-free all80/native CPU token coverage differs')
    return rows, {r['row_id']:r for r in audit['infer']}


def verify_training_sources(args, original_training, interface):
    source_contract(interface)
    rows, training_ids, excluded_ids, _, _ = proven.verify_sources_and_data(args,original_training,interface)
    infer_rows, _ = verify_inference_sources(args,interface,args.inference_prompts,args.original_inference_prompts,args.model_source_manifest)
    if any(str(r['pmc_id']) not in excluded_ids for r in infer_rows):
        raise ValueError('Full80 must remain in original excluded input documents')
    return rows, training_ids, excluded_ids


def verify_actual_training_encoding(args, encoded, output, helpers):
    summary, audit, _ = references(args)
    proven.verify_encoded(encoded,audit['train'],summary)
    helpers.write(Path(output)/'fullcontext-cpu-reference-reuse.json', {
        'status':'actual_native_encoding_matches_preexisting_CPU_audit',
        'cpu_summary_sha256':proven.CPU_SHA,'cpu_row_audit_sha256':proven.ROW_AUDIT_SHA,
        'original_native_audit_source_sha256':summary['source_sha256'],
        'new_trainer_CPU_meta_run_claimed':False,'original_labels_training_sha256':ORIGINAL_TRAIN_SHA,
        'append_training_sha256':proven.FULL_TRAIN_SHA,'actual_encoded_sha256':helpers.json_sha(encoded),
        'training_rows':124,'full_tokens':965436,'supervised_tokens':7150,
        'original_training_cap':8192,'new_fullcontext_training_cap':14000,'no_truncation':True})


def verify_inference_encoding(row, ids, expected, rendered=None):
    if (row['row_id'] != expected['row_id'] or str(row['pmc_id']) != expected['pmc_id']
            or len(ids) != expected['input_tokens'] or len(ids) > INFERENCE_CAP
            or construct.jsha(ids) != expected['token_ids_sha256'] or not expected['no_answer']
            or 'answer' in row):
        raise ValueError('Actual full80 native input differs from frozen CPU row: '+row['row_id'])
    if rendered is not None and hashlib.sha256(rendered.encode()).hexdigest() != expected['rendered_prefix_sha256']:
        raise ValueError('Rendered full80 prompt changed: '+row['row_id'])


def verify_actual_inference_encoding(args, encoded):
    summary, audit, _ = references(args)
    if len(encoded) != 80 or [r['row_id'] for r,_ in encoded] != [r['row_id'] for r in audit['infer']]:
        raise ValueError('Require complete80 input-only native encodings')
    for (row,ids), expected in zip(encoded,audit['infer']):
        verify_inference_encoding(row,ids,expected)
    if sum(len(ids) for _,ids in encoded) != summary['inference_input_tokens']['total']:
        raise ValueError('Full80 native token total differs')


def provenance():
    return {'schema':'append_only_same_closed_CID_PID_task_v1',
            'original_label_training_sha256':ORIGINAL_TRAIN_SHA,
            'append_training_sha256':proven.FULL_TRAIN_SHA,
            'original_inference_prompts_sha256':ORIGINAL_INFER_SHA,
            'append_inference_prompts_sha256':FULL_INFER_SHA,
            'fullcontext_preparation_sha256':proven.PREPARATION_SHA,
            'native_CPU_summary_sha256':proven.CPU_SHA,'native_CPU_rows_sha256':proven.ROW_AUDIT_SHA,
            'original_trainer_source_sha256':proven.TRAINER_SHA,
            'append_constructor_sha256':proven.CONSTRUCT_SHA,'legacy_constructor_sha256':construct.LEGACY_SHA,
            'proven_probe_contract_sha256':PROVEN_CONTRACT_SHA,
            'formal_input_contract_sha256':proven.sha(__file__),
            'new_trainer_CPU_meta_run_claimed':False}


def source_snapshot():
    return [('fullcontext_training_contract.py',__file__),('fullcontext_contract.py',inspect.getfile(proven)),
            ('construct.py',inspect.getfile(construct)),('legacy_full_context.py',inspect.getfile(construct.legacy))]
