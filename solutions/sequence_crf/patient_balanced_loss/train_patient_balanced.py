#!/usr/bin/env python3
"""SOURCE DRAFT: fixed alpha=.5 patient-balanced patient-first124 LoRA.

Derived mechanically from train_patient_first.py a3ec17f5; no formal execution
plan, budget, deployment or training result accompanies this source draft.
The only objective change is mixed_loss plus one RowLossScope per original row.
The complete native BF16 head and outer loss / actual group size are retained.
Original F1 used Torch2.7.1; this draft pins the tested Torch2.6.0+cu124 runtime.

CPU mode verifies data/template/loss/meta without weights or CUDA. Training mode
requires an allocated empty GPU and exact official files. No validation labels,
inference, quantization, packing, smoke adapter or alternative task is supported.
Fresh base is required; only this exact mixed-objective run may resume complete epochs.
Native CPU receipt binds the transposition; no parser retest or validation scoring.
On OOM/failure stop and preserve complete epoch checkpoints; never auto-retry.
Resume state is saved/restored; numerical GPU resume equivalence is not claimed.
Length6144 is only a rejection cap: batches pad to their actual maximum, never to cap.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import qwen35_smoke as helpers
import qwen35_relation_cost as protocol

HELPER_SHA='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
PROTOCOL_SHA='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
OLD_TRAINER_SHA='2a25b83d5d9e3dcc3f57cc8e2ff1aaf519a41a709e4ff289b2b26db9fab51423'
SOURCE_MANIFEST_SHA='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
CONFIG={'max_seq_length':6144,'batch_size':1,'gradient_accumulation':8,'epochs':2,
        'learning_rate':1e-4,'weight_decay':.01,'warmup_ratio':.05,'max_grad_norm':1.,
        'rank':16,'alpha':32,'dropout':.05,'seed':20261023,'targets':protocol.TARGETS,
        'dtype':'bfloat16','attention':'sdpa','quantization':False,'packing':False,
        'gradient_checkpointing':{'use_reentrant':False}}
OLD_TRAIN_SHA='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
TRAIN_SHA='14af86cffe1a8b87af022d13e42f9b910281f0da478e3cc731bc2fbc9ba68c3b'
PARENT_TRAINER_SHA='08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
TRANSPOSITION_SHA='68d28bafb7b463b68857f44957d7cf33d05116ff2e49872183f7e363458cb254'
TRAIN_TOKEN_AUDIT_SHA='36f6faaee683fcaff319714a3b08535208848b45185ab72b7958829417d03586'
NATIVE_CPU_RECEIPT_SHA='2c5d825470e8a089b79040c8625d0b40f53bf6e1759f45b5fca87a369634b2d5'
NATIVE_CPU_CONTRACT_SHA='5ba668a89f2fd3f894d08d99158a72e706f953ae0816dd4871b2cb9b6ae7c61f'
SCHEMA_HELPER_SHA='25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08'
EXCLUDED_SHA='2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
PREPARATION_SHA='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
COPIED_TRAINER_SHA='9d2e0eeb34e6bfa76ea90cbf6f9ffad996ab527e4746bb936e4f3cbc58724b8b'
COPIED_PATIENT_FIRST_TRAINER_SHA='a3ec17f55ca7007ab1f41879ff07e15586f78d7b3064a584464cd90fa935164e'
WEIGHTS_PREPARATION_SHA='c713111b51c70c2c39a34917f06a5dab4fb162c0fb724e0689c2d096abdeccda'
ROW_WEIGHTS_SHA='e1e4cc5ec55a01b198607f271c6e4d4d7c15824a50cfeb884c6dd876218d4429'
EPOCH_GROUPS_SHA='6b80ef9d11f41811ec5d0e54a90189c359e267c18fec452cab242cf107640452'
MIXED_LOSS_SHA='dc98bb97605419de568a5ba736607aa50d1907957f98daa618e12c87446fccd5'
ROW_LOSS_SCOPE_SHA='7b9900fc4896f53b02f4edcbbd94799258c3bb8f91fec7c3b3768054d4690da8'
NATIVE_LOSS_SHA='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'
OBJECTIVE={'name':'patient-balanced-chunk-equal-mixed-alpha05-v1','alpha':.5,
    'patients':87,'groups_per_epoch':16,'original_outer_backward_divides_by_actual_group_size':True,
    'value_label_positions':'full labels q; causal logits q-1',
    'old_component':'all supervised structure/key/value/EOS/newline mean CE',
    'new_component':'g*S/P * sum_p(meanCE(value_p)/n_p)',
    'original_F1_torch':'2.7.1','draft_target_torch':'2.6.0+cu124',
    'complete_native_BF16_head':True,'selected_FP32_CE_equivalent_old_half':True,
    'hook_diagnostic_kwargs':'only output_attentions=None and output_hidden_states=None',
    'actual_loss_call_observed_and_saved_before_parameter_guards':True,
    'old_objective_checkpoint_continuation_forbidden':True}
sha256=helpers.sha
write_json=helpers.write


def semantic_configuration(args):
    return {'fixed':CONFIG,'objective':OBJECTIVE,'paths_and_device':{k:v for k,v in vars(args).items()
            if k not in {'resume','stop_after_epoch'}}}


def load_objective_sources():
    """Load only the two exact sibling sources; no alternate loss implementation."""
    modules={}
    for name,relative,digest in [('mixed_loss','mixed_loss.py',MIXED_LOSS_SHA),
                               ('row_loss_scope_v2','interface_recovery/row_loss_scope_v2.py',ROW_LOSS_SCOPE_SHA)]:
        path=Path(__file__).resolve().parent/relative
        if sha256(path)!=digest:raise ValueError('Frozen objective source differs: '+name)
        spec=importlib.util.spec_from_file_location('_patient_balanced_'+name,path)
        if spec is None or spec.loader is None:raise ValueError('Cannot load fixed objective source')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules[name]=module
    return modules['mixed_loss'],modules['row_loss_scope_v2'].RowLossScope


def append_loss_call_observation(path,actual):
    """Called inside the v2 hook before compatibility assertions or loss math."""
    path=Path(path)
    with path.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(actual,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    parent=path.resolve().parent
    while True:
        descriptor=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if parent.parent==parent:break
        parent=parent.parent


def bind_row_weights(args,rows,encoded,output):
    """Bind every original row to immutable preparation bytes and native encoding.

    No tokenization, value slicing, patient counting or coefficients are rebuilt.
    All native IDs/labels/rendered hashes were independently checked by encode_all.
    """
    for path,digest,size in [(args.weights_preparation_manifest,WEIGHTS_PREPARATION_SHA,5074),
                            (args.row_weights,ROW_WEIGHTS_SHA,308095),
                            (args.epoch_groups,EPOCH_GROUPS_SHA,594436)]:
        if Path(path).stat().st_size!=size or sha256(path)!=digest:
            raise ValueError('Frozen patient-balanced preparation file differs: '+str(path))
    preparation=json.loads(Path(args.weights_preparation_manifest).read_text())
    if (preparation['status']!='PASS_FIXED_ALPHA05_TRAIN124_POSITION_AND_COEFFICIENT_PREPARATION'
        or (preparation['rows'],preparation['documents'],preparation['patients'],preparation['groups_per_epoch'])!=(124,32,87,16)
        or preparation['outputs']['row-weights.jsonl']['sha256']!=ROW_WEIGHTS_SHA
        or preparation['outputs']['epoch-groups.json']['sha256']!=EPOCH_GROUPS_SHA
        or preparation['references']['trainer']['sha256']!=COPIED_PATIENT_FIRST_TRAINER_SHA
        or preparation['references']['train']['sha256']!=TRAIN_SHA
        or preparation['references']['token_audit']['sha256']!=TRAIN_TOKEN_AUDIT_SHA):
        raise ValueError('Patient-balanced preparation ancestry differs')
    weights=[json.loads(line) for line in Path(args.row_weights).read_text().splitlines() if line.strip()]
    audit=json.loads(Path(args.train_token_audit).read_text())
    if len(weights)!=len(rows) or len(weights)!=len(encoded) or len(weights)!=124:
        raise ValueError('Patient-balanced row count differs')
    records=[]
    for index,(row,item,meta,expected) in enumerate(zip(rows,encoded,weights,audit)):
        actual={'row_id':row['row_id'],'pmc_id':str(row['pmc_id']),
            'full_tokens':len(item['input_ids']),'prefix_tokens':item['prefix_tokens'],
            'supervised_tokens':item['supervised_tokens'],'eos_token_id':item['eos_token_id'],
            'rendered_prefix_sha256':item['rendered_prefix_sha256'],
            'rendered_full_sha256':item['rendered_full_sha256'],
            'input_ids_sha256':hashlib.sha256(json.dumps(item['input_ids']).encode()).hexdigest(),
            'labels_sha256':hashlib.sha256(json.dumps(item['labels']).encode()).hexdigest()}
        q=[i for i,label in enumerate(item['labels']) if label!=-100]
        if (meta['row_index']!=index or meta['row_id']!=row['row_id'] or meta['pmc_id']!=str(row['pmc_id'])
            or meta['native_audit']!=actual or actual!=expected['patient_first']
            or meta['sequence_length']!=len(item['labels']) or meta['prefix_length']!=item['prefix_tokens']
            or meta['supervised_label_positions']!=q or meta['supervised_logit_positions']!=[i-1 for i in q]):
            raise ValueError('Original row/native four-SHA/position binding differs: '+row['row_id'])
        records.append({'row_index':index,'row_id':row['row_id'],'native_audit':actual,
                        'row_metadata_sha256':hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()})
    epoch_groups=json.loads(Path(args.epoch_groups).read_text())
    if (epoch_groups['patients']!=87 or epoch_groups['groups_per_epoch']!=16
        or epoch_groups['alpha']!={'numerator':1,'denominator':2}
        or epoch_groups['original_outer_backward_divides_by_group_size'] is not True
        or len(epoch_groups['epochs'])!=2):
        raise ValueError('Frozen patient-balanced epoch contract differs')
    for epoch,contract in enumerate(epoch_groups['epochs']):
        groups=batch_plan(124,1,8,CONFIG['seed'],epoch)
        if contract['epoch']!=epoch+1 or contract['seed']!=CONFIG['seed']+epoch or len(contract['groups'])!=16:
            raise ValueError('Frozen original epoch/seed differs')
        for group_index,(group,expected) in enumerate(zip(groups,contract['groups'])):
            if (expected['group_index']!=group_index or expected['row_indices']!=[indices[0] for indices in group]
                or expected['group_size']!=len(group) or expected['outer_group_divisor']!=len(group)
                or len(group)!=(4 if group_index==15 else 8)):
                raise ValueError('Original order or actual tail-group divisor differs')
    write_json(output/'row-metadata-binding.json',{'objective':OBJECTIVE,
        'preparation_sha256':WEIGHTS_PREPARATION_SHA,'row_weights_sha256':ROW_WEIGHTS_SHA,
        'epoch_groups_sha256':EPOCH_GROUPS_SHA,'rows':records,'all_original_native_four_SHA_exact':True,
        'no_retokenization_or_rederived_value_positions_or_patient_weights':True})
    return weights,epoch_groups


def require_scope_cleared(base,original):
    if hasattr(base,'_patient_balanced_active_scope') or base.loss_function is not original:
        raise RuntimeError('Stale, foreign or overlapping row loss scope; refuse to continue')


def load_schema_helper(path):
    if sha256(path)!=SCHEMA_HELPER_SHA:raise ValueError('Fixed patient-first schema source differs')
    spec=importlib.util.spec_from_file_location('_patient_first_training_schema',path)
    if spec is None or spec.loader is None:raise ValueError('Cannot load patient-first schema helper')
    schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
    return schema


def training_rows(train, excluded, schema):
    # Only train labels are opened. The old excluded file remains stripped input.
    data=[schema.parse_json(line) for line in Path(train).read_text().splitlines() if line.strip()]
    heldout=helpers.rows(excluded)
    if not data or not heldout:
        raise ValueError('Training and excluded input must be nonempty')
    heldout_ids = [str(x['pmc_id']) for x in heldout]
    if len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError('Duplicate excluded document')
    if any('entities' in x or 'association' in x for x in heldout):
        raise ValueError('Excluded input must not contain label fields')
    ids = [x.get('row_id') for x in data]
    if any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Training row IDs must be unique strings')
    document_ids = sorted({str(x['pmc_id']) for x in data})
    if len(document_ids) != 32 or set(document_ids) & set(heldout_ids):
        raise ValueError('Require the actual 32-document training set, disjoint from excluded input')
    for row in data:
        schema.validate_input(row)
        if set(row)!=schema.INPUT_KEYS|{'answer'}:raise ValueError('Require complete patient-first training row')
        system_lines=row['messages'][0]['content'].split('\n')
        if system_lines[:2]!=[schema.NEW_TASK,schema.NEW_SCHEMA] or len(system_lines)<3:
            raise ValueError('Patient-first system schema differs: '+row['row_id'])
        answer=schema.parse_json(row['answer'])
        schema.validate_matrix(answer,row['patient_ids'],list(row['code_map']))
        canonical={pid:[cid for cid in row['code_map'] if cid in answer[pid]] for pid in row['patient_ids']}
        if row['answer']!=schema.compact(canonical):raise ValueError('Patient/CID answer ordering differs')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
            raise ValueError('Messages must end with user: '+row['row_id'])
        if any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant'}
               or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Require text-only messages: '+row['row_id'])
        if not isinstance(row.get('answer'), str) or not isinstance(json.loads(row['answer']), (list,dict)):
            raise ValueError('Answer must be a JSON array/object: '+row['row_id'])
    return data, document_ids, sorted(heldout_ids)


def verify_sources_and_data(args):
    if sha256(inspect.getfile(helpers))!=HELPER_SHA or sha256(inspect.getfile(protocol))!=PROTOCOL_SHA:
        raise ValueError('Frozen Qwen3.5 helper/protocol source differs')
    fixed_inputs=[(args.train,TRAIN_SHA,1720127),(args.excluded_input,EXCLUDED_SHA,None),
        (args.data_manifest,PREPARATION_SHA,9323),(args.transposition_manifest,TRANSPOSITION_SHA,10469),
        (args.train_token_audit,TRAIN_TOKEN_AUDIT_SHA,163235),
        (args.native_cpu_receipt,NATIVE_CPU_RECEIPT_SHA,8020),(args.schema_helper,SCHEMA_HELPER_SHA,7505)]
    for path,digest,size in fixed_inputs:
        if sha256(path)!=digest or (size is not None and Path(path).stat().st_size!=size):
            raise ValueError('Frozen patient-first/provenance input differs: '+str(path))
    if sha256(args.model_source_manifest)!=SOURCE_MANIFEST_SHA:
        raise ValueError('Official15-file fixed source manifest differs')
    schema=load_schema_helper(args.schema_helper)
    rows,train_ids,excluded_ids=training_rows(args.train,args.excluded_input,schema)
    if len(rows)!=124 or len(train_ids)!=32 or len(excluded_ids)!=32:
        raise ValueError('Require124 rows/32train documents/32excluded input documents')
    preparation=json.loads(Path(args.data_manifest).read_text())
    # The OLD preparation binds OLD CID-to-PID training bytes. Never compare it
    # to the new transposed TRAIN_SHA; the new summary/receipt supply that bridge.
    if (preparation['training_sha256']!=OLD_TRAIN_SHA or preparation['excluded_input_sha256']!=EXCLUDED_SHA
        or preparation['training_document_ids']!=train_ids or preparation['excluded_document_ids']!=excluded_ids
        or not preparation['all_target_documents_excluded_from_own_generators']
        or not preparation['no_F1_or_outer0_in_any_generator_or_SFT_training']
        or not preparation['fresh_base_required'] or not preparation['old_adapter_continuation_forbidden']):
        raise ValueError('Old meta32 source provenance identity/isolation differs')
    transposition=json.loads(Path(args.transposition_manifest).read_text())
    if transposition['status']!='PASS_CPU_patient_first_closed_set_preparation_no_training_or_quality':
        raise ValueError('Transposition preparation is not complete')
    if (transposition['sources']['train']['sha256']!=OLD_TRAIN_SHA
        or transposition['sources']['preparation_manifest']['sha256']!=PREPARATION_SHA
        or transposition['sources']['encode_source']['sha256']!=HELPER_SHA
        or transposition['transform_source']['sha256']!=SCHEMA_HELPER_SHA
        or transposition['training']['document_ids']!=train_ids):
        raise ValueError('New transposition does not bind old train32 provenance')
    artifacts={Path(x['path']).name:x for x in transposition['artifacts']}
    if (artifacts['train124.patient-first.jsonl']['sha256']!=TRAIN_SHA
        or artifacts['train-token-audit.json']['sha256']!=TRAIN_TOKEN_AUDIT_SHA):
        raise ValueError('Transposed output/audit binding differs')
    f1_ids=transposition['F1_input_only']['document_ids']
    if (len(f1_ids)!=16 or len(set(f1_ids))!=16 or not set(f1_ids)<=set(excluded_ids)
        or set(f1_ids)&set(train_ids) or transposition['F1_input_only']['targets_read'] is not False):
        raise ValueError('Transposition train/F1 isolation differs')
    cpu=json.loads(Path(args.native_cpu_receipt).read_text())
    expected_sources={'train':TRAIN_SHA,'train_token_audit':TRAIN_TOKEN_AUDIT_SHA,'summary':TRANSPOSITION_SHA,
        'preparation_manifest':PREPARATION_SHA,'encode_helper':HELPER_SHA,'transform':SCHEMA_HELPER_SHA}
    if any(cpu['sources'][name]['sha256']!=digest for name,digest in expected_sources.items()):
        raise ValueError('Actual native CPU receipt source chain differs')
    if (cpu['status']!='PASS_native_TF5202_CPU_patient_first_train124_token_contract'
        or cpu['data_contract_source']['sha256']!=NATIVE_CPU_CONTRACT_SHA
        or cpu['training_document_ids']!=train_ids or cpu['F1_document_ids_from_summary_only']!=f1_ids
        or (cpu['rows'],cpu['total_full_tokens'],cpu['max_full_tokens'],cpu['supervised_tokens'])!=(124,400614,5556,4694)
        or not cpu['all_rows_patient_first_token_audit_exact'] or cpu['training_F1_overlap']!=0
        or cpu['truncation'] or cpu['F1_files_read']!=0 or cpu['runtime']['cuda_initialized']):
        raise ValueError('Actual native CPU receipt contract differs')
    return rows,train_ids,excluded_ids


def model_files(model_dir,source_manifest,require_weights):
    source=json.loads(Path(source_manifest).read_text())
    if source.get('model')!='Qwen/Qwen3.5-9B' or source.get('hf_revision')!=helpers.REVISION or len(source['files'])!=15:
        raise ValueError('Unexpected official model source/revision/file count')
    expected={x['name']:x for x in source['files']}
    if len(expected)!=15 or any(Path(n).name!=n for n in expected):
        raise ValueError('Duplicate or nonlocal official filename')
    names=sorted(expected if require_weights else helpers.CPU_FILES)
    actual={}
    for name in names:
        path=Path(model_dir)/name
        actual[name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        if actual[name]!={'bytes':expected[name]['size'],'sha256':expected[name]['sha256']}:
            raise ValueError('Actual official file differs: '+name)
    return {'model':source['model'],'hf_revision':source['hf_revision'],'source_manifest_sha256':sha256(source_manifest),
            'actual_files':actual,'all15_files_verified':require_weights}


def encode_all(rows,tokenizer,output,train_token_audit,native_cpu_receipt):
    audit=json.loads(Path(train_token_audit).read_text())
    cpu=json.loads(Path(native_cpu_receipt).read_text())
    if len(audit)!=124 or len({x['row_id'] for x in audit})!=124:
        raise ValueError('Frozen patient-first row audit differs')
    if [row['row_id'] for row in rows]!=[item['row_id'] for item in audit]:
        raise ValueError('Patient-first frozen row order differs')
    encoded=[helpers.encode(row,tokenizer) for row in rows]
    records=[]
    for row,item,expect in zip(rows,encoded,audit):
        # Per-list hashes deliberately use the historical default json.dumps
        # spacing; helpers.json_sha is the separate compact full-object hash.
        actual={'row_id':row['row_id'],'pmc_id':str(row['pmc_id']),
            'full_tokens':len(item['input_ids']),'prefix_tokens':item['prefix_tokens'],
            'supervised_tokens':item['supervised_tokens'],'eos_token_id':item['eos_token_id'],
            'rendered_prefix_sha256':item['rendered_prefix_sha256'],
            'rendered_full_sha256':item['rendered_full_sha256'],
            'input_ids_sha256':hashlib.sha256(json.dumps(item['input_ids']).encode()).hexdigest(),
            'labels_sha256':hashlib.sha256(json.dumps(item['labels']).encode()).hexdigest()}
        if actual!=expect['patient_first']:raise ValueError('Native patient-first encoding differs: '+row['row_id'])
        records.append(actual)
    lengths=[len(e['input_ids']) for e in encoded]
    if len(encoded)!=124 or max(lengths)!=5556 or sum(lengths)!=400614 or any(n>CONFIG['max_seq_length'] for n in lengths):
        raise ValueError('Frozen patient-first124 token accounting differs; no truncation permitted')
    if sum(e['supervised_tokens'] for e in encoded)!=4694:
        raise ValueError('Frozen patient-first supervised-token accounting differs')
    if (helpers.json_sha(encoded)!=cpu['encoded_helper_json_sha256']
        or helpers.json_sha(records)!=cpu['records_helper_json_sha256']):
        raise ValueError('All124 encoded rows differ from passed actual native CPU receipt')
    longest=[item for item in encoded if len(item['input_ids'])==5556]
    if (len(longest)!=1 or longest[0]['row_id']!='11235872:relations:8'
        or longest[0]['supervised_tokens']!=135 or longest[0]['prefix_tokens']!=5421
        or helpers.json_sha(longest[0])!=cpu['selected_encoded_helper_json_sha256']):
        raise ValueError('Passed5556 longest row binding differs')
    write_json(output/'token-mask-audit.json',{'rows':records,
        'rows_count':124,'total_full_tokens':sum(lengths),'max_full_tokens':max(lengths),
        'supervised_tokens':4694,'truncated_rows':0,'prefix_mask_eos_passed':True,
        'source_token_audit_sha256':TRAIN_TOKEN_AUDIT_SHA,'native_CPU_receipt_sha256':NATIVE_CPU_RECEIPT_SHA,
        'all_rows_native_CPU_exact':True,'encoded_sha256':helpers.json_sha(encoded)})
    return encoded


def validate_checkpoint(output,checkpoint,fingerprint):
    if checkpoint.resolve().parent!=output.resolve():raise ValueError('Resume checkpoint must belong to original output')
    if any(output.glob('.epoch-*.incomplete')):raise ValueError('Inspect incomplete checkpoint staging before resume')
    if not checkpoint.is_dir() or re.fullmatch(r'epoch-[0-9]{3}',checkpoint.name) is None:
        raise ValueError('Resume checkpoint must be an exact epoch-NNN directory')
    metadata=json.loads((checkpoint/'checkpoint.json').read_text())
    verify_fingerprint(metadata['fingerprint'],fingerprint)
    completed=int(metadata['completed_epochs']);expected=output/f'epoch-{completed:03d}'
    # H44: acknowledgements such as epoch-001.ack.json are ordinary files,
    # never checkpoint candidates and never opened by this selection.
    epoch_directories=[p for p in output.iterdir() if p.is_dir() and re.fullmatch(r'epoch-[0-9]{3}',p.name)]
    if checkpoint.resolve()!=expected.resolve() or checkpoint.name!=max(p.name for p in epoch_directories):
        raise ValueError('Resume requires latest complete epoch checkpoint')
    if not 1<=completed<2 or metadata['global_step']!=16*completed or metadata.get('smoke_only'):
        raise ValueError('Checkpoint is not an unfinished formal124x2 patient-first run')
    for name,digest in metadata['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe checkpoint member')
        if sha256(checkpoint/relative)!=digest:raise ValueError('Checkpoint SHA mismatch: '+name)
    if not {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}.issubset(metadata['files']):
        raise ValueError('Incomplete training state/adapter checkpoint')
    return metadata

def batch_plan(size, batch_size, accumulation, seed, epoch, smoke=False, lengths=None):
    if min(size, batch_size, accumulation) < 1:
        raise ValueError("Dataset and batch sizes must be positive")
    if smoke:
        if lengths is None or len(lengths) != size:
            raise ValueError("Smoke requires all sample lengths")
        return [[[max(range(size), key=lambda i: lengths[i])]]]
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, size, batch_size)]
    return [batches[i:i + accumulation] for i in range(0, len(batches), accumulation)]

def verify_fingerprint(expected, actual):
    if expected != actual:
        raise ValueError("Exact resume fingerprint mismatch; use a new output directory")

def validate_stop_epoch(stop_after_epoch, epochs, completed_epochs=0, smoke=False):
    if stop_after_epoch is None:
        return
    if smoke:
        raise ValueError("--stop-after-epoch is not supported with --smoke")
    if not 1 <= stop_after_epoch <= epochs:
        raise ValueError("--stop-after-epoch must be between 1 and --epochs")
    if stop_after_epoch <= completed_epochs:
        raise ValueError("--stop-after-epoch must be after the resumed completed epoch")

def epoch_outcome(completed_epochs, epochs, stop_after_epoch):
    if not 1 <= completed_epochs <= epochs:
        raise ValueError("Completed epoch outside configured training schedule")
    if stop_after_epoch is not None and completed_epochs > stop_after_epoch:
        raise ValueError("Training advanced beyond the requested stop epoch")
    if completed_epochs == epochs:
        return "complete"
    return "paused" if completed_epochs == stop_after_epoch else "running"

def collate(examples, pad_token_id, torch):
    width = max(len(x["input_ids"]) for x in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device="cuda:0"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda:0"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda:0")}

def durability_fsync_directory(path):
    descriptor=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def durability_inventory(root):
    members={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Checkpoint symlinks are forbidden')
        if path.is_file():members[path.relative_to(root).as_posix()]=path
        elif not path.is_dir():raise ValueError('Checkpoint has a nonregular member')
    return members


def durable_publish_checkpoint(output,staging,destination):
    # No state deserialization, optimizer/RNG calls or training policy changes.
    output=Path(output);staging=Path(staging);destination=Path(destination)
    if not output.is_dir() or output.is_symlink():raise ValueError('Require own existing checkpoint output directory')
    persistent_root=Path(os.environ['PATIENTPHEX_DURABILITY_ROOT']).resolve()
    if not persistent_root.is_dir() or persistent_root==output.resolve():raise ValueError('Require explicit existing persistent ancestor root')
    output.resolve().relative_to(persistent_root)

    if staging.is_symlink() or destination.is_symlink():raise ValueError('Checkpoint paths must not be symlinks')
    if staging.parent.resolve()!=output.resolve() or destination.parent.resolve()!=output.resolve():
        raise ValueError('Checkpoint stage and destination must share the own output parent')
    if destination.exists():raise ValueError('Checkpoint destination exists; no overwrite')
    if not staging.is_dir():raise ValueError('Checkpoint staging directory is missing')
    metadata_path=staging/'checkpoint.json'
    if metadata_path.is_symlink():raise ValueError('Checkpoint metadata must not be a symlink')
    metadata_raw=metadata_path.read_bytes();metadata=json.loads(metadata_raw)
    completed=metadata['completed_epochs']
    if type(completed) is not int or completed not in (1,2) or metadata['global_step']!=16*completed or metadata.get('smoke_only') is not False:
        raise ValueError('Require original124 complete-epoch checkpoint accounting')
    if destination.name!=f'epoch-{completed:03d}' or staging.name!=f'.epoch-{completed:03d}.incomplete':
        raise ValueError('Checkpoint epoch/staging namespace differs')
    receipt_path=output/f'durability-epoch-{completed:03d}.json'
    if receipt_path.exists() or receipt_path.is_symlink():raise ValueError('Durability receipt exists; no overwrite')
    source_sha=sha256(__file__)
    if metadata['fingerprint']['source_sha256']!=source_sha:
        raise ValueError('Checkpoint does not belong to this exact trainer source')
    expected=metadata['files']
    if not isinstance(expected,dict) or not expected:raise ValueError('Checkpoint member digest map is empty')
    expected=dict(expected)
    for name,digest in expected.items():
        relative=Path(name)
        if (not isinstance(name,str) or relative.is_absolute() or '..' in relative.parts
            or relative.as_posix()!=name or name=='checkpoint.json'
            or not isinstance(digest,str) or re.fullmatch(r'[0-9a-f]{64}',digest) is None):
            raise ValueError('Unsafe or invalid checkpoint member digest')
    if not {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(expected):
        raise ValueError('Checkpoint lacks full training state or adapter members')
    expected['checkpoint.json']=hashlib.sha256(metadata_raw).hexdigest()
    members=durability_inventory(staging)
    if set(members)!=set(expected):raise ValueError('Checkpoint member inventory differs before publish')
    # Existing writers have closed their files. Flush every member, then all
    # directories deepest-first (including the stage itself) before rename.
    for path in members.values():
        with path.open('rb') as handle:os.fsync(handle.fileno())
    directories=[path for path in staging.rglob('*') if path.is_dir()]
    for path in sorted(directories,key=lambda p:(-len(p.parts),str(p))):durability_fsync_directory(path)
    durability_fsync_directory(staging)
    if destination.exists() or receipt_path.exists():raise ValueError('Checkpoint destination/receipt appeared; refuse overwrite')
    staging.replace(destination)
    # Persist every newly-created ancestor directory entry through the explicit
    # project filesystem root before acknowledging this complete epoch.
    synced_ancestors=[];ancestor=output.resolve()
    while True:
        durability_fsync_directory(ancestor);synced_ancestors.append(str(ancestor))
        if ancestor==persistent_root:break
        ancestor=ancestor.parent
    actual_members=durability_inventory(destination)
    if set(actual_members)!=set(expected):raise ValueError('Checkpoint member inventory differs after publish')
    reopened={}
    for name,path in actual_members.items():
        actual_sha=sha256(path)
        if actual_sha!=expected[name]:raise ValueError('Reopened durable checkpoint SHA mismatch: '+name)
        reopened[name]={'bytes':path.stat().st_size,'sha256':actual_sha}
    # A failed post-rename hash leaves the checkpoint intact with NO success
    # receipt. Recovery must preserve this failure, never relabel it passed.
    receipt={'status':'PASS_file_and_directory_fsync_rename_and_reopened_SHA',
        'completed_epochs':completed,'global_step':metadata['global_step'],
        'checkpoint':str(destination),'source_sha256':source_sha,
        'checkpoint_metadata_sha256':expected['checkpoint.json'],'files':reopened,
        'file_directory_fsync':True,'reopened_SHA_all_exact':True,
        'persistent_root':str(persistent_root),'ancestor_directories_fsynced':synced_ancestors,
        'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'all_members_reopened_and_SHA_verified':True,'file_fsync_count':len(members),
        'checkpoint_directory_fsync_count':len(directories)+1,
        'parent_directory_fsync_after_rename':True,'receipt_parent_directory_fsync_required':True,
        'no_state_deserialization':True,'no_rng_or_training_math_calls':True,
        'completed_unix_seconds':time.time()}
    with receipt_path.open('x',encoding='utf-8') as handle:
        json.dump(receipt,handle,ensure_ascii=False,indent=2);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    durability_fsync_directory(output)
    return receipt


def run(args):
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:raise ValueError('Fresh invocation needs an empty output; use explicit own-run resume')
    if args.resume and (args.mode!='train' or not (output/'run_manifest.json').is_file()):raise ValueError('Resume requires original formal run_manifest')
    if args.resume and json.loads((output/'run_manifest.json').read_text())['fingerprint']['schema']!='patientphex-qwen35-meta32-patient-balanced-alpha05-lora-v1':
        raise ValueError('An old objective run cannot resume as patient-balanced training')
    output.mkdir(parents=True,exist_ok=True)
    started=time.time();stage='source_and_data';control=None;control_path=None;active_binding=None
    def status(name):
        nonlocal stage
        stage=name;print(json.dumps({'stage':name,'mode':args.mode}),flush=True)
    try:
        status('source_and_data');rows,training_ids,excluded_ids=verify_sources_and_data(args)
        import numpy as np
        import torch
        import transformers
        import peft
        import transformers.loss.loss_utils as native_loss
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM,get_linear_schedule_with_warmup
        from peft import PeftModel
        if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':raise RuntimeError('Require isolated TF5.20.2/PEFT0.21.2')
        if torch.__version__!='2.6.0+cu124':raise RuntimeError('Draft target is original A100 Torch2.6.0+cu124; old F1 used2.7.1')
        if sha256(inspect.getfile(native_loss))!=NATIVE_LOSS_SHA:raise ValueError('Native loss_utils source differs')
        mixed_loss,RowLossScope=load_objective_sources()
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
        torch.backends.cudnn.benchmark=False
        status('base_file_fingerprints')
        base_files=model_files(args.model,args.model_source_manifest,args.mode=='train')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='right')
        if tokenizer.pad_token_id is None:raise ValueError('Expected original tokenizer pad token')
        status('token_mask_audit');encoded=encode_all(rows,tokenizer,output,args.train_token_audit,args.native_cpu_receipt)
        status('patient_balanced_row_binding');row_weights,epoch_contract=bind_row_weights(args,rows,encoded,output)
        protocol.causal_loss_probe(torch,output)
        text_config=Qwen3_5TextConfig.from_pretrained(args.model,local_files_only=True)
        text_config._attn_implementation='sdpa';text_config.use_cache=False
        packages={}
        for name in ['torch','transformers','peft','accelerate','numpy','tokenizers','huggingface-hub','safetensors','fla-core','flash-linear-attention','causal-conv1d']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,
            'torch_cuda':torch.version.cuda,'modeling_source_sha256':sha256(inspect.getsourcefile(Qwen3_5ForCausalLM)),
            'peft_source_sha256':sha256(inspect.getsourcefile(PeftModel))}
        if args.mode=='train':
            status('gpu_preflight')
            if not args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Sole CUDA_VISIBLE_DEVICES must equal allocated UUID')
            check=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
            write_json(output/'gpu-preflight.json',{'uuid':args.gpu_uuid,'processes':check.stdout})
            if any(line.split(',')[0].strip()==args.gpu_uuid for line in check.stdout.splitlines()):raise RuntimeError('Allocated GPU already occupied; refuse to share')
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise RuntimeError('Require one BF16 CUDA device')
            environment.update(gpu_name=torch.cuda.get_device_name(0),gpu_uuid=args.gpu_uuid,gpu_capability=list(torch.cuda.get_device_capability(0)))
            torch.cuda.manual_seed_all(CONFIG['seed'])
        fingerprint={'schema':'patientphex-qwen35-meta32-patient-balanced-alpha05-lora-v1','configuration':semantic_configuration(args),
            'source_sha256':sha256(__file__),'helper_sha256':HELPER_SHA,'protocol_sha256':PROTOCOL_SHA,
            'copied_patient_first_trainer_sha256':COPIED_PATIENT_FIRST_TRAINER_SHA,
            'objective':OBJECTIVE,'weights_preparation_sha256':WEIGHTS_PREPARATION_SHA,
            'row_weights_sha256':ROW_WEIGHTS_SHA,'epoch_groups_sha256':EPOCH_GROUPS_SHA,
            'mixed_loss_sha256':MIXED_LOSS_SHA,'row_loss_scope_sha256':ROW_LOSS_SCOPE_SHA,
            'native_loss_utils_sha256':NATIVE_LOSS_SHA,
            'copied_epoch_loop_source_sha256':OLD_TRAINER_SHA,'copied204_trainer_sha256':COPIED_TRAINER_SHA,
            'meta32_preparation_sha256':PREPARATION_SHA,'original_CID_to_PID_train_sha256':OLD_TRAIN_SHA,'train_sha256':TRAIN_SHA,
            'parent_meta32_trainer_sha256':PARENT_TRAINER_SHA,'transposition_manifest_sha256':TRANSPOSITION_SHA,
            'train_token_audit_sha256':TRAIN_TOKEN_AUDIT_SHA,'native_CPU_receipt_sha256':NATIVE_CPU_RECEIPT_SHA,
            'native_CPU_contract_sha256':NATIVE_CPU_CONTRACT_SHA,'schema_helper_sha256':SCHEMA_HELPER_SHA,
            'excluded_input_sha256':EXCLUDED_SHA,'training_document_ids':training_ids,'excluded_document_ids':excluded_ids,
            'base_files':base_files,'environment':environment,'text_config':text_config.to_dict(),
            'encoded_sha256':helpers.json_sha(encoded)}
        # JSON round-trip normalizes config integer keys/tuples before comparing
        # a newly computed fingerprint with its persisted checkpoint form.
        fingerprint=json.loads(json.dumps(fingerprint,default=helpers.json_default,sort_keys=True))
        manifest_path=output/'run_manifest.json'
        if args.resume:verify_fingerprint(json.loads(manifest_path.read_text())['fingerprint'],fingerprint)
        manifest={'fingerprint':fingerprint,'examples':124,'max_tokens':5556,'total_tokens':400614,'supervised_tokens':4694,
            'steps_per_epoch':16,'epochs':2,'total_optimizer_steps':32,'warmup_steps':2,'smoke_only':False,
            'resume_unit':'complete epoch only; partial epoch work must replay','new_gpu_resume_numerical_equivalence_tested':False,
            'fresh_base_on_initial_run':True,'validation_labels_used':False}
        write_json(manifest_path,manifest)
        for name,source in [('train_patient_balanced.py',__file__),('qwen35_smoke.py',inspect.getfile(helpers)),
            ('qwen35_relation_cost.py',inspect.getfile(protocol)),('patient_first.py',args.schema_helper),
            ('mixed_loss.py',inspect.getfile(mixed_loss)),
            ('interface_recovery/row_loss_scope_v2.py',Path(__file__).resolve().parent/'interface_recovery/row_loss_scope_v2.py'),
            ('row-weights.jsonl',args.row_weights),('epoch-groups.json',args.epoch_groups),
            ('weights-preparation-manifest.json',args.weights_preparation_manifest)]:
            copied_source=output/name;copied_source.parent.mkdir(parents=True,exist_ok=True)
            copied_source.write_bytes(Path(source).read_bytes())
        if args.mode=='cpu':
            status('meta_full_hybrid_coverage')
            with torch.device('meta'):
                base=Qwen3_5ForCausalLM(text_config).to(dtype=torch.bfloat16)
                model,named,coverage=protocol.attach_adapter(base,torch,peft)
            if coverage['all_parameter_devices']!=['meta'] or torch.cuda.is_initialized():raise RuntimeError('CPU audit initialized non-meta/GPU parameters')
            write_json(output/'meta-coverage.json',coverage)
            write_json(output/'cpu_complete.json',{'source_sha256':sha256(__file__),'manifest_sha256':sha256(manifest_path),
                'rows':124,'documents':32,'max_tokens':5556,'full_tokens':400614,'supervised_tokens':4694,
                'adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,
                'cuda_initialized':False,'elapsed_seconds':time.time()-started,'not_training':True})
            print(json.dumps({'complete':'cpu','cuda_initialized':False,'source_sha256':sha256(__file__)}),flush=True);return
        checkpoint=Path(args.resume) if args.resume else None;resumed_epochs=0;metadata=None
        if checkpoint:metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
        validate_stop_epoch(args.stop_after_epoch,2,resumed_epochs)
        if (output/'training_complete.json').exists():raise ValueError('Run is already marked complete; refusing train/resume')
        controls=output/'run_controls';controls.mkdir(exist_ok=True);control_path=controls/f'{time.time_ns()}-{os.getpid()}.json'
        scope_directory=output/'loss-scopes'/control_path.stem;scope_directory.mkdir(parents=True)
        def observe_loss_call(actual):
            append_loss_call_observation(scope_directory/'actual-loss-calls.jsonl',actual)
        control={'resume':args.resume,'stop_after_epoch':args.stop_after_epoch,'started_completed_epochs':resumed_epochs,
            'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid(),
            'loss_scope_directory':str(scope_directory)}
        write_json(control_path,control);(output/'paused.json').unlink(missing_ok=True)
        status('load_complete_text_base')
        base,loading=Qwen3_5ForCausalLM.from_pretrained(args.model,config=text_config,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
        write_json(output/'base-loading-info.json',loading)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):raise RuntimeError('Incomplete or mismatched base weights')
        if any(not key.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for key in loading.get('unexpected_keys',[])):raise RuntimeError('Unexpected text checkpoint keys')
        if type(base).__name__!='Qwen3_5ForCausalLM' or any('visual' in n for n,_ in base.named_modules()):raise RuntimeError('Wrong model class')
        if sum(p.numel() for p in base.parameters())!=protocol.BASE_PARAMETERS:raise RuntimeError('Exact base parameter count differs')
        base.config.use_cache=False
        if checkpoint:
            model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=True)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            coverage={'resumed_verified_epoch_adapter':str(checkpoint),'target_names':protocol.TARGETS}
        else:model,named,coverage=protocol.attach_adapter(base,torch,peft)
        if not named or any('.lora_A.' not in n and '.lora_B.' not in n for n in named):raise RuntimeError('Only LoRA may train')
        if sum(p.numel() for p in named.values())!=protocol.ADAPTER_PARAMETERS or sum(p.numel() for p in model.parameters())!=protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS or not sum(p.numel() for p in model.parameters())<10_000_000_000:raise RuntimeError('Exact adapter/total parameter count differs')
        if {protocol.family(n) for n in named}!={'DeltaNet','full_attention','MLP'}:raise RuntimeError('Missing hybrid adapter family')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        if not model.is_gradient_checkpointing:raise RuntimeError('Gradient checkpointing not enabled')
        if model.get_base_model() is not base or base.loss_function is not native_loss.ForCausalLMLoss:
            raise RuntimeError('Require untouched native base instance loss before per-row scopes')
        if base.lm_head.weight.dtype!=torch.bfloat16:raise RuntimeError('Complete native head must remain BF16')
        original_loss=base.loss_function
        require_scope_cleared(base,original_loss)
        write_json(output/'model-coverage.json',coverage)
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        scheduler=get_linear_schedule_with_warmup(optimizer,2,32)
        completed_epochs=global_step=0
        if checkpoint:
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            completed_epochs,global_step=state['completed_epochs'],state['global_step']
            if completed_epochs!=resumed_epochs or global_step!=16*completed_epochs:raise ValueError('Serialized epoch/step differs from checkpoint metadata')
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            if scheduler.last_epoch!=global_step:raise ValueError('Restored scheduler step differs')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        model.train();first_update_checked=False;status('formal_training')
        for epoch in range(completed_epochs,2):
            groups=batch_plan(len(encoded),1,8,CONFIG['seed'],epoch)
            if len(groups)!=16 or sum(len(g) for g in groups)!=124:raise RuntimeError('Frozen epoch batch/step accounting differs')
            loss_sum=0.;epoch_started=time.time()
            for group_index,group in enumerate(groups):
                expected=epoch_contract['epochs'][epoch]['groups'][group_index]
                if [indices[0] for indices in group]!=expected['row_indices'] or len(group)!=expected['group_size']:
                    raise RuntimeError('Actual original epoch group differs from frozen metadata contract')
                require_scope_cleared(base,original_loss);scope_reports=[]
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:probes[protocol.family(name)]=(name,p,p.detach().clone())
                for microbatch,indices in enumerate(group):
                    if len(indices)!=1:raise RuntimeError('Fixed patient-balanced trainer requires original batch1')
                    index=indices[0];metadata=row_weights[index]
                    active_binding={'epoch':epoch+1,'group_index':group_index,'microbatch_index':microbatch,
                        'original_row_index':index,'row_id':metadata['row_id'],'actual_group_size':len(group)}
                    batch=collate([encoded[i] for i in indices],tokenizer.pad_token_id,torch)
                    # One scope owns the original base instance through forward
                    # AND checkpoint backward. Core includes g; outer /g stays once.
                    with RowLossScope(base,metadata,len(group),87,16,mixed_loss.mixed_causal_loss,
                                      native_loss.fixed_cross_entropy,original_loss,active_binding.copy(),observe_loss_call) as scope:
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                        (loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)
                    require_scope_cleared(base,original_loss)
                    scope_reports.append(scope.report());active_binding=None
                write_json(scope_directory/f'epoch-{epoch+1:03d}-group-{group_index:03d}.json',
                    {'epoch':epoch+1,'group_index':group_index,'actual_group_size':len(group),'rows':scope_reports})
                require_scope_cleared(base,original_loss)
                grad_norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['max_grad_norm'])
                if not bool(torch.isfinite(grad_norm)) or float(grad_norm)==0.:raise RuntimeError('Nonfinite/zero LoRA gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen base received a gradient')
                optimizer.step();scheduler.step();global_step+=1
                if probes:
                    if set(probes)!={'DeltaNet','full_attention','MLP'} or any(torch.equal(before,p.detach()) for _,p,before in probes.values()):raise RuntimeError('First nonzero-LR step did not change all three B-family probes')
                    write_json(output/'first-nonzero-update.json',{'step':global_step,'learning_rate':learning_rate,
                        'gradient_norm_before_clip':float(grad_norm),'changed_B_probes':{k:v[0] for k,v in probes.items()},'fresh_run':not bool(checkpoint)})
                    first_update_checked=True
                loss_sum+=group_loss
                if global_step%5==0:print(json.dumps({'epoch':epoch+1,'step':global_step,'loss':group_loss,'learning_rate':learning_rate,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}),flush=True)
            if not first_update_checked:raise RuntimeError('No nonzero-LR adapter update was verified')
            require_scope_cleared(base,original_loss)
            status('save_epoch_checkpoint')
            destination=output/f'epoch-{epoch+1:03d}';staging=output/f'.epoch-{epoch+1:03d}.incomplete'
            if destination.exists() or staging.exists():raise ValueError('Checkpoint destination/staging exists; no overwrite')
            staging.mkdir();model.save_pretrained(staging/'adapter',safe_serialization=True)
            if not all((staging/'adapter'/n).is_file() for n in ['adapter_model.safetensors','adapter_config.json']):raise RuntimeError('Adapter safe weights/config missing')
            state={'completed_epochs':epoch+1,'global_step':global_step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,staging/'training_state.pt')
            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            write_json(staging/'checkpoint.json',{'fingerprint':fingerprint,'files':files,'completed_epochs':epoch+1,'global_step':global_step,'smoke_only':False,'new_resume_numerical_equivalence_tested':False})
            durable_publish_checkpoint(output,staging,destination)
            write_json(output/'progress.json',{'completed_epochs':epoch+1,'global_step':global_step,'mean_accumulation_group_loss':loss_sum/len(groups),'epoch_seconds':time.time()-epoch_started,'checkpoint':str(destination),'training_only':True})
            outcome=epoch_outcome(epoch+1,2,args.stop_after_epoch);control.update(status=outcome,completed_epochs=epoch+1,global_step=global_step,checkpoint=str(destination),updated_unix_seconds=time.time());write_json(control_path,control)
            print(json.dumps({'status':outcome,'epoch':epoch+1,'global_step':global_step,'checkpoint':str(destination)}),flush=True)
            if outcome=='paused':
                require_scope_cleared(base,original_loss)
                write_json(output/'paused.json',{'completed_epochs':epoch+1,'configured_epochs':2,'global_step':global_step,'checkpoint':str(destination),'control_record':str(control_path),'result_type':'paused_complete_epoch_not_training_complete'});return
            status('formal_training')
        (output/'paused.json').unlink(missing_ok=True)
        require_scope_cleared(base,original_loss)
        write_json(output/'training_complete.json',{'smoke_only':False,'epochs':2,'global_step':global_step,'base_parameters':protocol.BASE_PARAMETERS,
            'trainable_adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),
            'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False})
        print(json.dumps({'complete':'training','epochs':2,'steps':global_step}),flush=True)
    except Exception as error:
        if control is not None:
            control.update(status='failed',failed_stage=stage,exception=type(error).__name__,message=str(error));write_json(control_path,control)
        write_json(output/f'failure-{time.time_ns()}.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),
            'source_sha256':sha256(__file__),'elapsed_seconds':time.time()-started,'training_completed':False,
            'active_row_binding':active_binding})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','train'],required=True)
    for name in ['train','excluded-input','data-manifest','transposition-manifest','train-token-audit','native-cpu-receipt','schema-helper','model','model-source-manifest','output-dir']:parser.add_argument('--'+name,required=True)
    for name in ['row-weights','weights-preparation-manifest','epoch-groups']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    validate_stop_epoch(args.stop_after_epoch,2)
    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):parser.error('CPU preparation does not pause/resume training')
    run(args)

if __name__=='__main__':main()
