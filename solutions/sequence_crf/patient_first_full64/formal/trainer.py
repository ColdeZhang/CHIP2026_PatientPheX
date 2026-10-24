#!/usr/bin/env python3
"""Prepared patient-first full64 Qwen3.5 LoRA:265 OOF rows,2 epochs.
Actual native265 bindings are frozen; formal launch still requires task authorization.

Complete epoch checkpoints fsync files/directories, publish atomically, and reopen SHA.
Only this exact patient-first run may resume complete epochs; no old adapter.
On OOM/failure stop and preserve completed checkpoints; never auto-retry.
Length6400 rejects oversized samples; no truncation, packing, or cap padding.

CPU mode verifies data/template/loss/meta without weights or CUDA. Training mode
requires an allocated empty GPU and exact official files. No validation labels,
inference, quantization, packing, smoke adapter or alternative task is supported.
Resume state is saved/restored; numerical GPU resume equivalence is not claimed.
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
CONFIG={'max_seq_length':6400,'batch_size':1,'gradient_accumulation':8,'epochs':2,
        'learning_rate':1e-4,'weight_decay':.01,'warmup_ratio':.05,'max_grad_norm':1.,
        'rank':16,'alpha':32,'dropout':.05,'seed':20261023,'targets':protocol.TARGETS,
        'dtype':'bfloat16','attention':'sdpa','quantization':False,'packing':False,
        'gradient_checkpointing':{'use_reentrant':False}}
OLD_TRAIN_SHA='27ecf4769e2917bfc06c7c47f679d2dea19ac9013626288093b4e4cb6dffab3c'
TRAIN_SHA='4fee5f9c3817730c08673bdd26bf5127d7cd7856d37d3f9082554f5db755c882'
OLD_FULL64_TRAINER_SHA='f83236587d8f307bb6e51ca44c9c69177efce27661b0f889bd00745808030f03'
PARENT_TRAINER_SHA='9d2e0eeb34e6bfa76ea90cbf6f9ffad996ab527e4746bb936e4f3cbc58724b8b'
TRANSPOSITION_SHA='da72bcb8f891452370d0b97ba4c276a4a7013d172664d9ebce7d8f9f8fb23836'
TRAIN_TOKEN_AUDIT_SHA='e6f8f9db7e73e88a53e675e1a32bca20054ddef59cd253bddc79622447fa5684'
NATIVE_CPU_RECEIPT_SHA='8137d850cf78024a7a53be52e9daf6c13cd84944b8aa7495d23db02825688303'
NATIVE_CPU_CONTRACT_SHA='d5c3acc2b5368be89f2859e3bd7d5bc176852772f96e43e598a561cd5d6a68ff'
SCHEMA_HELPER_SHA='25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08'
EXCLUDED_SHA='c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e'
PREPARATION_SHA='389b4a9c81a71f53b0abf99b610f011598a8a3b0a2a42f4ac7bb73d751ccac44'
ALL_ENCODED_SHA='aee739b1462fd1be2f6641f852709023d7051f20f3f94b5dc50d24cdf818e92c'
ALL_RECORDS_SHA='e4e2dcb4f20f53e5b067a6f89068ed9d8388607c3e626f79c309f198cc62c2a0'
SELECTED_ENCODED_SHA='69130af96b46a6f327ff1de6ab2de728d216cc5b52dc61e052a3e36923125789'
sha256=helpers.sha
write_json=helpers.write


def semantic_configuration(args):
    return {'fixed':CONFIG,'paths_and_device':{k:v for k,v in vars(args).items()
            if k not in {'resume','stop_after_epoch'}}}


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
    if len(document_ids) != 64 or set(document_ids) & set(heldout_ids):
        raise ValueError('Require the actual 64-document training set, disjoint from excluded input')
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
    # A placeholder is deliberately not a digest and blocks every execution mode.
    if any(re.fullmatch(r'[0-9a-f]{64}',digest) is None for digest in [NATIVE_CPU_RECEIPT_SHA,NATIVE_CPU_CONTRACT_SHA]):
        raise ValueError('DRAFT ONLY: bind actual native CPU receipt and contract source before executing')
    if sha256(inspect.getfile(helpers))!=HELPER_SHA or sha256(inspect.getfile(protocol))!=PROTOCOL_SHA:
        raise ValueError('Frozen Qwen3.5 helper/protocol source differs')
    fixed_inputs=[(args.train,TRAIN_SHA,3690463),(args.excluded_input,EXCLUDED_SHA,None),
        (args.data_manifest,PREPARATION_SHA,13653),(args.transposition_manifest,TRANSPOSITION_SHA,19253),
        (args.train_token_audit,TRAIN_TOKEN_AUDIT_SHA,348819),
        (args.native_cpu_receipt,NATIVE_CPU_RECEIPT_SHA,None),(args.schema_helper,SCHEMA_HELPER_SHA,7505)]
    for path,digest,size in fixed_inputs:
        if sha256(path)!=digest or (size is not None and Path(path).stat().st_size!=size):
            raise ValueError('Frozen patient-first full64/provenance input differs: '+str(path))
    if sha256(args.model_source_manifest)!=SOURCE_MANIFEST_SHA:
        raise ValueError('Official15-file fixed source manifest differs')
    schema=load_schema_helper(args.schema_helper)
    rows,train_ids,excluded_ids=training_rows(args.train,args.excluded_input,schema)
    if len(rows)!=265 or len(train_ids)!=64 or len(excluded_ids)!=16:
        raise ValueError('Require265 rows/64train documents/16excluded input documents')
    preparation=json.loads(Path(args.data_manifest).read_text())
    # The OLD preparation binds OLD CID-to-PID bytes. The summary supplies
    # the exact new PID-to-CID bridge; do not pretend the old manifest is new.
    if (preparation['training_sha256']!=OLD_TRAIN_SHA or preparation['excluded_dev_input_sha256']!=EXCLUDED_SHA
        or preparation['training_document_ids']!=train_ids or preparation['excluded_document_ids']!=excluded_ids
        or preparation['training_rows']!=265 or preparation['gold_added_candidates']!=0
        or preparation['gold_in_prompts'] is not False or preparation['all_target_documents_excluded_from_own_generators'] is not True
        or preparation['held_partition_union64'] is not True):
        raise ValueError('Old265 training provenance identity/isolation differs')
    transposition=json.loads(Path(args.transposition_manifest).read_text())
    if transposition['status']!='PASS_CPU_patient_first_full64_preparation_no_training_or_quality':
        raise ValueError('Transposition preparation is not complete')
    if (transposition['sources']['train']['sha256']!=OLD_TRAIN_SHA
        or transposition['sources']['preparation_manifest']['sha256']!=PREPARATION_SHA
        or transposition['sources']['encode_source']['sha256']!=HELPER_SHA
        or transposition['transform_source']['sha256']!=SCHEMA_HELPER_SHA
        or transposition['training']['document_ids']!=train_ids or transposition['training']['rows']!=265):
        raise ValueError('New transposition does not bind old train64 provenance')
    artifacts={Path(x['path']).name:x for x in transposition['artifacts']}
    if (artifacts['train265.patient-first.jsonl']['sha256']!=TRAIN_SHA
        or artifacts['train-token-audit.json']['sha256']!=TRAIN_TOKEN_AUDIT_SHA):
        raise ValueError('Transposed output/audit binding differs')
    dev=transposition['dev_input_only'];dev_ids=dev['document_ids']
    if (dev_ids!=excluded_ids or len(set(dev_ids))!=16 or set(dev_ids)&set(train_ids)
        or dev['targets_read'] is not False or dev['answer_fields']!=0 or dev['rows']!=64):
        raise ValueError('Transposition train/dev isolation differs')
    isolation=transposition['isolation']
    if (isolation['training_vs_dev_overlap']!=0
        or isolation['all_four_generator_train48_target16_chains_exclude_dev'] is not True
        or isolation['assembly_semantic_CRF_train_input_SHA_match_own_chain'] is not True
        or isolation['original265_input_fields_exact_against_four_pure_prompt_files'] is not True
        or isolation['no_gold_added_candidates'] is not True
        or transposition['training']['double_inverse_all_fields_and_targets_byteexact'] is not True):
        raise ValueError('Transposition source-chain or supervision bridge differs')
    provenance=preparation['provenance']
    if len(provenance)!=4 or sorted(x['fold'] for x in provenance)!=[0,1,2,3]:
        raise ValueError('Require original four full64 generator chains')
    all_targets=[]
    for item in provenance:
        fold=item['fold'];generator=item['train_ids'];target=item['input_ids']
        if (len(generator)!=len(set(generator)) or len(generator)!=48
            or len(target)!=len(set(target)) or len(target)!=16
            or set(generator)&set(target) or set(generator)|set(target)!=set(train_ids)
            or (set(generator)|set(target))&set(excluded_ids)):
            raise ValueError('Generator train48/target16/dev isolation differs')
        for prefix,field in [('assembly_manifest','assembly_manifest_sha256'),
                             ('semantic_manifest','semantic_manifest_sha256'),
                             ('crf_manifest','crf_manifest_sha256'),
                             ('training_prompts','prompts_sha256')]:
            if transposition['sources'][f'{prefix}_{fold}']['sha256']!=item[field]:
                raise ValueError('Four-chain source SHA bridge differs')
        all_targets.extend(target)
    if len(all_targets)!=64 or len(set(all_targets))!=64 or set(all_targets)!=set(train_ids):
        raise ValueError('Four target folds must partition exactly the train64 documents')
    statistics=transposition['training']['token_statistics']['patient_first']
    if (statistics['full_tokens']['sum']!=860405 or statistics['full_tokens']['maximum']!=6166
        or statistics['supervised_tokens']['sum']!=9818 or statistics['longest_row']!='3563609:relations:0'
        or transposition['training']['new_encoded_helper_json_sha256']!=ALL_ENCODED_SHA
        or transposition['training']['new_records_helper_json_sha256']!=ALL_RECORDS_SHA
        or transposition['training']['longest_new']['encoded_helper_json_sha256']!=SELECTED_ENCODED_SHA):
        raise ValueError('Transposition token/hash binding differs')
    cpu=json.loads(Path(args.native_cpu_receipt).read_text())
    expected_sources={'train':TRAIN_SHA,'train_token_audit':TRAIN_TOKEN_AUDIT_SHA,'summary':TRANSPOSITION_SHA,
        'preparation_manifest':PREPARATION_SHA,'encode_helper':HELPER_SHA,'transform':SCHEMA_HELPER_SHA}
    if any(cpu['sources'][name]['sha256']!=digest for name,digest in expected_sources.items()):
        raise ValueError('Actual native CPU receipt source chain differs')
    if (cpu['status']!='PASS_native_TF5202_CPU_patient_first_train265_token_contract'
        or cpu['data_contract_source']['sha256']!=NATIVE_CPU_CONTRACT_SHA
        or cpu['training_document_ids']!=train_ids or cpu['dev_document_ids_from_summary_only']!=dev_ids
        or (cpu['rows'],cpu['total_full_tokens'],cpu['max_full_tokens'],cpu['supervised_tokens'])!=(265,860405,6166,9818)
        or cpu['all_rows_patient_first_token_audit_exact'] is not True or cpu['training_dev_overlap']!=0
        or cpu['truncation'] or cpu['dev_files_read']!=0 or cpu['runtime']['cuda_initialized']
        or cpu['encoded_helper_json_sha256']!=ALL_ENCODED_SHA
        or cpu['records_helper_json_sha256']!=ALL_RECORDS_SHA
        or cpu['selected_encoded_helper_json_sha256']!=SELECTED_ENCODED_SHA):
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
    if len(audit)!=265 or len({x['row_id'] for x in audit})!=265:
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
    if len(encoded)!=265 or max(lengths)!=6166 or sum(lengths)!=860405 or any(n>CONFIG['max_seq_length'] for n in lengths):
        raise ValueError('Frozen patient-first265 token accounting differs; no truncation permitted')
    if sum(e['supervised_tokens'] for e in encoded)!=9818:
        raise ValueError('Frozen patient-first supervised-token accounting differs')
    if (helpers.json_sha(encoded)!=cpu['encoded_helper_json_sha256'] or helpers.json_sha(encoded)!=ALL_ENCODED_SHA
        or helpers.json_sha(records)!=cpu['records_helper_json_sha256'] or helpers.json_sha(records)!=ALL_RECORDS_SHA):
        raise ValueError('All265 encoded rows differ from passed actual native CPU receipt')
    longest=[item for item in encoded if len(item['input_ids'])==6166]
    if (len(longest)!=1 or longest[0]['row_id']!='3563609:relations:0'
        or longest[0]['supervised_tokens']!=122 or longest[0]['prefix_tokens']!=6044
        or helpers.json_sha(longest[0])!=cpu['selected_encoded_helper_json_sha256'] or helpers.json_sha(longest[0])!=SELECTED_ENCODED_SHA):
        raise ValueError('Passed6166 longest row binding differs')
    write_json(output/'token-mask-audit.json',{'rows':records,
        'rows_count':265,'total_full_tokens':sum(lengths),'max_full_tokens':max(lengths),
        'supervised_tokens':9818,'truncated_rows':0,'prefix_mask_eos_passed':True,
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
    # H44: ACK files and other prefix matches never become checkpoint candidates.
    epoch_directories=[p for p in output.iterdir() if p.is_dir() and re.fullmatch(r'epoch-[0-9]{3}',p.name)]
    if checkpoint.resolve()!=expected.resolve() or checkpoint.name!=max(p.name for p in epoch_directories):
        raise ValueError('Resume requires latest complete epoch checkpoint')
    if not 1<=completed<2 or metadata['global_step']!=34*completed or metadata.get('smoke_only'):
        raise ValueError('Checkpoint is not an unfinished formal265x2 run')
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
    if type(completed) is not int or completed not in (1,2) or metadata['global_step']!=34*completed or metadata.get('smoke_only') is not False:
        raise ValueError('Require original265 complete-epoch checkpoint accounting')
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
    output.mkdir(parents=True,exist_ok=True)
    started=time.time();stage='source_and_data';control=None;control_path=None
    def status(name):
        nonlocal stage
        stage=name;print(json.dumps({'stage':name,'mode':args.mode}),flush=True)
    try:
        status('source_and_data');rows,training_ids,excluded_ids=verify_sources_and_data(args)
        import numpy as np
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM,get_linear_schedule_with_warmup
        from peft import PeftModel
        if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':raise RuntimeError('Require isolated TF5.20.2/PEFT0.21.2')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
        torch.backends.cudnn.benchmark=False
        status('base_file_fingerprints')
        base_files=model_files(args.model,args.model_source_manifest,args.mode=='train')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='right')
        if tokenizer.pad_token_id is None:raise ValueError('Expected original tokenizer pad token')
        status('token_mask_audit');encoded=encode_all(rows,tokenizer,output,args.train_token_audit,args.native_cpu_receipt)
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
        fingerprint={'schema':'patientphex-qwen35-full64-patient-first-lora-v1','configuration':semantic_configuration(args),
            'source_sha256':sha256(__file__),'helper_sha256':HELPER_SHA,'protocol_sha256':PROTOCOL_SHA,
            'copied_epoch_loop_source_sha256':OLD_TRAINER_SHA,'parent204_trainer_sha256':PARENT_TRAINER_SHA,'old_full64_trainer_sha256':OLD_FULL64_TRAINER_SHA,
            'full64_preparation_sha256':PREPARATION_SHA,'original_CID_to_PID_train_sha256':OLD_TRAIN_SHA,'train_sha256':TRAIN_SHA,
            'transposition_manifest_sha256':TRANSPOSITION_SHA,'train_token_audit_sha256':TRAIN_TOKEN_AUDIT_SHA,
            'native_CPU_receipt_sha256':NATIVE_CPU_RECEIPT_SHA,'native_CPU_contract_sha256':NATIVE_CPU_CONTRACT_SHA,
            'schema_helper_sha256':SCHEMA_HELPER_SHA,'excluded_input_sha256':EXCLUDED_SHA,'training_document_ids':training_ids,'excluded_document_ids':excluded_ids,
            'base_files':base_files,'environment':environment,'text_config':text_config.to_dict(),
            'encoded_sha256':helpers.json_sha(encoded)}
        # JSON round-trip normalizes config integer keys/tuples before comparing
        # a newly computed fingerprint with its persisted checkpoint form.
        fingerprint=json.loads(json.dumps(fingerprint,default=helpers.json_default,sort_keys=True))
        manifest_path=output/'run_manifest.json'
        if args.resume:verify_fingerprint(json.loads(manifest_path.read_text())['fingerprint'],fingerprint)
        manifest={'fingerprint':fingerprint,'examples':265,'max_tokens':6166,'total_tokens':860405,'supervised_tokens':9818,
            'steps_per_epoch':34,'epochs':2,'total_optimizer_steps':68,'warmup_steps':4,'smoke_only':False,
            'resume_unit':'complete epoch only; partial epoch work must replay','new_gpu_resume_numerical_equivalence_tested':False,
            'fresh_base_on_initial_run':True,'validation_labels_used':False}
        write_json(manifest_path,manifest)
        for name,source in [('train_patient_first_full64.py',__file__),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol)),('patient_first.py',args.schema_helper)]:
            (output/name).write_bytes(Path(source).read_bytes())
        if args.mode=='cpu':
            status('meta_full_hybrid_coverage')
            with torch.device('meta'):
                base=Qwen3_5ForCausalLM(text_config).to(dtype=torch.bfloat16)
                model,named,coverage=protocol.attach_adapter(base,torch,peft)
            if coverage['all_parameter_devices']!=['meta'] or torch.cuda.is_initialized():raise RuntimeError('CPU audit initialized non-meta/GPU parameters')
            write_json(output/'meta-coverage.json',coverage)
            write_json(output/'cpu_complete.json',{'source_sha256':sha256(__file__),'manifest_sha256':sha256(manifest_path),
                'rows':265,'documents':64,'max_tokens':6166,'full_tokens':860405,'supervised_tokens':9818,
                'adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,
                'cuda_initialized':False,'elapsed_seconds':time.time()-started,'not_training':True})
            print(json.dumps({'complete':'cpu','cuda_initialized':False,'source_sha256':sha256(__file__)}),flush=True);return
        checkpoint=Path(args.resume) if args.resume else None;resumed_epochs=0;metadata=None
        if checkpoint:metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
        validate_stop_epoch(args.stop_after_epoch,2,resumed_epochs)
        if (output/'training_complete.json').exists():raise ValueError('Run is already marked complete; refusing train/resume')
        controls=output/'run_controls';controls.mkdir(exist_ok=True);control_path=controls/f'{time.time_ns()}-{os.getpid()}.json'
        control={'resume':args.resume,'stop_after_epoch':args.stop_after_epoch,'started_completed_epochs':resumed_epochs,
            'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid()}
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
        write_json(output/'model-coverage.json',coverage)
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        scheduler=get_linear_schedule_with_warmup(optimizer,4,68)
        completed_epochs=global_step=0
        if checkpoint:
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            completed_epochs,global_step=state['completed_epochs'],state['global_step']
            if completed_epochs!=resumed_epochs or global_step!=34*completed_epochs:raise ValueError('Serialized epoch/step differs from checkpoint metadata')
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            if scheduler.last_epoch!=global_step:raise ValueError('Restored scheduler step differs')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        model.train();first_update_checked=False;status('formal_training')
        for epoch in range(completed_epochs,2):
            groups=batch_plan(len(encoded),1,8,CONFIG['seed'],epoch)
            if len(groups)!=34 or sum(len(g) for g in groups)!=265:raise RuntimeError('Frozen epoch batch/step accounting differs')
            loss_sum=0.;epoch_started=time.time()
            for group in groups:
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:probes[protocol.family(name)]=(name,p,p.detach().clone())
                for indices in group:
                    batch=collate([encoded[i] for i in indices],tokenizer.pad_token_id,torch)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                    (loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)
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
                write_json(output/'paused.json',{'completed_epochs':epoch+1,'configured_epochs':2,'global_step':global_step,'checkpoint':str(destination),'control_record':str(control_path),'result_type':'paused_complete_epoch_not_training_complete'});return
            status('formal_training')
        (output/'paused.json').unlink(missing_ok=True)
        write_json(output/'training_complete.json',{'smoke_only':False,'epochs':2,'global_step':global_step,'base_parameters':protocol.BASE_PARAMETERS,
            'trainable_adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),
            'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False})
        print(json.dumps({'complete':'training','epochs':2,'steps':global_step}),flush=True)
    except Exception as error:
        if control is not None:
            control.update(status='failed',failed_stage=stage,exception=type(error).__name__,message=str(error));write_json(control_path,control)
        write_json(output/f'failure-{time.time_ns()}.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),
            'source_sha256':sha256(__file__),'elapsed_seconds':time.time()-started,'training_completed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','train'],required=True)
    for name in ['train','excluded-input','model','model-source-manifest','output-dir','data-manifest','transposition-manifest','train-token-audit','native-cpu-receipt','schema-helper']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    validate_stop_epoch(args.stop_after_epoch,2)
    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):parser.error('CPU preparation does not pause/resume training')
    run(args)

if __name__=='__main__':main()
