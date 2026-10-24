#!/usr/bin/env python3
"""Freeze both O/O and R/O full predictions before one F1 evaluation per arm.

Reuse the historical parser, response verifier, format validator and all/text v2.
This is F1 development from a known old two-epoch baseline, not held-out evidence.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time

ARMS = ('original', 'reverse-first')
METRICS = ('Score', 'menF1', 'docF1', 'micF1', 'macF1')
SCHEMA = 'patientphex-order-warmstart-paired-F1-v1'
STAGE_SCHEMA = 'patientphex-qwen35-meta32-order-warmstart-stage-v1'
MOUNT_ID = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
WARM_CP = '5cecc79c3286089ad413b94ec1c52c0f78e84df2e3fc8b161ca2168f6a9e0120'
WARM_FILES = {'adapter_model.safetensors': '4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff',
              'adapter_config.json': '96564441328e6759568e60c49ea87db432b9ec45cfd57001ef81640d4ff9c4f9'}
TRAIN_SHA = 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
EXCLUDED_SHA = '2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
REVERSE_SHA = '34778f082d1057cab4646a3ef259fd0f15f4e48f13944ad5216bbde042e609b8'
GOLD_SHA = '665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3'
SHARED_SHA = {
    'tree_helper': '6093741856c42fe56dc0f13d365b3eaf4c962e2f67befa5fa3ce865362989bef',
    'fusion_helper': '53a954f0b31db34e87e64ccf8733446d426defeb084e4fa2ba3623489a3b7942',
    'parser': '817fdc4494e7ed349085fe97c5744409d66d0f1de476089ac8c5696a84a6b4a1',
    'parser_base': '41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0',
    'input': '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415',
    'old_entities': '8acd182b5eee9e1cf56d43f299924de410635d2bc309c8c10bf51be9a59b7068',
    'model_source': '74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7',
    'prompts': 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe',
    'old_baseline_pred': 'ffb2c54ac4af0136fe1bb4fa08a34e14d3e0e27c4354a2b9c65aa50ea0f5f379',
    'old_baseline_metrics': '7e36dd93a436bdaadd4b0527b1bb388d6e438d8f4bcf8748d00020a3a1c4f477',
    'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
    'evaluator': 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781',
    'validator': 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f',
    'runner': '33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669',
}
ARM_REFS = {'training_manifest', 'training_complete', 'checkpoint', 'export_ack',
            'responses', 'response_manifest', 'response_freeze', 'inference_plan'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked(record):
    require(ref(record['path']) == record, 'Changed referenced file: ' + record['path'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def command(argv, output):
    result = subprocess.run(argv, text=True, capture_output=True)
    save(output, {'argv': argv, 'returncode': result.returncode,
                  'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0, result.stderr)


def plan_references(plan):
    """Validate reference structure without opening any data, especially gold."""
    require(plan['schema'] == SCHEMA and set(plan['arms']) == set(ARMS), 'Wrong paired plan')
    require(set(plan['references']) == set(SHARED_SHA), 'Shared reference whitelist differs')
    require(plan['historical_F1_gold_sha256'] == GOLD_SHA, 'Historical F1 gold pin differs')
    records = list(plan['references'].values())
    for arm in ARMS:
        require(set(plan['arms'][arm]) == {'references'}, 'Unexpected arm plan fields')
        require(set(plan['arms'][arm]['references']) == ARM_REFS, 'Arm reference whitelist differs')
        records.extend(plan['arms'][arm]['references'].values())
    gold = Path(plan['gold_path']).resolve()
    for record in records:
        require(set(record) == {'path', 'bytes', 'sha256'}, 'Require three-field immutable references')
        path = Path(record['path']).resolve()
        require(path != gold, 'Gold must never be a prechecked reference')
        if path.exists() and gold.exists():
            require(not os.path.samefile(path, gold), 'Gold hardlink cannot be prechecked')
    for name, digest in SHARED_SHA.items():
        require(plan['references'][name]['sha256'] == digest, 'Historical public reference differs: ' + name)
    return records


def verify_training(arm, checkpoint, manifest, terminal, trainer_sha):
    variants = ['O', 'O'] if arm == 'original' else ['R', 'O']
    fp = checkpoint['fingerprint']
    require(checkpoint['completed_epochs'] == 2 and checkpoint['global_step'] == 32
            and checkpoint['smoke_only'] is False, 'Require complete final new-stage epoch2')
    require(fp == manifest['fingerprint'] and fp['schema'] == STAGE_SCHEMA, 'Training fingerprint differs')
    require(fp['source_sha256'] == terminal['source_sha256'] == trainer_sha, 'Actual trainer source differs')
    require(fp['order_arm'] == terminal['order_arm'] == arm and fp['epoch_variants'] == variants,
            'Arm/order mismatch')
    require(fp['train_sha256'] == TRAIN_SHA and fp['excluded_input_sha256'] == EXCLUDED_SHA
            and fp['reverse_train_sha256'] == REVERSE_SHA, 'Strict32 training inputs differ')
    train_ids, excluded_ids = fp['training_document_ids'], fp['excluded_document_ids']
    require(len(train_ids) == len(set(train_ids)) == 32 and len(excluded_ids) == len(set(excluded_ids)) == 32
            and not set(train_ids) & set(excluded_ids), 'Strict32 isolation differs')
    warm = fp['warm_start']
    require(warm == manifest['warm_start'] == terminal['warm_start'], 'Warm-start provenance differs')
    require(warm['kind'] == 'new_stage_adapter_warm_start_fresh_optimizer_RNG'
            and warm['source_checkpoint_sha256'] == WARM_CP and warm['adapter_files'] == WARM_FILES
            and warm['source_epochs'] == 2 and warm['source_optimizer_calls'] == 32
            and warm['old_optimizer_loaded'] is False and warm['old_training_state_opened'] is False
            and warm['not_exact_old_stage_resume'] is True, 'Require exact old32 adapter with fresh new stage')
    require(manifest['examples'] == 124 and manifest['steps_per_epoch'] == 16
            and manifest['epochs'] == manifest['new_stage_epochs'] == manifest['old_stage_completed_epochs'] == 2
            and manifest['total_optimizer_steps'] == 32 and manifest['warmup_steps'] == 2
            and manifest['smoke_only'] is False and manifest['validation_labels_used'] is False
            and manifest['fresh_optimizer_on_initial_run'] is True
            and manifest['fresh_base_on_initial_run'] is False and manifest['epoch_variants'] == variants,
            'Require matched 248 exposures/32 calls, no validation-label training')
    require(terminal['epochs'] == terminal['new_stage_epochs'] == terminal['old_stage_epochs'] == 2
            and terminal['global_step'] == 32 and terminal['smoke_only'] is False
            and terminal['base_parameters'] == 8953803264
            and terminal['trainable_adapter_parameters'] == 43278336
            and terminal['total_parameters'] == 8997081600
            and terminal['started_completed_epochs'] == 1
            and terminal['elapsed_scope'] == 'current_invocation_only_not_full_stage'
            and terminal['full_stage_elapsed_seconds'] is None, 'Final-stage completion contract differs')
    adapter = {name: checkpoint['files']['adapter/' + name] for name in WARM_FILES}
    require(adapter['adapter_model.safetensors'] != WARM_FILES['adapter_model.safetensors'],
            'Final adapter is unchanged warm-start source')
    return adapter


def comparable_fingerprint(fingerprint):
    result = json.loads(json.dumps(fingerprint))
    result.pop('order_arm')
    result.pop('epoch_variants')
    config = result['configuration']['paths_and_device']
    config.pop('order_arm')
    config.pop('output_dir')
    return result


def verify_export(receipt, checkpoint, refs):
    persistent = PurePosixPath(receipt['persistent_directory'])
    prefix = ('case-retrieval', 'qwen35-preparation', 'order-warmstart-stage', 'checkpoints')
    require(receipt['status'] == 'persistent_checkpoint_verified' and receipt['persistent_mount_id'] == MOUNT_ID
            and receipt['fsync_all_files'] is True and receipt['completed_epochs'] == 2
            and receipt['global_step'] == 32 and Path(receipt['source_checkpoint_directory']).name == 'epoch-002'
            and not persistent.is_absolute() and '..' not in persistent.parts
            and persistent.parts[:len(prefix)] == prefix and len(persistent.parts) > len(prefix),
            'Final persistent export receipt differs')
    require(receipt['checkpoint_sha256'] == refs['checkpoint']['sha256']
            and receipt['run_manifest_sha256'] == refs['training_manifest']['sha256'], 'Export metadata differs')
    expected = dict(checkpoint['files'])
    expected.update({'checkpoint.json': refs['checkpoint']['sha256'],
                     '_run/run_manifest.json': refs['training_manifest']['sha256']})
    require({'training_state.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json'} <= set(expected),
            'Incomplete epoch export')
    for name, record in receipt['files'].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and record['bytes'] > 0
                and len(record['sha256']) == 64, 'Invalid exported file record')
    for name, digest in expected.items():
        require(receipt['files'][name]['sha256'] == digest, 'Exported actual bytes differ: ' + name)


def delta(candidate, baseline):
    require(all(math.isfinite(candidate[k]) and math.isfinite(baseline[k]) for k in METRICS), 'Nonfinite metric')
    return {key: candidate[key] - baseline[key] for key in METRICS}


def keep_suggestion(candidate, baseline):
    return candidate['Score'] > baseline['Score'] and candidate['macF1'] >= baseline['macF1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan_path = directory / 'run-plan.json'
    plan = read(plan_path)
    require(sha(__file__) == plan['control_sha256'], 'Evaluation source differs from plan')
    records = plan_references(plan)
    for record in records:
        checked(record)
    require(Path(plan['references']['parser_base']['path']).resolve()
            == Path(plan['references']['parser']['path']).resolve().with_name('run.py'), 'Parser sibling differs')
    # Exclusive reservation prevents repeat/parallel execution in this evaluation directory.
    save(directory / 'evaluation-reservation.json', {'at_utc': now(), 'plan': ref(plan_path), 'quality_started': False})
    started = time.monotonic()
    shared = plan['references']
    h = module('order_warmstart_tree', shared['tree_helper']['path'])
    f = module('order_warmstart_fusion', shared['fusion_helper']['path'])
    h.verify_parser(shared['parser']['path'])
    prompts = rows(shared['prompts']['path'])
    pairs = h.pair_rows(prompts)
    require(len(prompts) == 80 and len(pairs) == 1557 and sum(len(p['code_map']) for p in prompts) == 587,
            'Require unchanged original80 prompts/1557 literal pairs/587 CID decisions')
    baseline = rows(shared['old_baseline_pred']['path'])
    require(len(baseline) == 16 and sum(len(d['entities']) for d in baseline) == 1189, 'Cached old prediction differs')
    nonrelation = [{k: v for k, v in d.items() if k != 'association'} for d in baseline]
    model_source = read(shared['model_source']['path'])
    states, frozen = {}, {}
    # Verify BOTH complete raw arms and final lineage before assembling either.
    for arm in ARMS:
        refs = plan['arms'][arm]['references']
        cp, manifest, terminal = (read(refs[k]['path']) for k in ('checkpoint', 'training_manifest', 'training_complete'))
        adapter = verify_training(arm, cp, manifest, terminal, plan['trainer_sha256'])
        verify_export(read(refs['export_ack']['path']), cp, refs)
        infer = read(refs['inference_plan']['path'])
        require(infer['checkpoint_sha256'] == refs['checkpoint']['sha256']
                and infer['training_complete_sha256'] == refs['training_complete']['sha256']
                and infer['adapter_files'] == adapter
                and infer['prompts_sha256'] == shared['prompts']['sha256']
                and infer['runner_sha256'] == shared['runner']['sha256']
                and infer['model_source_sha256'] == shared['model_source']['sha256'], 'Inference plan lineage differs')
        rawfreeze = read(refs['response_freeze']['path'])
        require(rawfreeze['plan_sha256'] == refs['inference_plan']['sha256']
                and rawfreeze['raw_sha256'] == refs['responses']['sha256']
                and rawfreeze['manifest_sha256'] == refs['response_manifest']['sha256']
                and rawfreeze['rows'] == 80, 'Complete80 response freeze differs')
        raw, response_manifest = rows(refs['responses']['path']), read(refs['response_manifest']['path'])
        require(len(raw) == 80 and response_manifest['fingerprint']['configuration']['adapter'] == infer['adapter_directory'],
                'Wrong final inference adapter/row count')
        f.verify_sft_response(h, response_manifest, raw, refs['responses']['path'], shared['prompts']['path'], adapter, model_source)
        _, parse_stats = h.llm_features(prompts, raw, pairs)
        states[arm] = {'fingerprint': cp['fingerprint'], 'parse': parse_stats, 'adapter_files': adapter}
    require(comparable_fingerprint(states[ARMS[0]]['fingerprint']) == comparable_fingerprint(states[ARMS[1]]['fingerprint']),
            'Arms differ beyond candidate-order schedule and output directory')
    for arm in ARMS:
        out = directory / arm
        out.mkdir(exist_ok=False)
        refs = plan['arms'][arm]['references']
        prediction = out / 'pred.jsonl'
        command([plan['python'], shared['parser']['path'], '--input', shared['input']['path'],
                 '--entities', shared['old_entities']['path'], '--prompts', shared['prompts']['path'],
                 '--responses', refs['responses']['path'], '--output', str(prediction), '--fallback', 'baseline'],
                out / 'assembly-command.json')
        assembled = rows(prediction)
        require(len(assembled) == 16 and sum(len(d['entities']) for d in assembled) == 1189
                and [{k: v for k, v in d.items() if k != 'association'} for d in assembled] == nonrelation,
                'Full prediction changed nonrelation fields or entity coverage')
        assembly = read(out / 'pred.manifest.json')
        require(assembly['code_sha256'] == shared['parser']['sha256']
                and {k: assembly['stats'].get(k, 0) for k in states[arm]['parse']} == states[arm]['parse'],
                'Assembly parser/statistics differ')
        frozen[arm] = {'prediction': ref(prediction), 'assembly_manifest': ref(out / 'pred.manifest.json'),
                       'responses': refs['responses'], 'response_freeze': refs['response_freeze'],
                       'checkpoint': refs['checkpoint'], 'export_ack': refs['export_ack'],
                       'adapter_files': states[arm]['adapter_files'], 'parse': states[arm]['parse']}
    for record in records:
        checked(record)
    paired = directory / 'paired-predictions-freeze.json'
    save(paired, {'status': 'both_complete80_full_predictions_frozen_before_gold', 'at_utc': now(),
                 'plan': ref(plan_path), 'old': shared['old_baseline_pred'], 'arms': frozen,
                 'documents': 16, 'entities': 1189, 'nonrelation_fields_exact': True})
    save(directory / 'evaluation-started.json', {'at_utc': now(), 'paired_freeze': ref(paired),
                                              'one_v2_call_per_new_arm': True, 'old_metrics_cached_only': True})
    for arm in ARMS:
        out = directory / arm
        command([plan['python'], shared['validator']['path'], '--input', shared['input']['path'],
                 '--pred', frozen[arm]['prediction']['path'], '--hpo', shared['hpo']['path'],
                 '--output', str(out / 'validation.json')], out / 'validation-command.json')
        require(read(out / 'validation.json')['valid'], 'Prediction format invalid: ' + arm)
    # FIRST gold byte access: both full predictions and their shared freeze already exist.
    require(sha(plan['gold_path']) == plan['historical_F1_gold_sha256'], 'Historical F1 gold changed')
    metrics = {}
    old = read(shared['old_baseline_metrics']['path'])
    for arm in ARMS:
        out = directory / arm
        checked(frozen[arm]['prediction'])
        command([plan['python'], shared['evaluator']['path'], '--gold', plan['gold_path'],
                 '--pred', frozen[arm]['prediction']['path'], '--body-scope', 'all', '--no-id-doc', 'text',
                 '--output', str(out / 'metrics.json')], out / 'evaluation-command.json')
        result = read(out / 'metrics.json')
        require(result['documents'] == 16 and result['patients'] == 39
                and all(result['metrics'][k] == old['metrics'][k] for k in ('menF1', 'docF1')),
                'Fixed F1 coverage/nonrelation score changed')
        metrics[arm] = result
    for record in records:
        checked(record)
    for arm in ARMS:
        checked(frozen[arm]['prediction'])
        checked(frozen[arm]['assembly_manifest'])
    comparisons = {arm + '_minus_old': delta(metrics[arm]['metrics'], old['metrics']) for arm in ARMS}
    comparisons['reverse-first_minus_original'] = delta(metrics['reverse-first']['metrics'], metrics['original']['metrics'])
    summary = {'status': 'complete_paired_warmstart_original80_F1_development', 'paired_freeze': ref(paired),
               'scope': 'Known old two-epoch baseline; repeatedly observed F1 development, not independent generalization or official score.',
               'interpretation': 'Compare matched new-stage O/O against R/O. Comparisons to old include additional training and optimizer/RNG restart; do not attribute them solely to candidate order or extra epochs.',
               'old_cached_metrics': shared['old_baseline_metrics'], 'old_metrics': old['metrics'],
               'arms': {arm: {'metrics': metrics[arm]['metrics'], 'counts': metrics[arm]['counts'],
                              'metrics_file': ref(directory / arm / 'metrics.json'),
                              'format_file': ref(directory / arm / 'validation.json'),
                              'adapter_files': states[arm]['adapter_files']} for arm in ARMS},
               'delta': comparisons,
               'suggest_keep_vs_old': {arm: keep_suggestion(metrics[arm]['metrics'], old['metrics']) for arm in ARMS},
               'suggest_reverse_first_vs_original': keep_suggestion(metrics['reverse-first']['metrics'], metrics['original']['metrics']),
               'keep_rule': 'Score strictly increases and macF1 does not decrease; advisory only.',
               'automatic_migration': False, 'fit_union_threshold_search': False, 'outer_F0_or_official_used': False,
               'elapsed_CPU_seconds': time.monotonic() - started}
    save(directory / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
