#!/usr/bin/env python3
"""Predeclared Med standalone and old-SFT32 union, frozen together before F1."""
from pathlib import Path, PurePosixPath
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import time

UTILITY = Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry/evaluate_paired.py')
UTILITY_SHA = '563da3a19527b28e96e07418eb2fee063762301faf9492630292d61615e8a79d'
SCHEMA = 'patientphex-medgemma-primary-union-F1-v1'
ENDPOINTS = ('primary-medgemma', 'secondary-union')
POLICY = {
    'primary': 'MedGemma native full-bundle text-only fresh strict32 two-epoch adapter; frozen original80 parser and baseline-rule fallback',
    'secondary': 'literal per-document per-patient set union of cached old-SFT32 and primary MedGemma relations',
    'selection_comparator': 'cached old-SFT32 baseline',
    'retention': 'relation micF1+macF1 strictly increases AND macF1 does not decrease',
    'union_vs_medgemma': 'report differences only; no additional selection/combinations',
    'evaluation': 'both complete predictions frozen first; each one unchanged v2 all/text call; old metrics cached',
    'nonrelation': 'original baseline bytes outside each top-level association value preserved exactly',
    'automatic_migration': False, 'parameter_or_threshold_search': False,
    'F0_originaldev_official': False,
}
MED_SHA = {
    'med_runner': '87c187c889a8631b745bac1b05b8d6ea55b7ff9efacb0d4524142513c2e57a32',
    'med_trainer': '3daa85b011d80a371589fbd2b4cbe1f6dda28885ddd8d4d27089d283d16d965f',
    'med_interface': 'aecef7eaab0565ab98ce76c034f01b543be3342b00a84d4bae71c36d614d55be',
    'med_model_source': '690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1',
    'med_generation': '0c945640d895383814ec0662afe115c57c47fed332fff5b46b5e228b48531eaf',
}
SHARED_NAMES = {'tree_helper', 'parser', 'parser_base', 'input', 'old_entities', 'prompts',
                'old_baseline_pred', 'old_baseline_metrics', 'hpo', 'evaluator', 'validator'}
DYNAMIC_PATHS = {
    'training_manifest': 'epoch-002/run_manifest.json', 'training_complete': 'train-002/training_complete.json',
    'checkpoint': 'epoch-002/checkpoint.json', 'export_ack': 'epoch-002/export-ack.json',
    'inference_plan': 'inference-plan.json', 'responses': 'responses.jsonl',
    'response_manifest': 'responses.manifest.json', 'response_checkpoint': 'responses.checkpoint.json',
    'response_loading': 'responses.loading.json', 'response_freeze': 'responses-freeze.json',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utilities():
    require(hashlib.sha256(UTILITY.read_bytes()).hexdigest() == UTILITY_SHA, 'Frozen utility source differs')
    spec = importlib.util.spec_from_file_location('frozen_paired_utilities', UTILITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def association_span(line):
    """Locate the one top-level value without reserializing any other JSON bytes."""
    decoder, position, names, found = json.JSONDecoder(), 0, set(), None
    def whitespace(index):
        while index < len(line) and line[index].isspace():
            index += 1
        return index
    position = whitespace(position)
    require(line[position:position + 1] == '{', 'Require JSON object line')
    position += 1
    while True:
        position = whitespace(position)
        if line[position:position + 1] == '}':
            position += 1
            break
        key, position = decoder.raw_decode(line, position)
        require(isinstance(key, str) and key not in names, 'Duplicate/nonstring top-level key')
        names.add(key)
        position = whitespace(position)
        require(line[position:position + 1] == ':', 'Missing object colon')
        begin = whitespace(position + 1)
        value, end = decoder.raw_decode(line, begin)
        if key == 'association':
            require(isinstance(value, list), 'Association must be a list')
            found = (begin, end)
        position = whitespace(end)
        if line[position:position + 1] == ',':
            position += 1
            continue
        require(line[position:position + 1] == '}', 'Invalid object separator')
        position += 1
        break
    require(not line[position:].strip() and found is not None, 'Missing association/trailing JSON')
    return found


def nonrelation_bytes(lines):
    output = []
    for line in lines:
        begin, end = association_span(line)
        output.append((line[:begin] + '[]' + line[end:]).encode('utf-8'))
    return b''.join(output)


def relation_map(document):
    result = {}
    for association in document['association']:
        require(set(association) == {'patient_id', 'phenotype'}, 'Unexpected association fields')
        patient = str(association['patient_id'])
        phenotypes = association['phenotype']
        require(patient not in result and isinstance(phenotypes, list)
                and all(isinstance(value, str) for value in phenotypes)
                and len(phenotypes) == len(set(phenotypes)), 'Duplicate/invalid patient or phenotype')
        result[patient] = set(phenotypes)
    return result


def assemble_two(old_lines, med_documents):
    baseline = [json.loads(line) for line in old_lines]
    require(len(baseline) == len(med_documents), 'Document coverage differs')
    require(len({str(d['pmc_id']) for d in baseline}) == len(baseline), 'Duplicate baseline document')
    primary, union = [], []
    counts = {'old': 0, 'med': 0, 'intersection': 0, 'union': 0}
    for raw, old, med in zip(old_lines, baseline, med_documents):
        require({k: v for k, v in old.items() if k != 'association'}
                == {k: v for k, v in med.items() if k != 'association'}, 'Nonrelation fields/document order changed')
        old_map, med_map = relation_map(old), relation_map(med)
        require(set(old_map) == set(med_map), 'Patient coverage differs')
        combined = copy.deepcopy(old['association'])
        for association in combined:
            patient = str(association['patient_id'])
            left, right = old_map[patient], med_map[patient]
            association['phenotype'] = sorted(left | right)
            counts['old'] += len(left)
            counts['med'] += len(right)
            counts['intersection'] += len(left & right)
            counts['union'] += len(left | right)
        begin, end = association_span(raw)
        primary.append(raw[:begin] + json.dumps(med['association'], ensure_ascii=False) + raw[end:])
        union.append(raw[:begin] + json.dumps(combined, ensure_ascii=False) + raw[end:])
    unchanged = nonrelation_bytes(old_lines)
    require(nonrelation_bytes(primary) == nonrelation_bytes(union) == unchanged, 'Nonrelation byte drift')
    return primary, union, counts


def retain(candidate, baseline):
    values = [candidate[k] for k in ('micF1', 'macF1')] + [baseline[k] for k in ('micF1', 'macF1')]
    require(all(math.isfinite(value) for value in values), 'Nonfinite relation metric')
    return candidate['micF1'] + candidate['macF1'] > baseline['micF1'] + baseline['macF1'] and candidate['macF1'] >= baseline['macF1']


def reference_gate(plan, w):
    require(plan['schema'] == SCHEMA and plan['policy'] == POLICY, 'Predeclared endpoint choices differ')
    require(set(plan['references']) == SHARED_NAMES | set(MED_SHA), 'Public reference whitelist differs')
    require(set(plan['endpoint_references']) == set(DYNAMIC_PATHS), 'Endpoint reference whitelist differs')
    require(plan['historical_F1_gold_sha256'] == w.GOLD_SHA, 'Historical F1 gold pin differs')
    for name in SHARED_NAMES:
        require(plan['references'][name]['sha256'] == w.SHARED_SHA[name], 'Frozen old reference differs: ' + name)
    for name, digest in MED_SHA.items():
        require(plan['references'][name]['sha256'] == digest, 'Fixed Med source differs: ' + name)
    records = list(plan['references'].values()) + list(plan['endpoint_references'].values()) + [plan['pretraining_amendment']]
    gold = Path(plan['gold_path']).resolve()
    for record in records:
        require(set(record) == {'path', 'bytes', 'sha256'}, 'Require immutable three-field reference')
        path = Path(record['path']).resolve()
        require(path != gold, 'Gold cannot be a prechecked reference')
        if path.exists() and gold.exists():
            require(not os.path.samefile(path, gold), 'Gold hardlink cannot be prechecked')
    return records


def verify_endpoint(plan, w, h, prompts, pairs):
    common, refs = plan['references'], plan['endpoint_references']
    cp, run, complete = (w.read(refs[name]['path']) for name in ('checkpoint', 'training_manifest', 'training_complete'))
    fp = cp['fingerprint']
    require(cp['completed_epochs'] == 2 and cp['global_step'] == 32 and cp['smoke_only'] is False, 'Incomplete Med epoch2')
    require(fp == run['fingerprint'] and fp['schema'] == 'patientphex-medgemma15-fullbundle-text-relation-lora-v1', 'Med fingerprint differs')
    require(fp['source_sha256'] == complete['source_sha256'] == MED_SHA['med_trainer']
            and fp['interface_sha256'] == MED_SHA['med_interface']
            and fp['train_sha256'] == w.TRAIN_SHA and fp['excluded_input_sha256'] == w.EXCLUDED_SHA
            and fp['inference_input_sha256'] == common['prompts']['sha256'], 'Med source/data lineage differs')
    require(len(set(fp['training_document_ids'])) == 32 and len(set(fp['excluded_document_ids'])) == 32
            and not set(fp['training_document_ids']) & set(fp['excluded_document_ids'])
            and {str(row['pmc_id']) for row in prompts} <= set(fp['excluded_document_ids']), 'Strict32 isolation differs')
    require(run['examples'] == 124 and run['steps_per_epoch'] == 16 and run['epochs'] == 2
            and run['total_optimizer_steps'] == 32 and run['warmup_steps'] == 2
            and run['fresh_base_on_initial_run'] is True and run['validation_labels_used'] is False
            and run['smoke_only'] is False, 'Require fresh124x2 Med stage')
    require(complete['epochs'] == 2 and complete['global_step'] == 32 and complete['smoke_only'] is False
            and complete['base_parameters'] == 4300079472 and complete['trainable_adapter_parameters'] == 29802496
            and complete['total_parameters'] == 4329881968, 'Formal Med completion differs')
    adapter = {name: cp['files']['adapter/' + name] for name in ('adapter_config.json', 'adapter_model.safetensors')}
    ack = w.read(refs['export_ack']['path'])
    destination = PurePosixPath(ack['persistent_directory'])
    require(ack['status'] == 'persistent_checkpoint_verified' and ack['persistent_mount_id'] == w.MOUNT_ID
            and ack['fsync_all_files'] is True and ack['completed_epochs'] == 2 and ack['global_step'] == 32
            and destination.parts == ('research', 'medgemma-relation-control', 'formal2ep', 'checkpoints', 'epoch-002')
            and ack['checkpoint_sha256'] == refs['checkpoint']['sha256']
            and ack['run_manifest_sha256'] == refs['training_manifest']['sha256'], 'Persistent final Med receipt differs')
    required = {**cp['files'], 'checkpoint.json': refs['checkpoint']['sha256'],
                '_run/run_manifest.json': refs['training_manifest']['sha256'],
                '_run/train_medgemma32.py': MED_SHA['med_trainer'], '_run/medgemma_interface.py': MED_SHA['med_interface']}
    require({'training_state.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'} <= set(required), 'Incomplete saved Med state')
    for name, digest in required.items():
        require(ack['files'][name]['sha256'] == digest and ack['files'][name]['bytes'] > 0, 'Persistent file SHA differs: ' + name)
    infer, frozen = (w.read(refs[name]['path']) for name in ('inference_plan', 'response_freeze'))
    require(infer['checkpoint_sha256'] == refs['checkpoint']['sha256']
            and infer['training_complete_sha256'] == refs['training_complete']['sha256']
            and infer['adapter_files'] == adapter and infer['prompts_sha256'] == common['prompts']['sha256']
            and infer['runner_sha256'] == MED_SHA['med_runner']
            and infer['model_source_sha256'] == MED_SHA['med_model_source'], 'Med inference lineage differs')
    require(frozen['plan_sha256'] == refs['inference_plan']['sha256'] and frozen['rows'] == 80
            and frozen['raw_sha256'] == refs['responses']['sha256']
            and frozen['manifest_sha256'] == refs['response_manifest']['sha256'], 'Complete original80 response freeze differs')
    meta, raw = w.read(refs['response_manifest']['path']), w.rows(refs['responses']['path'])
    actual, config = meta['fingerprint'], meta['fingerprint']['configuration']
    require(actual == w.read(refs['response_checkpoint']['path']) and meta['load_information'] == w.read(refs['response_loading']['path']), 'Inference state/loading receipt differs')
    source = w.read(common['med_model_source']['path'])
    files = {value['name']: {key: value[key] for key in ('bytes', 'sha256')} for value in source['files']}
    require(len(files) == 19 and actual['model_files'] == files == fp['base_files']['actual_files']
            and fp['base_files']['all19_files_verified'] is True, 'Actual full Med asset fingerprint differs')
    require(actual['source_sha256'] == MED_SHA['med_runner'] and actual['interface_sha256'] == MED_SHA['med_interface']
            and actual['source_manifest_sha256'] == MED_SHA['med_model_source']
            and actual['prompts_sha256'] == common['prompts']['sha256'] and actual['adapter_files'] == adapter
            and config['adapter'] == infer['adapter_directory'], 'Actual Med adapter/source differs')
    require(actual['generation_configuration'] == w.read(common['med_generation']['path'])
            and actual['batch_size'] == 1 and actual['thinking'] is False
            and actual['precision'] == 'native_fullbundle_text_only_bfloat16_sdpa_no_quantization'
            and config['max_input_tokens'] == 14000 and config['max_new_tokens'] == 1600, 'Fixed native Med decoding differs')
    require(meta['base_parameters'] == 4300079472 and meta['total_parameters'] == 4329881968
            and meta['expected_rows'] == len(raw) == 80 and meta['output_sha256'] == refs['responses']['sha256']
            and meta['eos_token_ids'] == [1, 106] and meta['pad_token_id'] == 0, 'Med response coverage/counts differ')
    require(all(not meta['load_information'].get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), 'Incomplete native loading')
    require(actual['runtime']['transformers'] == '5.20.2' and actual['runtime']['peft'] == '0.21.2', 'Native runtime differs')
    for row in raw:
        require(row['removed_demonstrations'] == 0 and isinstance(row['hit_token_limit'], bool)
                and 0 < row['input_tokens'] <= 14000 and 0 <= row['output_tokens'] <= 1600, 'Changed prompt/token policy')
    _, parse = h.llm_features(prompts, raw, pairs)
    return adapter, parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    directory = parser.parse_args().directory.resolve()
    w, started = utilities(), time.monotonic()
    plan_path = directory / 'run-plan.json'
    plan = w.read(plan_path)
    require(plan['control_sha256'] == w.sha(__file__), 'Evaluation entry differs')
    records = reference_gate(plan, w)
    for record in records:
        w.checked(record)
    amendment = w.read(plan['pretraining_amendment']['path'])
    require(amendment['policy'] == POLICY and amendment['endpoint_source_sha256'] == w.sha(__file__), 'Pretraining choices not bound')
    w.save(directory / 'evaluation-reservation.json', {'at_utc': w.now(), 'plan': w.ref(plan_path), 'quality_started': False})
    common, refs = plan['references'], plan['endpoint_references']
    require(Path(common['parser_base']['path']).resolve() == Path(common['parser']['path']).resolve().with_name('run.py'), 'Parser sibling differs')
    h = w.module('med_frozen_tree', common['tree_helper']['path'])
    h.verify_parser(common['parser']['path'])
    prompts = w.rows(common['prompts']['path'])
    pairs = h.pair_rows(prompts)
    require(len(prompts) == 80 and len(pairs) == 1557 and sum(len(row['code_map']) for row in prompts) == 587, 'Original F1 input universe differs')
    adapter, parse = verify_endpoint(plan, w, h, prompts, pairs)
    parsed = directory / 'med-parsed.pred.jsonl'
    require(not parsed.exists(), 'Assembly output already exists')
    w.command([plan['python'], common['parser']['path'], '--input', common['input']['path'],
               '--entities', common['old_entities']['path'], '--prompts', common['prompts']['path'],
               '--responses', refs['responses']['path'], '--output', str(parsed), '--fallback', 'baseline'],
              directory / 'assembly-command.json')
    med = w.rows(parsed)
    require(len(med) == 16 and sum(len(d['entities']) for d in med) == 1189, 'Full Med prediction coverage differs')
    parsed_meta = w.read(parsed.with_suffix('.manifest.json'))
    require(parsed_meta['code_sha256'] == common['parser']['sha256']
            and {key: parsed_meta['stats'].get(key, 0) for key in parse} == parse, 'Original parser/statistics differ')
    old_lines = Path(common['old_baseline_pred']['path']).read_bytes().decode('utf-8').splitlines(keepends=True)
    primary, union, union_counts = assemble_two(old_lines, med)
    frozen = {}
    for name, lines in zip(ENDPOINTS, (primary, union)):
        out = directory / name
        out.mkdir(exist_ok=False)
        prediction = out / 'pred.jsonl'
        with prediction.open('xb') as handle:
            handle.write(''.join(lines).encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        frozen[name] = {'prediction': w.ref(prediction)}
    for record in records:
        w.checked(record)
    freeze_path = directory / 'paired-predictions-freeze.json'
    w.save(freeze_path, {'status': 'both_predeclared_complete_full_predictions_frozen_before_gold', 'at_utc': w.now(),
                        'plan': w.ref(plan_path), 'pretraining_amendment': plan['pretraining_amendment'],
                        'old': common['old_baseline_pred'], 'endpoints': frozen, 'med_adapter_files': adapter,
                        'parser_prediction': w.ref(parsed), 'parser_manifest': w.ref(parsed.with_suffix('.manifest.json')),
                        'parse': parse, 'union_literal_counts': union_counts, 'documents': 16, 'entities': 1189,
                        'nonrelation_bytes_exact': True,
                        'masked_nonrelation_sha256': hashlib.sha256(nonrelation_bytes(old_lines)).hexdigest()})
    freeze_reference = w.ref(freeze_path)
    w.save(directory / 'evaluation-started.json', {'at_utc': w.now(), 'freeze': freeze_reference, 'new_v2_calls': 2, 'cached_old_only': True})
    for name in ENDPOINTS:
        out = directory / name
        w.command([plan['python'], common['validator']['path'], '--input', common['input']['path'],
                   '--pred', frozen[name]['prediction']['path'], '--hpo', common['hpo']['path'],
                   '--output', str(out / 'validation.json')], out / 'validation-command.json')
        require(w.read(out / 'validation.json')['valid'], 'Invalid full prediction: ' + name)
    # First gold-byte access is after BOTH complete immutable predictions and the paired freeze.
    w.checked(freeze_reference)
    require(w.sha(plan['gold_path']) == plan['historical_F1_gold_sha256'], 'Historical F1 gold changed')
    old, metrics = w.read(common['old_baseline_metrics']['path']), {}
    for name in ENDPOINTS:
        out = directory / name
        w.checked(frozen[name]['prediction'])
        w.command([plan['python'], common['evaluator']['path'], '--gold', plan['gold_path'],
                   '--pred', frozen[name]['prediction']['path'], '--body-scope', 'all', '--no-id-doc', 'text',
                   '--output', str(out / 'metrics.json')], out / 'evaluation-command.json')
        result = w.read(out / 'metrics.json')
        require(result['documents'] == 16 and result['patients'] == 39
                and all(result['metrics'][key] == old['metrics'][key] for key in ('menF1', 'docF1')), 'F1/nonrelation metric scope changed')
        metrics[name] = result
    for record in records:
        w.checked(record)
    w.checked(freeze_reference)
    for value in frozen.values():
        w.checked(value['prediction'])
    summary = {'status': 'complete_predeclared_Med_standalone_and_union_F1_development', 'policy': POLICY,
               'scope': 'Known old baseline and repeatedly observed F1 development; no F0, originaldev, independent-generalization or official claim.',
               'paired_freeze': freeze_reference, 'old_cached_metrics': common['old_baseline_metrics'], 'old_metrics': old['metrics'],
               'endpoints': {name: {'metrics': metrics[name]['metrics'], 'counts': metrics[name]['counts'],
                                    'metrics_file': w.ref(directory / name / 'metrics.json'),
                                    'format_file': w.ref(directory / name / 'validation.json'),
                                    'delta_vs_old': w.delta(metrics[name]['metrics'], old['metrics']),
                                    'retain_vs_old_advisory': retain(metrics[name]['metrics'], old['metrics'])} for name in ENDPOINTS},
               'union_minus_med_report_only': w.delta(metrics[ENDPOINTS[1]]['metrics'], metrics[ENDPOINTS[0]]['metrics']),
               'automatic_migration': False, 'elapsed_CPU_seconds': time.monotonic() - started}
    w.save(directory / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
