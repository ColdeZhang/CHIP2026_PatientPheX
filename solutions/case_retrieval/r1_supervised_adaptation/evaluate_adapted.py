"""Fixed final-adapter F1 standalone/union, jointly frozen before two v2 calls."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

LEGACY_SHA = '4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522'
QUALITY_SHA = '7699fa7467e81bc17645982bd45421c85ec8f5757d3afb561a9d7db25ce4208b'
ADDENDUM_SHA = 'fcea128888e8a7761bd6db32931f3f84b855af6e54230871d11651cf1eeaa22b'
ENDPOINTS = ('primary-r1-sft', 'diagnostic-oldSFT32-union-r1-sft')
FINAL_ADAPTER = '20f15b26f33219588eca7868fea13ba90dd505e0be92bc545d659b8362bbe2db'
FINAL_CHECKPOINT = '2fbd451798ec24feb780b4cf2d5dc90237ead0a6397c64e61bc00fc903bf9e46'
FINAL_STATE = 'f49f32af8b40004e38bdfc1debdc27958affd785bd32639e4acbbcffa78fccbd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(record):
    assert set(record) == {'path', 'bytes', 'sha256'}
    path = Path(record['path'])
    assert path.is_absolute() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    return path


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_legacy(record):
    path = checked(record); assert record['sha256'] == LEGACY_SHA
    specification = importlib.util.spec_from_file_location('frozen_r1_endpoint_helpers', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_recovery(root, terminal, w):
    """Check original bytes only; no model, native tokenizer, decoder or score rerun."""
    assert terminal['exit_code'] == 0 and not terminal['timeout_triggered']
    assert terminal['target_UUID_has_no_compute_process']
    for name, expected in terminal['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        actual = ref(root / relative)
        assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256']
    return {name: ref(root / name) for name in terminal['files']}


def verify_inference(plan, w, prompts, audit):
    refs = plan['endpoint_references']
    for record in refs.values():
        checked(record)
    root = Path(refs['inference_result']['path']).resolve().parent.parent
    infer, result, terminal = (read(refs[name]['path']) for name in ('inference_plan', 'inference_result', 'terminal'))
    assert result['status'] == 'COMPLETE_FIXED80_FINAL_ADAPTER_79FRESH_1CACHE_NO_QUALITY'
    assert result['new_generations'] == 79 and result['cached_generations'] == 1 and result['completed_rows'] == 80
    assert result['quality_calls'] == 0
    assert infer['budget_seconds'] == 900 and infer['grace_seconds'] == 30
    assert infer['seed'] == 20261024 and infer['max_new_tokens'] == 4096
    assert (infer['temperature'], infer['top_p'], infer['top_k']) == (.6, .95, 0)
    assert infer['new_indices'] == [i for i in range(80) if i != 73]
    assert result['cached_indices'] == [73] and result['new_indices'] == infer['new_indices']
    for key, digest in [('adapter', FINAL_ADAPTER), ('checkpoint', FINAL_CHECKPOINT), ('training_state', FINAL_STATE)]:
        assert result['adapter_binding'][key]['sha256'] == infer['adapter_binding'][key]['sha256'] == digest
    assert result['adapter_verification']['tensors'] == 504 and result['adapter_verification']['saved_actual_FP32_shapes_values_exact']
    assert result['actual_rotary']['native_tolerance_pass'] and result['actual_rotary']['scaling'] == 1.0
    assert result['all80_published_native_generic_chat_IDs_and_roundtrip_exact']
    for actual, recorded in [(refs['inference_plan'], result['plan']), (refs['inference_source'], result['source']),
                             (refs['inference_plan'], terminal['plan']), (refs['launcher'], terminal['source'])]:
        w.same_bytes(actual, recorded)
    files = validate_recovery(root, terminal, w)
    assert terminal['budget_seconds'] == 900 and terminal['grace_seconds'] == 30
    assert terminal['release']['gpu'].split(',')[1].strip() == infer['gpu_uuid']
    assert not any(line.split(',')[0].strip() == infer['gpu_uuid']
                   for line in terminal['release']['compute_processes'].splitlines())
    for name, expected in infer['sources'].items():
        assert {key: files[name][key] for key in ('bytes', 'sha256')} == expected
    response_path = checked(refs['responses'])
    data = gzip.decompress(response_path.read_bytes()) if response_path.suffix == '.gz' else response_path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == result['responses_uncompressed_sha256']
    actual = [json.loads(line) for line in data.splitlines()]
    assert len(actual) == len(prompts) == len(audit) == len(result['row_sources']) == 80
    assert len({row['row_id'] for row in actual}) == 80
    codec = w.load_normalizer(plan['references']['normalized_final']['path'], plan['references']['strict_final']['path'])
    evidence = []
    cached = read(refs['cached_row73']['path'])
    assert refs['cached_row73']['sha256'] == '00e2dcf183487482cf37e32f6da0fbb58c6d0c950afc20ad19f4d61d7d631f88'
    for i, (prompt, item, cpu) in enumerate(zip(prompts, actual, audit)):
        assert item['original_index'] == cpu['index'] == i
        assert item['row_id'] == prompt['row_id'] == cpu['row_id']
        assert item['input_tokens'] == cpu['native_input_tokens']
        assert item['input_ids_sha256'] == cpu['native_ids_sha256'] and item['quality_evaluated'] is False
        assert hashlib.sha256(json.dumps(prompt, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == cpu['original_row_sha256']
        original = read(root / 'actual' / ('row%d.json' % i))
        ids = read(root / 'actual' / ('row%d.ids.json' % i))
        raw = read(root / 'actual' / ('row%d.raw.json' % i))
        source = result['row_sources'][i]
        assert source['original_index'] == i and source['row_id'] == item['row_id']
        for field, suffix in [('row', '.json'), ('ids', '.ids.json'), ('raw', '.raw.json')]:
            w.same_bytes(ref(root / 'actual' / ('row%d' % i + suffix)), source[field])
        if i == 73:
            assert original == cached and item['fresh_NN'] is False and item['original_probe_fresh_NN'] is True
            assert all(item[k] == value for k, value in cached.items() if k != 'fresh_NN')
        else:
            assert item == original and item['fresh_NN'] is True and item['seed'] == 20261024
            assert source['kind'] == 'fresh_final_adapter_generation'
        generated = ids['generated_ids']
        assert item['generated_tokens'] == len(generated) <= 4096
        assert item['native_EOS_observed'] == bool(generated and generated[-1] == 151645)
        assert not item['stopped_at_wall_budget']
        assert item['stopped_at_token_cap'] == (not item['native_EOS_observed'] and len(generated) >= 4096)
        w.same_bytes(ref(root / 'actual' / ('row%d.ids.json' % i)), original['ids_reference'])
        w.same_bytes(ref(root / 'actual' / ('row%d.raw.json' % i)), original['raw_reference'])
        assert raw['raw_generated_text'] == item['raw_generated_text']
        parsed = w.parse_evidence(prompt, item, codec)
        coverage = item['coverage']
        assert coverage['missing_CIDs_are_negative'] is False and coverage['fallback_used'] is False
        assert coverage['invalid_whole_row_abstention'] is (parsed is None)
        assert coverage['provided_CIDs'] == (list(parsed['provided_mapping']) if parsed else [])
        assert coverage['abstained_CIDs'] == (parsed['abstained_CIDs'] if parsed else list(prompt['code_map']))
        evidence.append(parsed)
    assert sum(row['native_input_tokens'] for row in audit) == 249937
    coverage = dict(rows=80, fresh=79, cached=1,
        normalized_valid_rows=sum(item is not None for item in evidence),
        strict_valid_rows=sum(row['strict_final_valid'] for row in actual),
        natural_EOS_rows=sum(row['native_EOS_observed'] for row in actual),
        token_cap_rows=sum(row['stopped_at_token_cap'] for row in actual),
        invalid_whole_row_abstentions=sum(item is None for item in evidence),
        provided_CIDs=sum(len(item['provided_mapping']) for item in evidence if item is not None),
        total_CIDs=587, missing_CIDs_are_negative=False)
    for name, value in coverage.items():
        if name not in ('rows', 'fresh', 'cached'):
            assert result[name] == value
    w.same_bytes(refs['responses'], result['responses_gzip'])
    return evidence, coverage, files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); plan_ref = ref(args.plan); plan = read(args.plan)
    assert plan['source_sha256'] == sha(__file__) and plan['execution_ready']
    assert plan['endpoints'] == list(ENDPOINTS) and plan['new_v2_calls'] == 2 and plan['old_v2_calls'] == 0
    assert plan['quality_contract']['sha256'] == QUALITY_SHA and plan['quality_addendum']['sha256'] == ADDENDUM_SHA
    quality = read(checked(plan['quality_contract'])); checked(plan['quality_addendum'])
    w = load_legacy(plan['legacy_helpers'])
    assert plan['historical_F1_gold_sha256'] == w.GOLD_SHA256
    records = [plan['quality_contract'], plan['quality_addendum'], plan['legacy_helpers']]
    common = plan['references']; assert set(common) == set(w.SHARED_SHA)
    for name, record in common.items():
        assert record['sha256'] == w.SHARED_SHA[name]
        assert Path(record['path']).resolve() != Path(plan['gold_path']).resolve()
        checked(record); records.append(record)
    for benchmark in quality['benchmarks'].values():
        record = {key: benchmark[key] for key in ('path', 'bytes', 'sha256')}
        assert read(checked(record))['metrics'] == benchmark['metrics']; records.append(record)
    prompts = w.rows(common['prompts']['path']); inputs = w.rows(common['input']['path']); audit = read(common['input_audit']['path'])
    assert len(prompts) == 80 and len(inputs) == 16 and sum(len(r['code_map']) for r in prompts) == 587
    assert sum(len(r['code_map']) * len(r['patient_ids']) for r in prompts) == 1557
    evidence, coverage, actual_files = verify_inference(plan, w, prompts, audit)
    old_lines = Path(common['old_baseline_pred']['path']).read_bytes().decode().splitlines(keepends=True)
    primary_docs, parse_stats = w.make_primary(old_lines, inputs, prompts, evidence)
    assert len(primary_docs) == 16 and sum(len(row['entities']) for row in primary_docs) == 1189
    assert sum(parse_stats.get(k, 0) for k in ('provided_CIDs', 'abstained_invalid_row_CIDs', 'abstained_missing_CIDs')) == 587
    primary, union, counts = w.assemble_two(old_lines, primary_docs); counts['new_R1_SFT'] = counts.pop('med')
    directory = Path(plan['output_dir']); directory.mkdir(parents=True, exist_ok=False); fsync_directory(directory.parent)
    frozen = {}
    for name, lines in zip(ENDPOINTS, (primary, union)):
        target = directory / name; target.mkdir()
        with (target / 'pred.jsonl').open('xb') as handle:
            handle.write(''.join(lines).encode()); handle.flush(); os.fsync(handle.fileno())
        fsync_directory(target); frozen[name] = ref(target / 'pred.jsonl')
    for record in records + list(plan['endpoint_references'].values()): checked(record)
    assert ref(args.plan) == plan_ref
    freeze_path = directory / 'paired-predictions-freeze.json'
    save(freeze_path, dict(status='BOTH_COMPLETE_FIXED_PREDICTIONS_BEFORE_FIRST_GOLD_BYTE', at_utc=datetime.now(timezone.utc).isoformat(),
         plan=plan_ref, endpoints=frozen, coverage=coverage, parse=parse_stats, union_literal_counts=counts,
         documents=16, patients=39, entities=1189, nonassociation_bytes_exact=True, actual_source_files=actual_files,
         gold_read=False, quality_calls=0))
    fsync_directory(directory); freeze = ref(freeze_path)
    save(directory / 'evaluation-started.json', dict(freeze=freeze, new_v2_calls=2, old_v2_calls=0))
    # The import and first gold byte both follow complete joint prediction freeze.
    validator = w.module('validate_predictions', common['validator']['path'])
    evaluator = w.module('adapted_r1_fixed_v2', common['evaluator']['path'])
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    hpo = validator.read_hpo(common['hpo']['path']); predictions = {}
    for name in ENDPOINTS:
        predictions[name] = w.rows(checked(frozen[name]))
        report = validator.validate_predictions(inputs, predictions[name], hpo)
        save(directory / name / 'validation.json', report); assert report['valid']
    assert sha(plan['gold_path']) == w.GOLD_SHA256
    gold = w.rows(plan['gold_path']); assert [str(d['pmc_id']) for d in gold] == [str(d['pmc_id']) for d in inputs]
    strong = quality['benchmarks']['same_pool_strong_Qwen_SFT32_fulltext']['metrics']
    endpoints = {}
    for name, previous in zip(ENDPOINTS, ('R1_before_primary', 'R1_before_diagnostic_union')):
        checked(frozen[name]); metric = evaluator.evaluate_report(gold, predictions[name], 'all', 'text')
        assert metric['documents'] == 16 and metric['patients'] == 39
        assert all(metric['metrics'][k] == strong[k] for k in ('menF1', 'docF1'))
        assert all(math.isfinite(value) for value in metric['metrics'].values())
        save(directory / name / 'metrics.json', metric)
        endpoints[name] = dict(metrics=metric['metrics'], counts=metric['counts'], prediction=frozen[name],
            metrics_file=ref(directory / name / 'metrics.json'),
            delta_vs_same_identity_pre_adaptation=w.delta(metric['metrics'], quality['benchmarks'][previous]['metrics']),
            delta_vs_strong_fulltext=w.delta(metric['metrics'], strong),
            keep=metric['metrics']['Score'] > strong['Score'])
    passing = [name for name in ENDPOINTS if endpoints[name]['keep']]
    selected = max(passing, key=lambda name: endpoints[name]['metrics']['Score']) if passing else None
    for record in records + list(frozen.values()): checked(record)
    checked(freeze); assert ref(args.plan) == plan_ref
    summary = dict(status='COMPLETE_FIXED_R1_SFT_F1_DEVELOPMENT_PAIR', endpoints=endpoints, selected=selected,
        keep=bool(passing), comparator=strong, retention='Score strictly greater only; all components reported; identity never relabelled.',
        prospective_addendum=plan['quality_addendum'], paired_freeze=freeze, new_v2_calls=2, old_v2_calls=0,
        no_F0_originaldev_official=True, elapsed_CPU_seconds=time.monotonic() - started)
    save(directory / 'summary.json', summary); fsync_directory(directory)
    print(json.dumps(dict(summary=ref(directory / 'summary.json'), keep=summary['keep'], selected=selected)))


if __name__ == '__main__':
    main()
