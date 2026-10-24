#!/usr/bin/env python3
"""Two predeclared full-report SFT32 endpoints; no parsing/repair of model text.

The binder may enable a copy of the disabled template only after actual complete80.
Both predictions are frozen together before evaluator import or the first gold byte.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = 'patientphex-sft32-full-report-full80-F1-v1'
COMPLETE = 'COMPLETE_FIXED80_SFT32_FULL_REPORT_NO_QUALITY'
ENDPOINTS = ('diagnostic-fulltext-alone', 'primary-oldSFT32-union-fulltext')
GOLD_SHA256 = '665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3'
SHARED_SHA = {
 'input': '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415',
 'old_baseline_pred': 'ffb2c54ac4af0136fe1bb4fa08a34e14d3e0e27c4354a2b9c65aa50ea0f5f379',
 'old_baseline_metrics': '7e36dd93a436bdaadd4b0527b1bb388d6e438d8f4bcf8748d00020a3a1c4f477',
 'prompts': 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe',
 'expanded_prompts': '2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085',
 'input_audit': '35f9a00ecabf0b0114d756d31b13fc5a947155cb3c3ec8f6a36392f1f0674abb',
 'input_actual': 'f029779093c37fbf42b68274a19309298662e9644c0f9ca6ba1f304d4e8b3c38',
 'probe_plan': '1ab430d24eab017a73eb7b403e4a1da816d408caba21f8caaf9b449e2697e79a',
 'probe_raw': '8653359b5a2df2eb4473cd34681d4b2f6bf151ba4b8f3657fa5d6e01306f38fc',
 'probe_format': 'a1505a529bfd1d8a94a7d23276e8687c841059762b1b50dabb1410084a8aef3f',
 'probe_ids': '571acae79df36553cf6d813527e2a4c69036146d1b3aed366b8bd2a2e8bfa199',
 'probe_summary': '344ce7a0cd0b8676523975f735e1eec1b241b3cf820f0fbf8095af84ad7ae9d4',
 'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
 'evaluator': 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781',
 'validator': 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f',
}
STATIC_INFERENCE = {'inference_source', 'inference_launcher', 'inference_plan'}
ACTUAL_NAMES = {'responses', 'canonical', 'summary', 'complete', 'wrapper', 'gpu_release', 'actual_input'}
POLICY = {
 'diagnostic': 'Only complete natural-EOS supplied-CID/PID canonical mappings; invalid whole rows abstain.',
 'primary': 'Literal per-document per-patient old-SFT32 union fulltext-alone.',
 'canonical': 'Consume frozen runner canonical records; never reparse raw, repair answers, or generate again.',
 'literal': 'Keep original concept strings including compound HPO/NIL; unchanged v2 handles exact HP compounds.',
 'selection_comparator': 'Cached old-SFT32 metrics, zero old reevaluations.',
 'retention': 'Primary relation micF1+macF1 strictly increases AND macF1 does not decrease.',
 'evaluation': 'Freeze both complete predictions together before gold; exactly one v2 all/text each.',
 'nonrelation': 'All bytes outside each old baseline top-level association value remain unchanged.',
 'scope': 'Known F1 development only; no F0/originaldev/official/generalization claim.',
 'automatic_migration': False, 'parameter_or_threshold_search': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(record):
    require(type(record) is dict and set(record) == {'path', 'bytes', 'sha256'}, 'Require immutable three-field ref')
    path = Path(record['path'])
    require(path.is_absolute() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
            'Artifact SHA/size mismatch: ' + str(path))
    return path


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def rows(path):
    return [json.loads(line, object_pairs_hook=unique_object)
            for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def jsha(item):
    return hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load frozen source')
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


HELPERS_SHA = '0fa4ef7be27fbde82ed8e22020cbcef05fdfc396a5dae0d1711674bed9efb828'
helper_path = Path(__file__).resolve().with_name('literal_helpers.py')
require(sha(helper_path) == HELPERS_SHA, 'Frozen generic helper source differs')
helpers = module('sft32_full_report_literal_helpers', helper_path)
association_span = helpers.association_span
nonrelation_bytes = helpers.nonrelation_bytes
relation_map = helpers.relation_map
assemble_two = helpers.assemble_two
retain = helpers.retain


def canonical_gate(prompt, raw, item):
    """Validate supplied canonical identities, without parsing the raw response."""
    require(raw['row_id'] == item['row_id'] == prompt['row_id'] and
            raw['pmc_id'] == item['pmc_id'] == prompt['pmc_id'], 'Row/document identity differs')
    require(type(raw['response']) is str and type(raw['natural_eos']) is bool and
            type(raw['hit_token_limit']) is bool, 'Invalid raw text/EOS/cap type')
    require(type(raw['output_tokens']) is int and 0 < raw['output_tokens'] <= 1600 and
            type(raw['input_tokens']) is int and 0 < raw['input_tokens'] <= 14400,
            'Actual token count outside frozen cap')
    require(raw['hit_token_limit'] == (raw['output_tokens'] >= 1600 and not raw['natural_eos']),
            'Cap bookkeeping differs from native policy')
    require(type(raw['removed_demonstrations']) is int and raw['removed_demonstrations'] == 0 and
            isinstance(raw['batch_seconds'], (int, float)) and math.isfinite(raw['batch_seconds']) and
            raw['batch_seconds'] >= 0, 'Prompt deletion or invalid elapsed time')
    require(type(item['strict_complete']) is bool and type(item['candidate_patients']) is dict,
            'Canonical status/mapping type differs')
    if not item['strict_complete']:
        require(item['candidate_patients'] == {} and type(item['abstention']) is str and item['abstention'],
                'Invalid row must wholly abstain; no negative fill')
        return None
    require(raw['natural_eos'] and not raw['hit_token_limit'] and item['abstention'] is None,
            'Complete mapping lacks natural EOS or has stale abstention')
    mapping = item['candidate_patients']
    require(set(mapping) == set(prompt['code_map']), 'Complete canonical must cover every supplied CID exactly')
    for values in mapping.values():
        require(type(values) is list and all(type(pid) is str and pid in prompt['patient_ids'] for pid in values)
                and len(values) == len(set(values)), 'Unknown/duplicate/nonliteral PID')
    return mapping


def make_primary(old_lines, inputs, prompts, evidence):
    require(len(prompts) == len(evidence) and len({p['row_id'] for p in prompts}) == len(prompts),
            'Incomplete or duplicate row coverage')
    baseline = [json.loads(line, object_pairs_hook=unique_object) for line in old_lines]
    require([str(d['pmc_id']) for d in baseline] == [str(d['pmc_id']) for d in inputs] and
            len(inputs) == len({str(d['pmc_id']) for d in inputs}), 'Baseline/input coverage or order differs')
    allowed, seen, covered, relations, stats = {}, set(), set(), defaultdict(lambda: defaultdict(set)), Counter()
    for doc in inputs:
        require('entities' not in doc and 'association' not in doc, 'Input contains labels')
        patients = [p['patient_id'] for p in doc['patient']]
        require(all(type(pid) is str for pid in patients) and len(patients) == len(set(patients)),
                'Ambiguous original patient IDs')
        allowed[str(doc['pmc_id'])] = patients
    for prompt, mapping in zip(prompts, evidence):
        pmc = prompt['pmc_id']; covered.add(pmc)
        require(pmc in allowed and len(prompt['patient_ids']) == len(set(prompt['patient_ids'])) and
                set(prompt['patient_ids']) == set(allowed[pmc]), 'Prompt/input patient identity differs')
        require(list(prompt['code_map']) == ['C%03d' % i for i in range(len(prompt['code_map']))],
                'Duplicate, missing or nonsequential CID')
        for concept in prompt['code_map'].values():
            require(type(concept) is str and (pmc, concept) not in seen, 'Duplicate/nonliteral document concept')
            seen.add((pmc, concept))
        if mapping is None:
            stats['whole_row_abstentions'] += 1
            stats['abstained_CIDs'] += len(prompt['code_map'])
            continue
        require(set(mapping) == set(prompt['code_map']), 'Canonical mapping incomplete')
        stats['strict_complete_rows'] += 1
        stats['provided_CIDs'] += len(mapping)
        for cid, patients in mapping.items():
            require(all(type(pid) is str and pid in allowed[pmc] for pid in patients), 'Unknown patient')
            stats['explicit_empty_CIDs'] += not patients
            for pid in patients:
                relations[pmc][pid].add(prompt['code_map'][cid])
    require(covered == set(allowed), 'Prompt document coverage differs')
    output = []
    for old in baseline:
        pmc = str(old['pmc_id'])
        require(all(type(a['patient_id']) is str for a in old['association']) and
                set(relation_map(old)) == set(allowed[pmc]), 'Baseline patient coverage differs')
        document = copy.deepcopy(old)
        document['association'] = [{'patient_id': a['patient_id'],
            'phenotype': sorted(relations[pmc][a['patient_id']])} for a in old['association']]
        output.append(document)
    return output, dict(stats)


def reference_gate(plan, require_ready=True):
    # This check is deliberately before any reference stat/hash/open (including gold).
    require(plan.get('execution_ready') is (True if require_ready else False) and
            plan.get('schema') == SCHEMA and plan.get('policy') == POLICY, 'Disabled/incompatible evaluation plan')
    require(plan['control_sha256'] == sha(__file__) and plan['helpers_sha256'] == HELPERS_SHA and
            plan['historical_F1_gold_sha256'] == GOLD_SHA256, 'Evaluation source/gold metadata differs')
    common, inference, actual = plan['references'], plan['inference_references'], plan['actual_references']
    require(set(common) == set(SHARED_SHA) and set(inference) == STATIC_INFERENCE and
            set(actual) == ACTUAL_NAMES, 'Reference whitelist differs')
    records = list(common.values()) + list(inference.values())
    if require_ready:
        records += list(actual.values())
    else:
        require(all(value is None for value in actual.values()), 'Preparation contains unbound actual results')
    gold_path = Path(plan['gold_path']).resolve()
    require(str(gold_path).endswith('/round2-inner4/fold-1/validation16.gold.jsonl'), 'Wrong F1 gold metadata path')
    for record in records:
        require(Path(record['path']).resolve() != gold_path and record['sha256'] != GOLD_SHA256,
                'Gold cannot enter prequality references')
        checked(record)
    for name, digest in SHARED_SHA.items():
        require(common[name]['sha256'] == digest, 'Frozen common identity differs: ' + name)
    return common, inference, actual, records


def verify_inference(common, inference, actual, terminal_dir):
    """Actual whole80 gate; pure CPU and no label/quality reads."""
    fixed = read(inference['inference_plan']['path']); probe = read(common['probe_plan']['path'])
    require(fixed['cap_seconds'] == 1200 and fixed['term_grace_seconds'] == 30 and
            fixed['input_limit'] == 14400 and fixed['generation_limit'] == 1600 and
            fixed['cached_original_indices'] == [19], 'Frozen runtime caps/cache differ')
    for key in ('model', 'adapter', 'adapter_files', 'generation', 'gpu_uuid', 'python'):
        require(fixed[key] == probe[key], 'Probe/runtime configuration drift: ' + key)
    local_plan_names = {'infer_full80.py': inference['inference_source'], 'launch.py': inference['inference_launcher'],
        'expanded80.jsonl': common['expanded_prompts'], 'row-audit.json': common['input_audit'],
        'cache-probe-raw.json': common['probe_raw'], 'cache-probe-ids.json': common['probe_ids'],
        'cache-probe-format.json': common['probe_format'], 'cache-probe-summary.json': common['probe_summary']}
    for name, record in local_plan_names.items():
        require(fixed['files'][name]['sha256'] == record['sha256'] and fixed['files'][name]['bytes'] == record['bytes'],
                'Runtime deployment differs: ' + name)
    summary = read(actual['summary']['path']); complete = read(actual['complete']['path'])
    require(summary == complete and summary['status'] == COMPLETE and summary['expected_rows'] == 80 and
            summary['fresh_rows'] == 79 and summary['cached_rows'] == 1 and summary['cached_original_indices'] == [19],
            'Inference not the unique complete fixed80 run')
    require(summary['quality_calls'] == 0 and summary['gold_read'] is False and
            summary['raw_ids_saved_before_decode'] is True and summary['raw_response_saved_before_parse'] is True,
            'Inference quality/raw-first policy differs')
    for key, source in [('responses_sha256', actual['responses']), ('canonical_sha256', actual['canonical']),
                        ('plan_sha256', inference['inference_plan']), ('source_sha256', inference['inference_source'])]:
        require(summary[key] == source['sha256'], 'Actual completion identity differs: ' + key)
    require(summary['base_parameters'] == 8953803264 and summary['total_parameters'] == 8997081600,
            'Actual model/adapter parameter inventory differs')
    wrapper, release = read(actual['wrapper']['path']), read(actual['gpu_release']['path'])
    require(wrapper['returncode'] == 0 and wrapper['cap_seconds'] == 1200 and wrapper['term_grace_seconds'] == 30 and
            wrapper['plan_sha256'] == inference['inference_plan']['sha256'] and
            wrapper['source_sha256'] == inference['inference_launcher']['sha256'], 'Actual wrapper incomplete/different')
    require(release['target_compute_rows'] == [] and len(release['target_csv']) == 1 and
            release['target_csv'][0].split(',')[1].strip() == fixed['gpu_uuid'], 'GPU release not verified for owned UUID')
    asset = read(actual['actual_input']['path'])
    require(asset['rows'] == 80 and asset['documents'] == 16 and asset['candidate_codes'] == 587 and
            asset['input_tokens'] == 670122 and asset['max_input_tokens'] == 14390 and
            asset['all80_actual_ids_equal_prior_native_CPU'] is True and asset['no_truncation'] is True and
            asset['gold_read'] is False and asset['max_position_embeddings'] >= 15990,
            'Actual input/context/asset receipt differs')
    require({key: value['sha256'] for key, value in asset['adapter_files'].items()} == fixed['adapter_files'],
            'Actual adapter bytes differ')
    require({key: asset[key] for key in fixed['cached_probe_runtime']} == fixed['cached_probe_runtime'],
            'Actual library/tokenizer/TF32 runtime differs from cached probe')
    originals = rows(common['prompts']['path']); expanded = rows(common['expanded_prompts']['path'])
    audit = read(common['input_audit']['path']); raw = rows(actual['responses']['path']); canonical = rows(actual['canonical']['path'])
    require(len(originals) == len(expanded) == len(audit) == len(raw) == len(canonical) == 80,
            'Incomplete full80 files')
    require(len({p['row_id'] for p in originals}) == 80 and len({p['pmc_id'] for p in originals}) == 16 and
            sum(len(p['code_map']) for p in originals) == 587 and
            sum(len(p['code_map']) * len(p['patient_ids']) for p in originals) == 1557, 'Original F1 universe differs')
    expected_order = sorted(range(80), key=lambda i: (audit[i]['expanded']['tokens'], originals[i]['row_id']))
    require([r['row_id'] for r in raw] == [originals[i]['row_id'] for i in expected_order] and
            [r['row_id'] for r in canonical] == [r['row_id'] for r in raw], 'Missing/duplicate/wrong-order actual rows')
    probe_eos = read(common['probe_ids']['path'])['configured_eos_ids']
    by_raw = {r['row_id']: r for r in raw}; by_canonical = {r['row_id']: r for r in canonical}
    evidence, shard_refs = [], []
    terminal = Path(terminal_dir).resolve()
    require(Path(actual['responses']['path']).parent.resolve() == terminal, 'Wrong terminal root')
    for index, (prompt, exp, cpu) in enumerate(zip(originals, expanded, audit)):
        require(cpu['index'] == index and cpu['row_id'] == prompt['row_id'] == exp['row_id'] and
                cpu['pmc_id'] == prompt['pmc_id'] == exp['pmc_id'] and jsha(prompt) == cpu['original_row_sha256'] and
                jsha(exp) == cpu['expanded_row_sha256'], 'Original/expanded CPU identity differs')
        require({k: v for k, v in prompt.items() if k != 'messages'} ==
                {k: v for k, v in exp.items() if k != 'messages'}, 'Expanded nonmessage fields drift')
        r, c = by_raw[prompt['row_id']], by_canonical[prompt['row_id']]
        require(r['input_tokens'] == cpu['expanded']['tokens'] and
                c['provenance'] == ('cached_probe' if index == 19 else 'fresh'), 'Actual input/provenance drift')
        evidence.append(canonical_gate(prompt, r, c))
        records = [ref(terminal / folder / ('%03d.json' % index))
                   for folder in ('raw-responses', 'canonical-records', 'generated-ids')]
        shard_refs.extend(records)
        require(read(records[0]['path']) == r and read(records[1]['path']) == c, 'Raw/canonical shard differs from aggregate')
        ids = read(records[2]['path']); values = ids['generated_ids']; eos = ids['configured_eos_ids']
        require(ids['row_id'] == prompt['row_id'] and ids['input_ids_sha256'] == cpu['expanded']['input_ids_sha256'] and
                type(values) is list and all(type(v) is int and v >= 0 for v in values) and
                len(values) == r['output_tokens'] and eos == probe_eos and
                r['natural_eos'] == ids['natural_eos'] == bool(values and values[-1] in eos) and
                r['hit_token_limit'] == ids['hit_token_limit'], 'Saved generated-ID proof differs')
        if index == 19:
            require(r == read(common['probe_raw']['path']) and records[0]['sha256'] == common['probe_raw']['sha256'] and
                    records[2]['sha256'] == common['probe_ids']['sha256'] and
                    c['candidate_patients'] == read(common['probe_format']['path'])['parsed'], 'Cached probe bytes/canonical drift')
    valid = sum(x is not None for x in evidence)
    require(summary['strict_complete_rows'] == valid and summary['whole_row_abstentions'] == 80 - valid,
            'Summary canonical counts differ')
    coverage = {'rows': 80, 'fresh_rows': 79, 'cached_rows': 1, 'cached_original_indices': [19],
        'strict_complete_rows': valid, 'whole_row_abstentions': 80 - valid,
        'cap_rows': sum(r['hit_token_limit'] for r in raw), 'input_tokens': sum(r['input_tokens'] for r in raw),
        'raw_text_reparsed': False, 'missing_CIDs_filled': False, 'raw_shards_verified': 240}
    require(coverage['input_tokens'] == 670122, 'Actual total input tokens differ')
    return originals, evidence, coverage, shard_refs


def freeze_pair(directory, plan_ref, plan, common, actual, old_lines, documents, coverage, parse_stats, shard_refs):
    """This pure assembly boundary has no evaluator import or gold access."""
    alone, union, counts = assemble_two(old_lines, documents); counts['fulltext'] = counts.pop('med')
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for name, lines in zip(ENDPOINTS, (alone, union)):
        target = directory / name; target.mkdir()
        path = target / 'pred.jsonl'
        with path.open('xb') as stream:
            stream.write(''.join(lines).encode('utf-8')); stream.flush(); os.fsync(stream.fileno())
        frozen[name] = {'prediction': ref(path)}
    for record in list(common.values()) + list(plan['inference_references'].values()) + list(actual.values()) + shard_refs:
        checked(record)
    checked(plan_ref)
    freeze_path = directory / 'paired-predictions-freeze.json'
    save(freeze_path, {'status': 'both_predeclared_complete_full_predictions_frozen_before_gold',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'plan': plan_ref, 'endpoints': frozen,
        'actual_references': actual, 'inference_references': plan['inference_references'],
        'old_baseline': common['old_baseline_pred'], 'raw_shards': shard_refs,
        'coverage': coverage, 'parse': parse_stats, 'union_literal_counts': counts,
        'documents': len(documents), 'entities': sum(len(d['entities']) for d in documents),
        'nonrelation_bytes_exact': True, 'masked_nonrelation_sha256': hashlib.sha256(nonrelation_bytes(old_lines)).hexdigest(),
        'gold_read': False, 'new_quality_calls': 0})
    for value in frozen.values():
        checked(value['prediction'])
    return frozen, ref(freeze_path)


def delta(current, baseline):
    return {key: current[key] - baseline[key] for key in ('micF1', 'macF1', 'menF1', 'docF1', 'Score')}


def run(plan_path):
    started = time.monotonic(); plan_path = Path(plan_path).resolve(); plan_ref = ref(plan_path); plan = read(plan_path)
    common, inference, actual, records = reference_gate(plan)
    prompts, evidence, coverage, shard_refs = verify_inference(common, inference, actual, plan['terminal_dir'])
    inputs = rows(common['input']['path'])
    old_lines = Path(common['old_baseline_pred']['path']).read_bytes().decode('utf-8').splitlines(keepends=True)
    documents, parse_stats = make_primary(old_lines, inputs, prompts, evidence)
    require(len(documents) == 16 and sum(len(d['entities']) for d in documents) == 1189 and
            sum(len(d['association']) for d in documents) == 39 and
            parse_stats.get('provided_CIDs', 0) + parse_stats.get('abstained_CIDs', 0) == 587,
            'Full document/entity/patient/CID coverage differs')
    directory = Path(plan['output_dir']).resolve()
    frozen, freeze_ref = freeze_pair(directory, plan_ref, plan, common, actual, old_lines, documents,
                                    coverage, parse_stats, shard_refs)
    checked(freeze_ref)
    # Exclusive global marker prevents a second quality attempt even using another output path.
    save(Path(plan['terminal_dir']) / 'evaluation-started-full-report-v1.json',
         {'freeze': freeze_ref, 'fixed_new_v2_calls': 2, 'old_v2_calls': 0})
    save(directory / 'evaluation-started.json', {'freeze': freeze_ref, 'fixed_new_v2_calls': 2, 'old_v2_calls': 0})
    # H50: canonical validator registration must precede the evaluator's sibling import.
    # Both modules and the first gold hash/read are strictly after the common freeze.
    if 'validate_predictions' in sys.modules:
        require(sha(sys.modules['validate_predictions'].__file__) == SHARED_SHA['validator'], 'Conflicting validator module')
    validator = module('validate_predictions', common['validator']['path'])
    evaluator = module('sft32_full_report_fixed_evaluator', common['evaluator']['path'])
    require(evaluator.VERSION == 'patientphex-public-approx-v2', 'Public evaluator version differs')
    hpo = validator.read_hpo(common['hpo']['path']); predictions = {}
    for name in ENDPOINTS:
        predictions[name] = rows(checked(frozen[name]['prediction']))
        validation = validator.validate_predictions(inputs, predictions[name], hpo)
        save(directory / name / 'validation.json', validation)
        require(validation['valid'], 'Invalid full prediction: ' + name)
    require(sha(plan['gold_path']) == GOLD_SHA256, 'Historical F1 gold identity differs')
    truth = rows(plan['gold_path'])
    require(len(truth) == 16 and [str(d['pmc_id']) for d in truth] == [str(d['pmc_id']) for d in inputs],
            'Historical F1 document order differs')
    old = read(common['old_baseline_metrics']['path']); metrics = {}
    require(old['metrics']['Score'] == 0.6609420026531936, 'Cached baseline score differs')
    for name in ENDPOINTS:
        checked(frozen[name]['prediction'])
        metrics[name] = evaluator.evaluate_report(truth, predictions[name], 'all', 'text')
        require(metrics[name]['documents'] == 16 and metrics[name]['patients'] == 39 and
                all(metrics[name]['metrics'][key] == old['metrics'][key] for key in ('menF1', 'docF1')),
                'Nonrelation metrics or F1 scope drift')
        save(directory / name / 'metrics.json', metrics[name])
    for record in records + shard_refs:
        checked(record)
    checked(plan_ref); checked(freeze_ref)
    for value in frozen.values():
        checked(value['prediction'])
    summary = {'status': 'complete_predeclared_SFT32_full_report_pair_F1_development', 'policy': POLICY,
        'paired_freeze': freeze_ref, 'old_cached_metrics': common['old_baseline_metrics'], 'old_metrics': old['metrics'],
        'endpoints': {name: {'metrics': metrics[name]['metrics'], 'counts': metrics[name]['counts'],
            'metrics_file': ref(directory / name / 'metrics.json'), 'format_file': ref(directory / name / 'validation.json'),
            'delta_vs_old': delta(metrics[name]['metrics'], old['metrics'])} for name in ENDPOINTS},
        'primary_retain_vs_old': retain(metrics[ENDPOINTS[1]]['metrics'], old['metrics']),
        'diagnostic_is_not_selection_endpoint': True, 'new_v2_calls': 2, 'old_v2_calls': 0,
        'automatic_migration': False, 'elapsed_CPU_seconds': time.monotonic() - started}
    save(directory / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'summary': ref(directory / 'summary.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
