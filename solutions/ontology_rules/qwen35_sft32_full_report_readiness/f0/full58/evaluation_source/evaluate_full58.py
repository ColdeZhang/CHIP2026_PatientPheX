#!/usr/bin/env python3
"""Two predeclared full-report SFT48 endpoints; no parsing/repair of model text.

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
SCHEMA = 'patientphex-sft48-full-report-full58-F0-v1'
COMPLETE = 'COMPLETE_FIXED58_SFT48_FULL_REPORT_NO_QUALITY'
ENDPOINTS = ('diagnostic-fulltext-alone', 'primary-oldSFT48-union-fulltext')
GOLD_SHA256 = '630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48'
SHARED_SHA = {'input': 'ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37', 'old_baseline_pred': '63aefe68b07664bd1e64d31e0008d05259020c727e836fbd709a5837e8929edb', 'old_baseline_metrics': 'e16ff5b0d593ca431ef20ca68afc1ddf8051374da077428553fcd436d35b8218', 'prompts': 'd387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0', 'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b', 'evaluator': 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781', 'validator': 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f', 'expanded_prompts': 'd633225256dbf1cd5b64e94c74d7ce7f220b834cd748158082b306f12191fab2', 'input_audit': '2fa5c495f8341ba4bf5f6ea4f731f75128f159f28ecb6302abaf7ead67872d02', 'input_actual': 'ccdcd461fdb8dc7119bc0755ed7c253bcee64c5ae7738c0c53a0259a30517116', 'probe_plan': 'b21fb7f92e1621926ac51eae2497f6b0c519df155081dbf744090ca7e20d1734'}
DYNAMIC_COMMON = {"probe_raw", "probe_format", "probe_ids", "probe_summary", "probe_assets"}
ADAPTER_FILES = {'adapter_config.json': '165dbc59c340d68b9756443331cdc1a5f3e81c1834b1237a99d004c33e14cbb9', 'adapter_model.safetensors': '8e32daeea3c078af25a7d01e9d9cebf1fe8d99bbfaf2ca5e2c172e643d346513'}

STATIC_INFERENCE = {'inference_source', 'inference_launcher', 'inference_plan'}
ACTUAL_NAMES = {'responses', 'canonical', 'summary', 'complete', 'wrapper', 'gpu_release', 'actual_input'}
POLICY = {
 'diagnostic': 'Only complete natural-EOS supplied-CID/PID canonical mappings; invalid whole rows abstain.',
 'primary': 'Literal per-document per-patient old-SFT48 union fulltext-alone.',
 'canonical': 'Consume frozen runner canonical records; never reparse raw, repair answers, or generate again.',
 'literal': 'Keep original concept strings including compound HPO/NIL; unchanged v2 handles exact HP compounds.',
 'selection_comparator': 'Cached old-SFT48 metrics, zero old reevaluations.',
 'retention': 'Primary Score strictly increases AND macF1 does not decrease.',
 'evaluation': 'Freeze both complete predictions together before gold; exactly one v2 all/text each.',
 'nonrelation': 'All bytes outside each old baseline top-level association value remain unchanged.',
 'scope': 'Known F0 development only; no originaldev/official/generalization claim.',
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


HELPERS_SHA = '1c70b9c4a8fb61be607618564409a1ea3480f9e099900c4d9b9aaf4b6ded2bf0'
helper_path = Path(__file__).resolve().with_name('literal_helpers.py')
require(sha(helper_path) == HELPERS_SHA, 'Frozen generic helper source differs')
helpers = module('sft48_full_report_literal_helpers', helper_path)
association_span = helpers.association_span
nonrelation_bytes = helpers.nonrelation_bytes
relation_map = helpers.relation_map
assemble_two = helpers.assemble_two
def retain(candidate, baseline):
    values = [candidate[k] for k in ('Score', 'macF1')] + [baseline[k] for k in ('Score', 'macF1')]
    require(all(math.isfinite(value) for value in values), 'Nonfinite retention metric')
    return candidate['Score'] > baseline['Score'] and candidate['macF1'] >= baseline['macF1']


def canonical_gate(prompt, raw, item):
    """Validate supplied canonical identities, without parsing the raw response."""
    require(raw['row_id'] == item['row_id'] == prompt['row_id'] and
            raw['pmc_id'] == item['pmc_id'] == prompt['pmc_id'], 'Row/document identity differs')
    require(type(raw['response']) is str and type(raw['natural_eos']) is bool and
            type(raw['hit_token_limit']) is bool, 'Invalid raw text/EOS/cap type')
    require(type(raw['output_tokens']) is int and 0 < raw['output_tokens'] <= 1600 and
            type(raw['input_tokens']) is int and 0 < raw['input_tokens'] <= 16384,
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
            plan['historical_F0_gold_sha256'] == GOLD_SHA256, 'Evaluation source/gold metadata differs')
    common, inference, actual = plan['references'], plan['inference_references'], plan['actual_references']
    require(set(common) == set(SHARED_SHA) | DYNAMIC_COMMON and set(inference) == STATIC_INFERENCE and
            set(actual) == ACTUAL_NAMES, 'Reference whitelist differs')
    records = list(common.values()) + list(inference.values())
    if require_ready:
        records += list(actual.values())
    else:
        require(all(value is None for value in actual.values()), 'Preparation contains unbound actual results')
    gold_path = Path(plan['gold_path']).resolve()
    require(str(gold_path).endswith('/round2-inner4/fold-0/validation16.gold.jsonl'), 'Wrong F0 gold metadata path')
    for record in records:
        require(Path(record['path']).resolve() != gold_path and record['sha256'] != GOLD_SHA256,
                'Gold cannot enter prequality references')
        checked(record)
    for name, digest in SHARED_SHA.items():
        require(common[name]['sha256'] == digest, 'Frozen common identity differs: ' + name)
    return common, inference, actual, records


def verify_inference(common, inference, actual, terminal_dir):
    """Actual whole58 gate; pure CPU and no label/quality reads."""
    fixed = read(inference['inference_plan']['path']); probe = read(common['probe_plan']['path'])
    require(fixed['cap_seconds'] == 1200 and fixed['term_grace_seconds'] == 30 and
            fixed['input_limit'] == 16384 and fixed['generation_limit'] == 1600 and
            fixed['cached_original_indices'] == [30], 'Frozen runtime caps/cache differ')
    census = read(common['input_actual']['path'])
    require(census['rows'] == 58 and census['documents'] == 16 and census['patients'] == 47 and
            census['candidates'] == 415 and census['pairs'] == 1166 and census['probe_input_gate_passed'] is True and
            census['expanded']['total'] == 505090 and census['expanded']['maximum'] == 16245 and
            census['expanded']['longest_indices'] == [30] and census['recommended_input_limit'] == 16384 and
            census['generation_limit'] == 1600 and census['model_context_tokens'] >= 17984,
            'Actual F0 native CPU contract differs')
    require(fixed['cached_probe_index'] == 30 and fixed['expected_input_tokens'] == 505090 and
            fixed['expected_input_max'] == 16245 and fixed['adapter_files'] == ADAPTER_FILES and
            probe['probe_index'] == 30 and probe['input_limit'] == 16384 and probe['generation_limit'] == 1600,
            'Full58 plan/probe/original SFT48 identity differs')
    require(probe['actual_CPU']['sha256'] == common['input_actual']['sha256'] and
            probe['CPU_row_audit']['sha256'] == common['input_audit']['sha256'], 'Probe native-CPU binding differs')
    for key in ('model', 'adapter', 'adapter_files', 'generation', 'gpu_uuid', 'python'):
        require(fixed[key] == probe[key], 'Probe/runtime configuration drift: ' + key)
    local_plan_names = {'infer_full58.py': inference['inference_source'], 'launch.py': inference['inference_launcher'],
        'expanded58.jsonl': common['expanded_prompts'], 'row-audit.json': common['input_audit'],
        'cache-probe-raw.json': common['probe_raw'], 'cache-probe-ids.json': common['probe_ids'],
        'cache-probe-format.json': common['probe_format'], 'cache-probe-summary.json': common['probe_summary']}
    for name, record in local_plan_names.items():
        require(fixed['files'][name]['sha256'] == record['sha256'] and fixed['files'][name]['bytes'] == record['bytes'],
                'Runtime deployment differs: ' + name)
    summary = read(actual['summary']['path']); complete = read(actual['complete']['path'])
    require(summary == complete and summary['status'] == COMPLETE and summary['expected_rows'] == 58 and
            summary['fresh_rows'] == 57 and summary['cached_rows'] == 1 and summary['cached_original_indices'] == [30],
            'Inference not the unique complete fixed58 run')
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
    require(asset['rows'] == 58 and asset['documents'] == 16 and asset['candidate_codes'] == 415 and
            asset['input_tokens'] == 505090 and asset['max_input_tokens'] == 16245 and
            asset['all58_actual_ids_equal_prior_native_CPU'] is True and asset['no_truncation'] is True and
            asset['gold_read'] is False and asset['max_position_embeddings'] >= 17984,
            'Actual input/context/asset receipt differs')
    require({key: value['sha256'] for key, value in asset['adapter_files'].items()} == fixed['adapter_files'],
            'Actual adapter bytes differ')
    probe_assets = read(common['probe_assets']['path'])
    runtime_keys = {'torch','transformers','peft','tokenizers','actual_tokenizer_class','matmul_TF32','cudnn_TF32'}
    require(set(fixed['cached_probe_runtime']) == runtime_keys and
            {key: probe_assets[key] for key in runtime_keys} == fixed['cached_probe_runtime'],
            'Full58 cached runtime is not the actual original probe runtime')
    require(all(asset[key] == census['runtime'][key] for key in ('torch','transformers','tokenizers')),
            'Native CPU and actual full58 tokenizer libraries differ')
    require({key: asset[key] for key in fixed['cached_probe_runtime']} == fixed['cached_probe_runtime'],
            'Actual library/tokenizer/TF32 runtime differs from cached probe')
    originals = rows(common['prompts']['path']); expanded = rows(common['expanded_prompts']['path'])
    audit = read(common['input_audit']['path']); raw = rows(actual['responses']['path']); canonical = rows(actual['canonical']['path'])
    require(len(originals) == len(expanded) == len(audit) == len(raw) == len(canonical) == 58,
            'Incomplete full58 files')
    require(len({p['row_id'] for p in originals}) == 58 and len({p['pmc_id'] for p in originals}) == 16 and
            sum(len(p['code_map']) for p in originals) == 415 and
            sum(len(p['code_map']) * len(p['patient_ids']) for p in originals) == 1166, 'Original F0 universe differs')
    expected_order = sorted(range(58), key=lambda i: (audit[i]['expanded']['tokens'], originals[i]['row_id']))
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
                c['provenance'] == ('cached_probe' if index == 30 else 'fresh'), 'Actual input/provenance drift')
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
        if index == 30:
            require(r == read(common['probe_raw']['path']) and records[0]['sha256'] == common['probe_raw']['sha256'] and
                    records[2]['sha256'] == common['probe_ids']['sha256'] and
                    c['candidate_patients'] == read(common['probe_format']['path'])['parsed'], 'Cached probe bytes/canonical drift')
    valid = sum(x is not None for x in evidence)
    require(summary['strict_complete_rows'] == valid and summary['whole_row_abstentions'] == 58 - valid,
            'Summary canonical counts differ')
    coverage = {'rows': 58, 'fresh_rows': 57, 'cached_rows': 1, 'cached_original_indices': [30],
        'strict_complete_rows': valid, 'whole_row_abstentions': 58 - valid,
        'cap_rows': sum(r['hit_token_limit'] for r in raw), 'input_tokens': sum(r['input_tokens'] for r in raw),
        'raw_text_reparsed': False, 'missing_CIDs_filled': False, 'raw_shards_verified': 174}
    require(coverage['input_tokens'] == 505090, 'Actual total input tokens differ')
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
    require(len(documents) == 16 and sum(len(d['entities']) for d in documents) == 947 and
            sum(len(d['association']) for d in documents) == 47 and
            parse_stats.get('provided_CIDs', 0) + parse_stats.get('abstained_CIDs', 0) == 415,
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
    evaluator = module('sft48_full_report_fixed_evaluator', common['evaluator']['path'])
    require(evaluator.VERSION == 'patientphex-public-approx-v2', 'Public evaluator version differs')
    hpo = validator.read_hpo(common['hpo']['path']); predictions = {}
    for name in ENDPOINTS:
        predictions[name] = rows(checked(frozen[name]['prediction']))
        validation = validator.validate_predictions(inputs, predictions[name], hpo)
        save(directory / name / 'validation.json', validation)
        require(validation['valid'], 'Invalid full prediction: ' + name)
    require(sha(plan['gold_path']) == GOLD_SHA256, 'Historical F0 gold identity differs')
    truth = rows(plan['gold_path'])
    require(len(truth) == 16 and [str(d['pmc_id']) for d in truth] == [str(d['pmc_id']) for d in inputs],
            'Historical F0 document order differs')
    old = read(common['old_baseline_metrics']['path']); metrics = {}
    require(old['metrics']['Score'] == 0.6954439475814587, 'Cached baseline score differs')
    for name in ENDPOINTS:
        checked(frozen[name]['prediction'])
        metrics[name] = evaluator.evaluate_report(truth, predictions[name], 'all', 'text')
        require(metrics[name]['documents'] == 16 and metrics[name]['patients'] == 47 and
                all(metrics[name]['metrics'][key] == old['metrics'][key] for key in ('menF1', 'docF1')),
                'Nonrelation metrics or F0 scope drift')
        save(directory / name / 'metrics.json', metrics[name])
    for record in records + shard_refs:
        checked(record)
    checked(plan_ref); checked(freeze_ref)
    for value in frozen.values():
        checked(value['prediction'])
    summary = {'status': 'complete_predeclared_SFT48_full_report_pair_F0_development', 'policy': POLICY,
        'paired_freeze': freeze_ref, 'old_cached_metrics': common['old_baseline_metrics'], 'old_metrics': old['metrics'],
        'endpoints': {name: {'metrics': metrics[name]['metrics'], 'counts': metrics[name]['counts'],
            'metrics_file': ref(directory / name / 'metrics.json'), 'format_file': ref(directory / name / 'validation.json'),
            'delta_vs_old': delta(metrics[name]['metrics'], old['metrics'])} for name in ENDPOINTS},
        'primary_retain_vs_old': retain(metrics[ENDPOINTS[1]]['metrics'], old['metrics']),
        'diagnostic_is_not_selection_endpoint': True,
        'diagnostic_retain_vs_old_report_only': retain(metrics[ENDPOINTS[0]]['metrics'], old['metrics']), 'new_v2_calls': 2, 'old_v2_calls': 0,
        'automatic_migration': False, 'elapsed_CPU_seconds': time.monotonic() - started}
    save(directory / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'summary': ref(directory / 'summary.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
