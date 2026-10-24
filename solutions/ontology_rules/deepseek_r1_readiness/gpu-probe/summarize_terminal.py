"""Summarize already recovered native R1 receipts; no model, parser or quality run.

CLI: --plan ORIGINAL_EXECUTED_PLAN --terminal-dir RECOVERED/terminal
     --output NEW_SUMMARY [--wrapper RECOVERED/wrapper-result.json]
     [--source-dir ORIGINAL_THREE_SOURCES]
Only main reads files. Small pure helpers support synthetic boundary checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SOURCE_SHA256 = {
    'run_probe.py': '7eaa96d808e4f2b1ec49a12cc1768d4850b1b3278b20341f9d4949b9318c9fee',
    'launch_probe.py': '084aabe899a63d6cedf1e46586dd612e1a312c74d4d234932e925189a316c9d3',
    'strict_final.py': '20943d8681b4cf81aee0465b0f217d3491460a311be00f3bbc2818b721013fad',
}
PASS = 'PASS_THREE_NATIVE_FINAL_FORMAT_COST_ONLY_NOT_QUALITY'
FIRST_STOP = 'STOPPED_FIRST_TRUNCATED_OR_INVALID_FINAL_THIS_BUDGET_ONLY'
FAILED = 'FAILED_ENGINEERING_OR_RESOURCE_PRESERVED'
INPUT_TOKENS = [2171, 3425, 6002]
ROW_BOUND_FIELDS = ['row_id', 'input_tokens', 'generated_tokens', 'native_EOS_observed',
                    'stopped_at_token_cap', 'stopped_at_wall_budget', 'generation_seconds',
                    'tokens_per_second_including_prefill', 'strict_final_valid']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def match_ref(actual, recorded):
    require(actual['sha256'] == recorded['sha256'] and actual['bytes'] == recorded['bytes'],
            'Recovered bytes do not match recorded ref: ' + recorded.get('path', '?'))


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), 'Invalid numeric receipt: ' + name)
    return value


def id_summary(ids, count):
    require(type(count) is int and 0 <= count <= 1024 and isinstance(ids, list) and
            len(ids) == count and all(type(value) is int and value >= 0 for value in ids),
            'Invalid full generated-ID vector/count')
    return {'count': count,
            'canonical_json_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            'serialization': 'json.dumps(full_generated_ids,separators=(comma,colon)).encode(utf-8)'}


def row_summary(item, index, row_ids):
    require(item['row_id'] == row_ids[index] and item['input_tokens'] == INPUT_TOKENS[index],
            'Completed row identity/token prefix differs from frozen probe')
    ids = id_summary(item['generated_ids'], item['generated_tokens'])
    seconds = finite(item['generation_seconds'], 'generation_seconds', positive=True)
    rate = finite(item['tokens_per_second_including_prefill'], 'tokens_per_second_including_prefill')
    require(math.isclose(rate, ids['count'] / seconds, rel_tol=1e-9, abs_tol=1e-12),
            'Recorded token rate/count/time disagree')
    for name in ('native_EOS_observed', 'stopped_at_token_cap', 'stopped_at_wall_budget',
                 'strict_final_valid', 'quality_evaluated'):
        require(type(item[name]) is bool, 'Invalid row boolean: ' + name)
    require(item['quality_evaluated'] is False, 'Receipt unexpectedly claims quality evaluation')
    require(item['stopped_at_token_cap'] == (not item['native_EOS_observed'] and ids['count'] >= 1024),
            'Cap/EOS/count flags are inconsistent')
    require(not item['strict_final_valid'] or
            (item['native_EOS_observed'] and not item['stopped_at_token_cap'] and
             not item['stopped_at_wall_budget']), 'Valid-final receipt contradicts truncation flags')
    result = {name: item[name] for name in ROW_BOUND_FIELDS}
    result.update(selection_role=item.get('selection_role'), generated_ids=ids,
                  peak_allocated_bytes=finite(item['peak_allocated_bytes'], 'peak_allocated_bytes'),
                  peak_reserved_bytes=finite(item['peak_reserved_bytes'], 'peak_reserved_bytes'),
                  quality_evaluated=False)
    if not item['strict_final_valid']:
        result['invalid_reason'] = str(item.get('invalid_reason', 'Not recorded'))[:240]
    return result


def release_summary(wrapper, uuid):
    release = wrapper.get('actual_release')
    if release is None:
        return {'verified_release_snapshot': False, 'target_UUID_has_no_compute_process': None,
                'release_query_error': str(wrapper.get('release_query_error', 'No release receipt'))[:240]}
    fields = [part.strip() for part in release['gpu'].split(',')]
    require(len(fields) == 6 and fields[0] == '1' and fields[1] == uuid and 'A30' in fields[2],
            'Release snapshot does not describe the frozen physical GPU1/UUID')
    processes = []
    for line in release['compute_apps'].splitlines():
        fields_process = [part.strip() for part in line.split(',')]
        require(len(fields_process) == 2 and fields_process[1].isdigit(), 'Invalid compute-process receipt')
        processes.append((fields_process[0], int(fields_process[1])))
    clear = not any(gpu == uuid for gpu, _ in processes)
    if 'target_UUID_has_no_compute_process' in wrapper:
        require(wrapper['target_UUID_has_no_compute_process'] is clear, 'Release flag contradicts actual process list')
    return {'verified_release_snapshot': True, 'at': release['at'], 'physical_index': 1,
            'uuid': uuid, 'name': fields[2], 'memory_used_MiB': int(fields[3]),
            'memory_free_MiB': int(fields[4]), 'memory_total_MiB': int(fields[5]),
            'target_UUID_has_no_compute_process': clear,
            'scope': 'Recorded terminal snapshot only; no new GPU query or process action.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--terminal-dir', required=True)
    parser.add_argument('--wrapper')
    parser.add_argument('--source-dir', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    terminal = Path(args.terminal_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    wrapper_path = Path(args.wrapper).resolve() if args.wrapper else terminal.parent / 'wrapper-result.json'
    plan_ref, result_ref, wrapper_ref = ref(args.plan), ref(terminal / 'result.json'), ref(wrapper_path)
    plan, report, wrapper = read(args.plan), read(terminal / 'result.json'), read(wrapper_path)
    require(plan['execution_ready'] is True and plan['process_budget_seconds'] == 180 and
            plan['TERM_grace_seconds'] == 30 and plan['max_new_tokens'] == 1024,
            'Not the original authorized fixed-budget plan')
    require(plan['sources'] == SOURCE_SHA256, 'Native probe source set changed')
    source_refs = {name: ref(source_dir / name) for name in SOURCE_SHA256}
    require(all(source_refs[name]['sha256'] == expected for name, expected in SOURCE_SHA256.items()),
            'Local source bytes differ from frozen native probe')
    match_ref(plan_ref, report['plan'])
    match_ref(source_refs['run_probe.py'], report['source'])
    require(wrapper['plan_sha256'] == plan_ref['sha256'] and
            wrapper['source_sha256'] == SOURCE_SHA256['launch_probe.py'], 'Wrapper plan/source binding changed')
    require(wrapper['cap_seconds'] == 180 and wrapper['TERM_grace_seconds'] == 30 and
            wrapper['only_original_fixed_three_rows'] is True and wrapper['no_gold_quality'] is True,
            'Wrapper fixed-budget/no-quality contract changed')
    require(report['quality'] is False and report['original80_not_inferred'] is True,
            'Terminal is outside the format/cost-only scope')
    require(report['status'] in (PASS, FIRST_STOP, FAILED), 'Unknown/incomplete terminal status')
    if 'result_sha256' in wrapper:
        require(wrapper['result_sha256'] == result_ref['sha256'] and
                wrapper['result_status'] == report['status'] and wrapper['result_exists'] is True,
                'Wrapper result hash/status differs from recovered terminal')
    else:
        require(wrapper['status'] == 'failed_wrapper_preserved', 'Completed wrapper omits terminal binding')
    row_ids = plan['row_ids']
    require(len(row_ids) == len(set(row_ids)) == 3 and len(report['rows']) <= 3, 'Unexpected frozen row count')
    completed, bound_ids = [], {}
    for index, recorded in enumerate(report['rows']):
        require(Path(recorded['path']).name == 'row%d.json' % index, 'Terminal rows are not the fixed prefix')
        local = terminal / ('row%d.json' % index)
        local_ref = ref(local)
        match_ref(local_ref, recorded)
        item = read(local)
        require(all(item[name] == recorded[name] for name in ROW_BOUND_FIELDS), 'Row fields differ from terminal ref entry')
        summary = row_summary(item, index, row_ids)
        summary['file'] = local_ref
        completed.append(summary)
        bound_ids[index] = item['generated_ids']
    if report['status'] == PASS:
        require(len(completed) == 3 and all(item['strict_final_valid'] for item in completed), 'PASS row contract mismatch')
    elif report['status'] == FIRST_STOP:
        require(bool(completed) and not completed[-1]['strict_final_valid'] and
                all(item['strict_final_valid'] for item in completed[:-1]), 'First-stop prefix contract mismatch')
    partial, unbound = [], []
    for path in sorted(terminal.iterdir()):
        match = re.fullmatch(r'row([0-2])\.partial\.json', path.name)
        if match:
            index = int(match[1]); item = read(path)
            require(item['row_id'] == row_ids[index] and type(item['wall_stop']) is bool, 'Partial row identity/flag mismatch')
            ids = id_summary(item['generated_ids'], item['generated_tokens'])
            if index in bound_ids:
                require(item['generated_ids'] == bound_ids[index][:ids['count']], 'Partial IDs disagree with completed-row prefix')
            partial.append({'row_id': item['row_id'], 'file': ref(path), 'generated_ids': ids,
                            'seconds': finite(item['seconds'], 'partial_seconds'), 'wall_stop': item['wall_stop'],
                            'complete_final_claim': False, 'also_has_bound_completed_row': index in bound_ids,
                            'binding': 'Recovered intermediate file; no terminal row-ref binding is claimed.'})
        elif re.fullmatch(r'row[0-9]+\.json', path.name) and path.name not in {
                'row%d.json' % i for i in range(len(completed))}:
            unbound.append({'file': ref(path), 'complete_final_claim': False,
                            'binding': 'File is absent from terminal rows; not interpreted as a completed generation.'})
    loading = report.get('loading_information')
    loading_counts = ({name: len(loading.get(name, [])) for name in
                       ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
                      if loading is not None else None)
    if report['status'] in (PASS, FIRST_STOP):
        require(loading_counts is not None and not any(loading_counts.values()), 'Successful native loading contract missing')
        require(report['parameters']['tensors'] == 399 and report['parameters']['count'] == 8190735360 and
                report['parameters']['all_actual_shapes_dtype_finite'] is True, 'Actual parameter integrity receipt missing')
    summary = {
        'status': 'verified_received_terminal_summary_no_quality',
        'source': ref(__file__), 'plan': plan_ref, 'result': result_ref, 'wrapper_result': wrapper_ref,
        'verified_probe_sources': source_refs, 'terminal_status': report['status'],
        'wrapper_status': wrapper['status'], 'wrapper_returncode': wrapper.get('returncode'),
        'wrapper_result_hash_binding_present': 'result_sha256' in wrapper,
        'program_seconds': report.get('program_seconds'), 'wrapper_GPU_main_seconds': wrapper.get('GPU_main_seconds'),
        'load_seconds': report.get('load_seconds'), 'weight_shape_finite_seconds': report.get('weight_shape_finite_seconds'),
        'actual_loading_information_counts': loading_counts, 'actual_parameters': report.get('parameters'),
        'runtime': report.get('runtime'), 'peak_memory': report.get('peak_memory'),
        'completed_rows': completed, 'partial_snapshots': partial, 'unbound_row_files': unbound,
        'release': release_summary(wrapper, plan['gpu_uuid']),
        'quality_evaluated': False, 'original80_not_inferred': True,
        'first_stop_is_model_quality_failure': False,
        'interpretation': 'This receipt describes only native loading, final-format and cost within the fixed 1024-token/180-second budget. First truncation/invalid final stops further rows; neither that stop nor an engineering/resource error establishes model task quality.',
        'generated_content_policy': 'No generated text, parsed answers or ID arrays copied. Completed/partial files retain original bytes; ID arrays are referenced only by count and canonical JSON SHA256.',
        'scope': 'Local receipt/hash consistency summary. No model/tokenizer/parser/validator rerun, new GPU query, gold, fit, prediction or quality evaluation.',
    }
    for name in ('stop_reason', 'error'):
        if name in report:
            summary[name] = str(report[name])[:240]
    if 'error' in wrapper:
        summary['wrapper_error'] = str(wrapper['error'])[:240]
    with Path(args.output).open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': summary['status'], 'terminal_status': summary['terminal_status'],
                      'completed_rows': len(completed), 'partial_snapshots': len(partial),
                      'output': ref(args.output)}))


if __name__ == '__main__':
    main()
