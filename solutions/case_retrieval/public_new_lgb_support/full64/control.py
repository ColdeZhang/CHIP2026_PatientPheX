"""Replace only the original35 public-new support subset of frozen0e2c."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import sys
import time


def load_helper(value):
    path = Path(value['path'])
    assert path.stat().st_size == value['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
    spec = importlib.util.spec_from_file_location('frozen_public_support_helpers', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_subset(helper, line, removed, added):
    old = json.loads(line)
    updated = deepcopy(old['association'])
    did = str(old['pmc_id'])
    actual_removed, actual_added = set(), set()
    for previous, current in zip(old['association'], updated):
        pid = previous['patient_id']
        current['phenotype'] = [value for value in previous['phenotype']
                                if (did, pid, value) not in removed]
        actual_removed.update((did, pid, value) for value in previous['phenotype']
                              if (did, pid, value) in removed)
        extra = sorted(value for doc, patient, value in added if doc == did and patient == pid)
        assert not set(extra) & set(previous['phenotype'])
        current['phenotype'].extend(extra)
        actual_added.update((did, pid, value) for value in extra)
        assert {k: v for k, v in previous.items() if k != 'phenotype'} == {
            k: v for k, v in current.items() if k != 'phenotype'}
    assert actual_removed == {key for key in removed if key[0] == did}
    assert actual_added == {key for key in added if key[0] == did}
    if updated == old['association']:
        return line
    begin, end = helper.field_span(line, 'association')
    result = line[:begin] + json.dumps(updated, ensure_ascii=False) + line[end:]
    a, b = helper.field_span(result, 'association')
    assert (line[:begin].encode(), line[end:].encode()) == (result[:a].encode(), result[b:].encode())
    assert {k: v for k, v in old.items() if k != 'association'} == {
        k: v for k, v in json.loads(result).items() if k != 'association'}
    return result


def execute(plan_path, output):
    started = time.monotonic()
    plan = json.loads(plan_path.read_text())
    assert plan['policy_id'] == 'only-original35-replaced-by-frozen-LGB-support-v1'
    assert plan['cap_seconds'] == 30 and plan['new_report_calls'] == 1
    helper = load_helper(plan['sources']['helper'])
    assert helper.ref(__file__) == plan['sources']['control']
    for value in plan['sources'].values():
        helper.checked(value)
    inventory = helper.read(helper.checked(plan['source_inventory']))
    assert inventory['status'] == 'PASS_EXACT_ORIGINAL35_AND_RAW145_LINEAGE_NO_NEW_SUPPORT_FILTER_OR_SCORE'
    for value in inventory['evidence'].values():
        helper.checked(value)
    paths = {name: helper.checked(value) for name, value in plan['inputs'].items()}
    lines, baseline, ids = helper.document_rows(paths['current_base'])
    _, inputs, input_ids = helper.document_rows(paths['pure_input'])
    assert len(ids) == 16 and set(ids) == set(input_ids) == set(inventory['isolation']['dev_ids'])
    assert all(set(row) == {'pmc_id', 'pmid', 'patient', 'full_text'} for row in inputs)
    assert sum(len(row['patient']) for row in inputs) == 36
    assert sum(len(row['entities']) for row in baseline) == 1061
    raw_list, old_list = helper.read(paths['raw_keys']), helper.read(paths['original35'])
    raw = {tuple(key) for key in raw_list}
    original35 = {tuple(key) for key in old_list}
    assert len(raw) == len(raw_list) == 145 and len(original35) == len(old_list) == 35
    assert original35 <= raw and all(re.fullmatch(r'HP:[0-9]{7}', key[2]) for key in raw)
    old_keys = helper.associations(baseline)
    assert len(old_keys) == 391 and old_keys & raw == original35
    _, historical_baseline, historical_ids = helper.document_rows(paths['historical_baseline'])
    _, historical_supported, historical_supported_ids = helper.document_rows(paths['historical_supported'])
    assert historical_ids == historical_supported_ids == ids
    historical_keys, historical_support_keys = helper.associations(historical_baseline), helper.associations(historical_supported)
    assert len(historical_keys) == 333 and len(historical_support_keys) == 368
    assert historical_keys <= historical_support_keys <= old_keys
    assert historical_support_keys - historical_keys == original35
    _, reverse_reference, reverse_ids = helper.document_rows(paths['old_a179'])
    assert reverse_ids == ids and all(a['association'] == b['association'] for a, b in zip(baseline, reverse_reference))
    protected = old_keys - original35
    assert len(protected) == 356
    old_metrics = helper.read(paths['old_metrics'])
    old_result = helper.read(paths['old_result'])
    assert old_result['prediction'] == plan['inputs']['current_base']
    assert old_result['lightgbm_metrics'] == old_metrics['metrics'] == plan['old_metrics']
    assert old_metrics['metrics']['Score'] == .748583439566388
    # New support is projected from the already frozen full64 LGB entity output itself.
    # No estimator/vectorizer/tokenizer is loaded and raw LLM decisions are not parsed again.
    support = helper.entity_support(baseline)
    retained = {key for key in raw if (key[0], key[2]) in support}
    removed, added = original35 - retained, retained - original35
    assert removed <= original35 and added <= raw and not retained & protected
    prediction = output / 'supported.pred.jsonl'
    prediction.write_bytes(''.join(replace_subset(helper, line, removed, added) for line in lines).encode('utf-8'))
    _, predicted, predicted_ids = helper.document_rows(prediction)
    assert predicted_ids == ids and helper.associations(predicted) == protected | retained
    for before, after in zip(baseline, predicted):
        did = str(before['pmc_id'])
        for old, new in zip(before['association'], after['association']):
            pid = old['patient_id']
            assert [value for value in old['phenotype'] if (did, pid, value) not in original35] == [
                value for value in new['phenotype'] if (did, pid, value) not in retained]
    validator = helper.load_module('validate_predictions', plan['validator'])
    evaluator = helper.load_module('full64_support_evaluator', plan['evaluator'])
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    hpo = validator.read_hpo(helper.checked(plan['hpo']))
    valid = validator.validate_predictions(inputs, predicted, hpo)
    assert valid['valid'] and valid['error_count'] == valid['warning_count'] == 0
    helper.write(output / 'validation.json', valid)
    helper.write(output / 'retained-new-keys.json', sorted(retained))
    helper.write(output / 'support-delta.json', dict(removed=sorted(removed), added=sorted(added)))
    frozen = dict(status='complete16_prediction_frozen_before_gold_read', at_utc=helper.now(),
                  plan=helper.ref(plan_path), source_inventory=plan['source_inventory'], baseline=plan['inputs']['current_base'],
                  prediction=helper.ref(prediction), validation=helper.ref(output / 'validation.json'),
                  retained_keys=helper.ref(output / 'retained-new-keys.json'), support_delta=helper.ref(output / 'support-delta.json'),
                  counts=dict(raw=145, original_supported=35, new_supported=len(retained), removed=len(removed), added=len(added),
                              protected_relations=356, total_relations=356 + len(retained), entities=1061, patients=36, documents=16),
                  entities_and_nonassociation_literal_bytes_exact=True, protected_relation_values_and_order_exact=True,
                  no_fit_predict_NN=True, gold_read=False, seconds=time.monotonic() - started)
    helper.write(output / 'complete-freeze.json', frozen)
    helper.checked(frozen['prediction'])
    helper.write(output / 'evaluation-started.json', dict(at_utc=helper.now(), freeze=helper.ref(output / 'complete-freeze.json')))
    _, gold, gold_ids = helper.document_rows(helper.checked(plan['gold']))
    assert set(gold_ids) == set(ids)
    assert {str(row['pmc_id']): {k: v for k, v in row.items() if k not in ('entities', 'association')} for row in gold} == {
        str(row['pmc_id']): row for row in inputs}
    report = evaluator.evaluate_report(gold, predicted, 'all', 'text')
    assert report['documents'] == 16 and report['patients'] == 36
    assert all(report['metrics'][key] == old_metrics['metrics'][key] for key in ['menF1', 'docF1'])
    assert all(report['counts'][key] == old_metrics['counts'][key] for key in ['mention', 'document'])
    report.update(gold_sha256=plan['gold']['sha256'], prediction_sha256=frozen['prediction']['sha256'])
    helper.write(output / 'metrics.json', report)
    helper.checked(frozen['prediction'])
    keep = report['metrics']['Score'] > old_metrics['metrics']['Score']
    result = dict(status='complete_unique_originaldev_support_transfer', scope=plan['scope'],
                  plan=helper.ref(plan_path), freeze=helper.ref(output / 'complete-freeze.json'),
                  prediction=frozen['prediction'], metrics=helper.ref(output / 'metrics.json'),
                  old_metrics=old_metrics['metrics'], new_metrics=report['metrics'],
                  delta={key: report['metrics'][key] - old_metrics['metrics'][key] for key in old_metrics['metrics']},
                  counts_before=old_metrics['counts'], counts_after=report['counts'], support_counts=frozen['counts'],
                  keep=keep, keep_as_candidate_best=keep, meets_0_8_score=report['metrics']['Score'] >= .8,
                  new_evaluate_report_calls=1, old_evaluation_calls=0, new_fit_calls=0, new_model_predict_calls=0,
                  new_NN_calls=0, originaldev_is_development=True, no_sequence_baseline_modification=True,
                  no_official_actions=True, seconds=time.monotonic() - started)
    assert result['seconds'] < 30
    helper.write(output / 'result.json', result)
    return result


def self_test(helper):
    line = ' {"pmc_id":"d", "entities":[], "association" : [{"patient_id":"p","phenotype":["a","HP:0000001","b","HP:0000002"],"extra":"keep"}], "meta":"保留" }\r\n'
    changed = replace_subset(helper, line, {('d', 'p', 'HP:0000001')}, {('d', 'p', 'HP:0000003')})
    assert json.loads(changed)['association'][0]['phenotype'] == ['a', 'b', 'HP:0000002', 'HP:0000003']
    assert changed.endswith(' }\r\n') and replace_subset(helper, line, set(), set()) == line
    for removed, added in [({('d', 'p', 'absent')}, set()), (set(), {('d', 'missing', 'HP:0000003')})]:
        try:
            replace_subset(helper, line, removed, added)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid replacement accepted')
    return dict(status='PASS_SYNTHETIC_SUBSET_REPLACEMENT_LITERAL_AND_PROTECTED_ORDER', cases=4,
                real_prediction_assembled=False, gold_read=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test-helper', type=Path)
    args = parser.parse_args()
    if args.self_test_helper:
        path = args.self_test_helper.resolve()
        helper = load_helper(dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(self_test(helper)))
        return
    assert args.plan and args.output and args.output.is_dir()
    signal.alarm(30)
    result = execute(args.plan.resolve(), args.output.resolve())
    signal.alarm(0)
    print(json.dumps(dict(status=result['status'], keep=result['keep'], seconds=result['seconds'])))


if __name__ == '__main__':
    main()
