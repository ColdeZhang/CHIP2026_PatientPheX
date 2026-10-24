"""One cache-only, two-fold public-new relation support-stack control."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(value):
    path = Path(value['path'])
    actual = ref(path)
    assert actual['sha256'] == value['sha256'], path
    assert actual['bytes'] == value['bytes'], path
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def document_rows(path):
    lines = Path(path).read_bytes().decode('utf-8').splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    ids = [str(row['pmc_id']) for row in rows]
    assert len(ids) == len(set(ids))
    return lines, rows, ids


def field_span(line, wanted):
    """Locate a top-level value, rejecting duplicate keys and trailing content."""
    decoder = json.JSONDecoder()
    def space(position):
        while position < len(line) and line[position].isspace():
            position += 1
        return position
    position = space(0)
    assert line[position] == '{'
    position += 1
    names, found = set(), None
    while True:
        position = space(position)
        if line[position] == '}':
            break
        name, position = decoder.raw_decode(line, position)
        assert isinstance(name, str) and name not in names
        names.add(name)
        position = space(position)
        assert line[position] == ':'
        position = space(position + 1)
        begin = position
        _, position = decoder.raw_decode(line, position)
        if name == wanted:
            found = begin, position
        position = space(position)
        if line[position] == '}':
            break
        assert line[position] == ','
        position += 1
    assert not line[position + 1:].strip() and found is not None
    return found


def associations(rows):
    keys = set()
    for doc in rows:
        patients = set()
        for relation in doc['association']:
            pid = relation['patient_id']
            assert pid not in patients
            patients.add(pid)
            values = relation['phenotype']
            assert len(values) == len(set(values))
            keys.update((str(doc['pmc_id']), pid, value) for value in values)
    return keys


def entity_support(rows):
    result = set()
    for doc in rows:
        for entity in doc['entities']:
            if entity['identifier'] == '-1':
                continue
            for concept in entity['identifier'].split(';'):
                assert re.fullmatch(r'HP:[0-9]{7}', concept)
                result.add((str(doc['pmc_id']), concept))
    return result


def append_literal(line, extra):
    old = json.loads(line)
    relationships = deepcopy(old['association'])
    applied = set()
    for previous, current in zip(old['association'], relationships):
        values = sorted(c for did, pid, c in extra
                        if did == str(old['pmc_id']) and pid == previous['patient_id'])
        assert not set(values) & set(previous['phenotype'])
        current['phenotype'].extend(values)
        applied.update((str(old['pmc_id']), previous['patient_id'], c) for c in values)
        assert current['phenotype'][:len(previous['phenotype'])] == previous['phenotype']
        assert {k: v for k, v in current.items() if k != 'phenotype'} == {
            k: v for k, v in previous.items() if k != 'phenotype'}
    assert applied == {key for key in extra if key[0] == str(old['pmc_id'])}
    if relationships == old['association']:
        return line
    begin, end = field_span(line, 'association')
    result = line[:begin] + json.dumps(relationships, ensure_ascii=False) + line[end:]
    a, b = field_span(result, 'association')
    assert (line[:begin].encode(), line[end:].encode()) == (result[:a].encode(), result[b:].encode())
    assert {k: v for k, v in old.items() if k != 'association'} == {
        k: v for k, v in json.loads(result).items() if k != 'association'}
    return result


def load_module(name, value):
    path = checked(value)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(plan_path, output):
    start = time.monotonic()
    plan = read(plan_path)
    assert plan['policy_id'] == 'fixed-twofold-cached-support-stack-v1'
    assert plan['cap_seconds'] == 30 and plan['new_report_calls'] == 2
    assert [fold['fold'] for fold in plan['folds']] == [0, 1]
    assert ref(__file__) == plan['sources']['control']
    for value in plan['sources'].values():
        checked(value)
    lineage = read(checked(plan['source_lineage']))
    assert lineage['status'] == 'PASS_FOLD_MATCHED_TASK_FIT_ANCESTORS_EXCLUDE_EACH_TARGET_FOLD'
    for value in lineage['evidence'].values():
        checked(value)
    validator = load_module('validate_predictions', plan['validator'])
    evaluator = load_module('public_new_lgb_evaluator', plan['evaluator'])
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    hpo = validator.read_hpo(checked(plan['hpo']))
    frozen = []
    data = {}
    for fold in plan['folds']:
        number = fold['fold']
        target = output / f'fold{number}'
        target.mkdir()
        paths = {name: checked(value) for name, value in fold['inputs'].items()}
        lines, baseline, ids = document_rows(paths['baseline'])
        _, inputs, input_ids = document_rows(paths['input'])
        assert len(ids) == 16 and set(ids) == set(input_ids) == set(fold['document_ids'])
        assert all(set(row) == {'pmc_id', 'pmid', 'patient', 'full_text'} for row in inputs)
        assert sum(len(row['patient']) for row in inputs) == fold['patients']
        assert sum(len(row['entities']) for row in baseline) == fold['entities']
        old_keys = associations(baseline)
        assert len(old_keys) == fold['old_relations']
        raw_list = read(paths['raw_additions'])
        raw_keys = {tuple(value) for value in raw_list}
        assert len(raw_keys) == len(raw_list) == fold['raw_relations']
        assert not raw_keys & old_keys
        supplied_patients = {(str(row['pmc_id']), patient['patient_id'])
                             for row in inputs for patient in row['patient']}
        assert all((did, pid) in supplied_patients and re.fullmatch(r'HP:[0-9]{7}', concept)
                   for did, pid, concept in raw_keys)
        _, raw_pred, raw_ids = document_rows(paths['raw_augmented'])
        assert raw_ids == ids
        assert [json.loads(append_literal(line, raw_keys)) for line in lines] == raw_pred
        assert associations(raw_pred) == old_keys | raw_keys
        _, old_support_rows, old_ids = document_rows(paths['old_support'])
        _, new_support_rows, new_ids = document_rows(paths['new_support'])
        assert set(old_ids) == set(new_ids) == set(ids)
        old_support, new_support = entity_support(old_support_rows), entity_support(new_support_rows)
        old_extra = {key for key in raw_keys if (key[0], key[2]) in old_support}
        new_extra = {key for key in raw_keys if (key[0], key[2]) in new_support}
        assert len(old_extra) == fold['old_supported_relations']
        _, comparator, comparator_ids = document_rows(paths['old_supported_prediction'])
        assert comparator_ids == ids
        assert [json.loads(append_literal(line, old_extra)) for line in lines] == comparator
        old_metrics = read(paths['old_metrics'])
        assert old_metrics['prediction_sha256'] == fold['inputs']['old_supported_prediction']['sha256']
        assert old_metrics['gold_sha256'] == fold['gold']['sha256']
        assert old_metrics['metrics'] == fold['old_metrics']
        assert old_metrics['evaluator_version'] == evaluator.VERSION
        prediction = target / 'supported.pred.jsonl'
        prediction.write_bytes(''.join(append_literal(line, new_extra) for line in lines).encode('utf-8'))
        _, new_rows, new_prediction_ids = document_rows(prediction)
        assert new_prediction_ids == ids
        assert associations(new_rows) == old_keys | new_extra
        validation = validator.validate_predictions(inputs, new_rows, hpo)
        assert validation['valid'] and validation['error_count'] == validation['warning_count'] == 0
        write(target / 'validation.json', validation)
        write(target / 'retained-new-keys.json', sorted(new_extra))
        change = dict(raw=len(raw_keys), old_supported=len(old_extra), new_supported=len(new_extra),
                      added_relative_to_old=len(new_extra - old_extra), removed_relative_to_old=len(old_extra - new_extra))
        frozen.append(dict(fold=number, prediction=ref(prediction), comparator=fold['inputs']['old_supported_prediction'],
                           validation=ref(target / 'validation.json'), new_keys=ref(target / 'retained-new-keys.json'),
                           counts=change, all_old_relations_preserved=True, nonassociation_literal_bytes_preserved=True))
        data[number] = inputs, new_rows, old_metrics
    freeze = dict(status='both_full_predictions_frozen_before_quality_gold_read', at_utc=now(),
                  plan=ref(plan_path), source_lineage=plan['source_lineage'], folds=frozen,
                  new_quality_calls=0, seconds=time.monotonic() - start)
    write(output / 'complete-freeze.json', freeze)
    for value in frozen:
        checked(value['prediction'])
    write(output / 'evaluation-started.json', dict(at_utc=now(), freeze=ref(output / 'complete-freeze.json')))
    results = []
    for fold in plan['folds']:
        number = fold['fold']
        inputs, prediction, old = data[number]
        _, gold, gold_ids = document_rows(checked(fold['gold']))
        assert set(gold_ids) == set(fold['document_ids'])
        stripped = {str(row['pmc_id']): {k: v for k, v in row.items() if k not in ('entities', 'association')} for row in gold}
        assert stripped == {str(row['pmc_id']): row for row in inputs}
        report = evaluator.evaluate_report(gold, prediction, 'all', 'text')
        assert report['documents'] == 16 and report['patients'] == fold['patients']
        assert all(report['metrics'][key] == old['metrics'][key] for key in ['menF1', 'docF1'])
        assert all(report['counts'][key] == old['counts'][key] for key in ['mention', 'document'])
        report.update(gold_sha256=fold['gold']['sha256'], prediction_sha256=frozen[number]['prediction']['sha256'])
        write(output / f'fold{number}/metrics.json', report)
        results.append(dict(fold=number, old_metrics=old['metrics'], new_metrics=report['metrics'],
                            delta={key: report['metrics'][key] - old['metrics'][key] for key in old['metrics']},
                            counts_before=old['counts'], counts_after=report['counts'],
                            metric_path=ref(output / f'fold{number}/metrics.json'),
                            score_strictly_improves=report['metrics']['Score'] > old['metrics']['Score'],
                            macro_nondecreasing=report['metrics']['macF1'] >= old['metrics']['macF1']))
    keep = all(row['score_strictly_improves'] and row['macro_nondecreasing'] for row in results)
    result = dict(status='complete_fixed_twofold_development_control', plan=ref(plan_path),
                  freeze=ref(output / 'complete-freeze.json'), scope=plan['scope'], folds=results, keep=keep,
                  keep_rule='Score strictly increases AND macro nondecreases in EACH of F0/F1',
                  new_evaluate_report_calls=2, old_evaluation_calls=0, new_fit_calls=0,
                  new_model_predict_calls=0, new_NN_calls=0, original_dev_used=False,
                  old_metrics_reused=True, seconds=time.monotonic() - start)
    assert result['seconds'] < plan['cap_seconds']
    write(output / 'result.json', result)
    return result


def self_test():
    line = '{ "pmc_id" : "d", "entities": [], "note":"association \\" quoted", "association" : [{"patient_id":"p","phenotype":["old;literal"]}], "patient": [] }\n'
    # Build the escape-sensitive value with JSON, retaining nonstandard outer whitespace.
    line = line.replace('"association \\" quoted"', json.dumps('association " quoted'))
    extra = {('d', 'p', 'HP:0000002'), ('d', 'p', 'HP:0000001')}
    changed = append_literal(line, extra)
    assert json.loads(changed)['association'][0]['phenotype'] == ['old;literal', 'HP:0000001', 'HP:0000002']
    a, b = field_span(line, 'association'); x, y = field_span(changed, 'association')
    assert line[:a] == changed[:x] and line[b:] == changed[y:]
    assert append_literal(line, set()) == line
    assert entity_support([{'pmc_id': 'd', 'entities': [{'identifier': '-1'}, {'identifier': 'HP:0000001;HP:0000002'}]}]) == {('d', 'HP:0000001'), ('d', 'HP:0000002')}
    for bad in ['{"association":[],"association":[]}', '{"association":[]} trailing']:
        try:
            field_span(bad, 'association')
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid JSON field contract accepted')
    try:
        append_literal(line, {('d', 'missing', 'HP:0000001')})
    except AssertionError:
        pass
    else:
        raise AssertionError('unknown patient accepted')
    return dict(status='PASS_SYNTHETIC_LITERAL_AND_SET_CONTRACT', cases=6, no_real_input_or_gold_read=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    assert args.plan and args.output and args.output.is_dir()
    signal.alarm(30)
    result = execute(args.plan.resolve(), args.output.resolve())
    signal.alarm(0)
    print(json.dumps(dict(status=result['status'], keep=result['keep'], seconds=result['seconds'])))


if __name__ == '__main__':
    main()
