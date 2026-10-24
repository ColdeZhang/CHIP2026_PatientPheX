"""Exact cached16 reproduction of accepted3049; no model, gold or evaluator access."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(path, specification):
    actual = ref(path)
    assert actual['bytes'] == specification['bytes'] and actual['sha256'] == specification['sha256']
    return Path(path)


def load(name, path, specification):
    path = checked(path, specification)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'baseline', 'raw-keys', 'old-keys', 'support-helper',
                 'replacement-helper', 'expected-prediction', 'output-directory']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_directory.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING_CACHE_ONLY_REPRODUCTION', started_utc=datetime.now(timezone.utc).isoformat(),
                  source=ref(__file__), contract=ref(args.contract), gold_read=False,
                  fit_calls=0, model_predict_calls=0, NN_calls=0, quality_calls=0)
    def expired(*_):
        raise TimeoutError('Fixed30second cached reproduction limit')
    signal.signal(signal.SIGALRM, expired); signal.alarm(30)
    try:
        contract = json.loads(args.contract.read_text())
        assert contract['policy_id'] == 'accepted3049-frozen16-cache-only-byteexact-v1'
        assert contract['budget_seconds'] == 30
        checked(__file__, contract['source'])
        helper = load('support_helpers_cached3049', args.support_helper, contract['helpers']['support'])
        replacement = load('replace_subset_cached3049', args.replacement_helper, contract['helpers']['replacement'])
        baseline_path = checked(args.baseline, contract['inputs']['baseline'])
        raw_path = checked(args.raw_keys, contract['inputs']['raw_keys'])
        old_path = checked(args.old_keys, contract['inputs']['old_keys'])
        lines, baseline, ids = helper.document_rows(baseline_path)
        raw_list = json.loads(raw_path.read_text()); old_list = json.loads(old_path.read_text())
        raw, old = {tuple(x) for x in raw_list}, {tuple(x) for x in old_list}
        original = helper.associations(baseline)
        assert len(ids) == 16 and len(raw) == len(raw_list) == 145 and len(old) == len(old_list) == 35
        assert len(original) == 391 and original & raw == old and old <= raw
        assert sum(len(row['entities']) for row in baseline) == 1061
        protected = original - old; assert len(protected) == 356
        support = helper.entity_support(baseline)
        retained = {key for key in raw if (key[0], key[2]) in support}
        removed, added = old - retained, retained - old
        assert len(retained) == 54 and len(removed) == 2 and len(added) == 21 and not retained & protected
        payload = ''.join(replacement.replace_subset(helper, line, removed, added) for line in lines).encode('utf-8')
        prediction = output / 'supported.pred.jsonl'
        with prediction.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        _, predicted, predicted_ids = helper.document_rows(prediction)
        assert ids == predicted_ids and helper.associations(predicted) == protected | retained
        assert sum(len(row['entities']) for row in predicted) == 1061
        for before, after in zip(baseline, predicted):
            did = str(before['pmc_id'])
            assert len(before['association']) == len(after['association'])
            for old_row, new_row in zip(before['association'], after['association']):
                pid = old_row['patient_id']
                assert [value for value in old_row['phenotype'] if (did, pid, value) not in old] == [
                    value for value in new_row['phenotype'] if (did, pid, value) not in retained]
        # The target is read only after the output has been assembled and made durable.
        expected_path = checked(args.expected_prediction, contract['expected_prediction'])
        expected_bytes = expected_path.read_bytes()
        assert payload == expected_bytes
        checked(prediction, contract['expected_prediction'])
        directory_fd = os.open(str(output), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        report.update(status='PASS_COMPLETE16_LITERAL_BYTEEXACT_ACCEPTED3049_NO_MODEL_GOLD_OR_SCORE',
                      prediction=ref(prediction), compared_prediction=ref(expected_path),
                      actual_inputs=dict(baseline=ref(baseline_path), raw_keys=ref(raw_path), old_keys=ref(old_path)),
                      actual_helpers=dict(support=ref(args.support_helper), replacement=ref(args.replacement_helper)),
                      complete_bytes_equal=True, documents=16, entities=1061, relations=410,
                      protected_relations=356, raw=145, previous_supported=35, current_supported=54,
                      removed=2, added=21, literal_nonassociation_bytes_preserved=True,
                      protected_relation_values_and_order_preserved=True,
                      scope='Frozen16 cached baseline0e2c plus raw145 and original35 keys only; not raw text candidate/model/relationship generation or a self-contained full pipeline.')
    except BaseException as error:
        report.update(status='FAILED_CACHE_REPRODUCTION_PRESERVED_NO_RETRY', error_type=type(error).__name__,
                      error=str(error)[:1000])
        raise
    finally:
        signal.alarm(0)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
        save(output / 'reproduction.json', report)
        print(json.dumps({k: report.get(k) for k in ['status', 'seconds', 'complete_bytes_equal', 'error_type']}))


if __name__ == '__main__':
    main()
