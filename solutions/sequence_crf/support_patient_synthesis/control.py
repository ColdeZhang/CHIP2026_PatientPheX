"""One declared CPU synthesis: accepted support baseline plus six patient deltas.

Both constituent endpoints were known before this development synthesis.
No training, model inference, response parsing, search, or old reevaluation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from datetime import datetime, timezone


COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
METRICS = ('menF1', 'docF1', 'micF1', 'macF1', 'Score')


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check(value):
    assert ref(value['path']) == value
    return Path(value['path'])


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def publish(path, raw):
    path = Path(path); temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, path); temporary.unlink()
    for directory in (path.parent, path.parent.parent):
        descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def load_module(name, value):
    path = check(value)
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module; specification.loader.exec_module(module)
    return module


def keys(documents):
    result = set()
    for document in documents:
        patients = set()
        for association in document['association']:
            pid = association['patient_id']; assert pid not in patients
            patients.add(pid)
            for value in association['phenotype']:
                assert isinstance(value, str)
                concepts = value.split(';') if COMPOUND.fullmatch(value) else [value]
                result.update((str(document['pmc_id']), pid, concept) for concept in concepts)
    return result


def load_plan(path):
    plan = read(path)
    assert plan['schema'] == 'accepted-support-plus-six-patient-deltas-v1'
    assert plan['source'] == ref(__file__)
    assert plan['new_quality_calls'] == 1 and plan['baseline_quality_calls'] == 0
    assert plan['baseline_score'] == 0.7542169692819372
    assert plan['patient_deltas'] == 6 and plan['no_NN_or_search']
    for value in plan['references'].values():
        assert Path(value['path']).resolve() != Path(plan['gold']['path']).resolve()
        check(value)
    refs = plan['references']
    assert refs['base_pred']['sha256'] == '3049d9fe20ea84aeed448ab12e80bd0d8cf351cdf087c06dec6375103969571f'
    assert refs['patient_pred']['sha256'] == '765cd5a0614f5e3e9155607bcdff00262de3e6d937e3936954d8c58fe4a4549c'
    assert refs['old_a179']['sha256'] == 'a179a5de5b7271979a0fba851afc269d63125248a56681674e7388d0c6728c39'
    baseline = read(refs['base_metrics']['path'])
    assert baseline['metrics']['Score'] == plan['baseline_score']
    base_freeze = read(refs['base_freeze']['path'])
    assert base_freeze['prediction'] == refs['base_pred']
    assert base_freeze['retained_keys'] == refs['public54']
    assert base_freeze['support_delta'] == refs['support_delta']
    patient_freeze = read(refs['patient_freeze']['path'])
    assert patient_freeze['prediction'] == refs['patient_pred']
    assert patient_freeze['references']['baseline_pred'] == refs['old_a179']
    return plan, baseline, load_module('accepted_literal_helper', refs['literal_helper'])


def assemble(path, output, plan, baseline, helper, started):
    assert not output.exists()
    refs = plan['references']
    lines, base, ids = helper.document_rows(refs['base_pred']['path'])
    _, old, old_ids = helper.document_rows(refs['old_a179']['path'])
    _, patient, patient_ids = helper.document_rows(refs['patient_pred']['path'])
    _, inputs, input_ids = helper.document_rows(refs['input']['path'])
    assert ids == old_ids == patient_ids == input_ids and len(ids) == 16
    old_keys, patient_keys, base_keys = keys(old), keys(patient), keys(base)
    assert len(old_keys) == 391 and old_keys <= patient_keys and len(patient_keys) == 397
    assert len(base_keys) == 410
    assert sum(len(d['entities']) for d in base) == 1061
    assert sum(len(d['association']) for d in base) == 36
    for before, after, current in zip(old, patient, base):
        assert {k:v for k,v in before.items() if k != 'association'} == {k:v for k,v in after.items() if k != 'association'}
        assert [a['patient_id'] for a in before['association']] == [a['patient_id'] for a in after['association']] == [a['patient_id'] for a in current['association']]
        for a, b in zip(before['association'], after['association']):
            assert b['phenotype'][:len(a['phenotype'])] == a['phenotype']
            assert {k:v for k,v in a.items() if k != 'phenotype'} == {k:v for k,v in b.items() if k != 'phenotype'}
    six = patient_keys - old_keys; assert len(six) == plan['patient_deltas']
    added_log = read(refs['patient_added']['path'])
    assert six == {(str(x['pmc_id']), x['patient_id'], x['phenotype']) for x in added_log}
    public54 = {tuple(x) for x in read(refs['public54']['path'])}
    delta = read(refs['support_delta']['path'])
    removed = {tuple(x) for x in delta['removed']}
    added_support = {tuple(x) for x in delta['added']}
    assert len(public54) == 54 and public54 <= base_keys
    assert len(removed) == 2 and len(added_support) == 21
    assert old_keys - base_keys == removed and base_keys - old_keys == added_support
    assert not six & removed and not base_keys & removed
    actual_new = six - base_keys
    # Reuse the audited literal-span splice. Existing bytes outside association
    # stay exact; phenotype lists keep their existing literal prefix and metadata.
    combined_lines = [helper.append_literal(line, actual_new) for line in lines]
    combined = [json.loads(line) for line in combined_lines]
    result_keys = keys(combined)
    assert result_keys == base_keys | six and not result_keys & removed
    assert public54 <= result_keys
    for before, after in zip(base, combined):
        assert {k:v for k,v in before.items() if k != 'association'} == {k:v for k,v in after.items() if k != 'association'}
        assert len(before['association']) == len(after['association'])
        for a, b in zip(before['association'], after['association']):
            assert b['phenotype'][:len(a['phenotype'])] == a['phenotype']
            assert {k:v for k,v in a.items() if k != 'phenotype'} == {k:v for k,v in b.items() if k != 'phenotype'}
    output.mkdir(parents=True)
    publish(output/'prediction.jsonl', ''.join(combined_lines).encode())
    report = dict(at_utc=now(), documents=16, entities=1061, patients=36,
        baseline_relations=410, patient_original_a179_relations=391, patient_union_relations=397,
        six_delta_keys=sorted(six), overlap_with_public54=sorted(six & public54),
        overlap_with_baseline=sorted(six & base_keys), actual_new_keys=sorted(actual_new),
        actual_new_count=len(actual_new), final_relations=len(result_keys),
        removed_keys_kept_absent=sorted(removed), removed_relations=0,
        public54_all_preserved=True, all410_baseline_relations_preserved=True,
        baseline_nonassociation_literal_bytes_exact=True, old_phenotype_prefix_and_metadata_exact=True,
        patient_old391_preserved_in_original_patient_arm=True, gold_read=False,
        model_or_response_parser_calls=0, seconds=time.monotonic()-started)
    save(output/'assembly-report.json', report)
    save(output/'freeze.json', dict(at_utc=now(), source=ref(__file__), plan=ref(path),
        references=refs, prediction=ref(output/'prediction.jsonl'), assembly=ref(output/'assembly-report.json'),
        full16_prediction_complete=True, gold_read=False, new_quality_calls_before_freeze=0,
        only_quality_arm='accepted_3049_support_plus_patient_six_delta'))
    print(json.dumps(dict(freeze=ref(output/'freeze.json'), assembly=report)))


def evaluate(path, output, plan, baseline, helper, started):
    frozen = read(output/'freeze.json'); refs = plan['references']
    assert frozen['source'] == ref(__file__) and frozen['plan'] == ref(path)
    assert frozen['references'] == refs and frozen['full16_prediction_complete']
    assert not frozen['gold_read'] and frozen['new_quality_calls_before_freeze'] == 0
    assert frozen['only_quality_arm'] == 'accepted_3049_support_plus_patient_six_delta'
    check(frozen['prediction']); check(frozen['assembly'])
    evaluation = output/'evaluation'; evaluation.mkdir(exist_ok=False)
    save(evaluation/'before-gold.json', dict(at_utc=now(), freeze=ref(output/'freeze.json'),
        gold_not_yet_read=True, new_calls=1, baseline_calls=0))
    # evaluate.py imports this exact module name; bind it before the import.
    validator = load_module('validate_predictions', refs['validator'])
    evaluator = load_module('single_synthesis_evaluator', refs['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    assert evaluator.COMPOUND_HPO.pattern == COMPOUND.pattern
    inputs = validator.load_jsonl(refs['input']['path'])
    prediction = validator.load_jsonl(frozen['prediction']['path'])
    ontology = validator.read_hpo(refs['hpo']['path'])
    assert ontology['version'] == 'hp/releases/2026-06-23'
    validation = validator.validate_predictions(inputs, prediction, ontology)
    save(evaluation/'validation.json', validation); assert validation['valid']
    check(plan['gold'])  # First new synthesis gold bytes only after complete freeze.
    gold = validator.load_jsonl(plan['gold']['path'])
    metrics = evaluator.evaluate_report(gold, prediction, 'all', 'text')
    save(evaluation/'metrics.json', metrics)
    assembly = read(frozen['assembly']['path']); new = {tuple(x) for x in assembly['actual_new_keys']}
    truth = keys(gold)
    delta = {k:metrics['metrics'][k]-baseline['metrics'][k] for k in METRICS}
    assert delta['menF1'] == delta['docF1'] == 0
    result = dict(completed_at_utc=now(), baseline_metrics=baseline['metrics'], metrics=metrics['metrics'],
        delta=delta, actual_new_relations=len(new), added_tp=len(new & truth), added_fp=len(new-truth),
        removed_relations=0, candidate_best=delta['Score']>0, macro_decreased=delta['macF1']<0,
        numeric_target_met=metrics['metrics']['Score']>=0.8, target_accepted=False,
        target_rule='Score>=0.8 plus independent format, provenance and reproducibility review',
        prediction=frozen['prediction'], freeze=ref(output/'freeze.json'), plan=ref(path),
        source=ref(__file__), validation=ref(evaluation/'validation.json'),
        new_evaluator_calls=1, baseline_evaluator_calls=0, patient_alone_evaluator_calls=0,
        new_fit_or_model_calls=0, no_threshold_or_combination_search=True,
        seconds=time.monotonic()-started, scope=plan['scope'])
    save(evaluation/'result.json', result); print(json.dumps(result))


def main():
    if not __debug__: raise RuntimeError('Contract assertions must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['assemble', 'evaluate'])
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU synthesis cap exceeded')))
    signal.alarm(30)
    plan, baseline, helper = load_plan(args.plan)
    output = Path(plan['output_directory']).resolve()
    {'assemble':assemble, 'evaluate':evaluate}[args.mode](args.plan, output, plan, baseline, helper, started)


if __name__ == '__main__':
    main()
