"""One fixed document-supported PCS relation union; local CPU, no learned gate."""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HPO = re.compile(r"HP:[0-9]{7}")
FORMULA = "baseline UNION document_supported(geometry, task1)"


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def pairs(rows, ev):
    return {(doc['pmc_id'], pid, concept) for doc in rows
            for pid, concepts in ev.associations(doc).items() for concept in concepts}


def supported(concept, document_concepts):
    # Full valid compound relations have already been split by public v2.
    # No alt-ID, synonym, case, whitespace or ontology ancestry normalization.
    kind = 'hpo' if HPO.fullmatch(concept) else 'unlinked_text'
    return (kind, concept) in document_concepts


def add_relations(base, added, ev):
    predicted = copy.deepcopy(base)
    for doc in predicted:
        for association in doc['association']:
            association['phenotype'].extend(sorted(
                concept for pmc, pid, concept in added
                if pmc == doc['pmc_id'] and pid == association['patient_id']))
    assert pairs(predicted, ev) == pairs(base, ev) | added
    for old, new in zip(base, predicted):
        assert {k: v for k, v in old.items() if k != 'association'} == {
            k: v for k, v in new.items() if k != 'association'}
        assert len(old['association']) == len(new['association'])
        for a, b in zip(old['association'], new['association']):
            assert b['phenotype'][:len(a['phenotype'])] == a['phenotype']
            assert {k: v for k, v in a.items() if k != 'phenotype'} == {
                k: v for k, v in b.items() if k != 'phenotype'}
    return predicted


def contracts(ev):
    def entity(identifier, text, note=None):
        return dict(identifier=identifier, text=text, note=note)
    doc = dict(pmc_id='toy', untouched={'a': [3]}, entities=[
        entity('HP:0000001;HP:0000002', 'linked'),
        entity('HP:0000003', 'excluded', 'NO'),
        entity('-1', 'Exact; surface'),
        entity('-1', 'Case'),
        entity('MESH:D000001', 'mesh'),
    ], association=[
        dict(patient_id='p1', extra={'x': 1}, phenotype=['HP:0000001;HP:0000009', 'Old; NIL']),
        dict(patient_id='p2', phenotype=[]),
    ])
    concepts = ev.document_concepts(doc, doc, 'all', 'text')
    checks = {}
    def check(name, condition):
        assert condition, name
        checks[name] = True
    check('compound_entity_support_splits', supported('HP:0000001', concepts) and supported('HP:0000002', concepts))
    check('NO_entity_not_support', not supported('HP:0000003', concepts))
    check('no_parent_or_altid_expansion', not supported('HP:0000004', concepts))
    check('NIL_semicolon_exact', supported('Exact; surface', concepts))
    check('NIL_no_substring_or_whitespace_repair', not supported('Exact', concepts) and not supported('Exact; surface ', concepts))
    check('NIL_case_sensitive', supported('Case', concepts) and not supported('case', concepts))
    check('non_HPO_id_not_HPO_support', not supported('MESH:D000001', concepts))
    check('unlinked_id_literal_not_surface_support', not supported('-1', concepts))
    relation_doc = dict(association=[dict(patient_id='p', phenotype=[
        'HP:0000001;HP:0000002', 'HP:0000001', 'Exact; surface',
        'HP:0000001;NIL', 'HP:0000001; HP:0000002'])])
    values = ev.associations(relation_doc)['p']
    check('valid_compound_split_and_dedup', {'HP:0000001', 'HP:0000002'} <= values and 'HP:0000001;HP:0000002' not in values)
    check('mixed_malformed_compounds_remain_literal', {'HP:0000001;NIL', 'HP:0000001; HP:0000002', 'Exact; surface'} <= values)
    check('mixed_not_partial_HPO_support', not supported('HP:0000001;NIL', concepts))
    mixed_nil = ev.document_concepts(dict(entities=[entity('-1', 'HP:0000001;NIL')]), {}, 'all', 'text')
    check('mixed_can_only_match_exact_NIL_surface', supported('HP:0000001;NIL', mixed_nil))
    new = add_relations([doc], {('toy', 'p1', 'HP:0000002'), ('toy', 'p2', 'HP:0000002')}, ev)
    check('support_not_patient_reassignment', ('toy', 'p1', 'HP:0000002') in pairs(new, ev) and ('toy', 'p2', 'HP:0000002') in pairs(new, ev))
    check('full_baseline_compound_NIL_metadata_preserved', doc['association'][0]['phenotype'] == ['HP:0000001;HP:0000009', 'Old; NIL'] and new[0]['association'][0]['phenotype'][:2] == doc['association'][0]['phenotype'])
    return dict(status='PASS', count=len(checks), checks=checks, gold_read=False, NN_calls=0)


def run(plan_path, mode):
    plan = json.loads(Path(plan_path).read_text())
    refs, out = plan['refs'], Path(plan['output'])
    assert plan['fold'] == 1 and plan['formula'] == FORMULA
    assert refs['source'] == ref(__file__)
    checked(refs['decision'])
    gold_path = os.path.realpath(refs['gold']['path'])
    allow_gold = False
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if path.endswith('.gold.jsonl') and (path != gold_path or not allow_gold):
                raise PermissionError('Held gold blocked until frozen sole endpoint')
    sys.addaudithook(guard)
    evaluator = checked(refs['evaluator'])
    checked(refs['validator'])
    sys.path.insert(0, str(evaluator.parent))
    ev = importlib.import_module('evaluate')
    va = importlib.import_module('validate_predictions')
    assert ev.VERSION == 'patientphex-public-approx-v2'
    if mode == 'contracts':
        print(json.dumps(contracts(ev), ensure_ascii=False))
        return
    if mode == 'assemble':
        out.mkdir(parents=True, exist_ok=False)
        put(out / 'started.json', dict(at_utc=now(), plan=ref(plan_path)))
        put(out / 'contracts.json', contracts(ev))
        inputs = va.load_jsonl(checked(refs['input']))
        base = va.load_jsonl(checked(refs['baseline']))
        geometry = va.load_jsonl(checked(refs['geometry']))
        task1 = va.load_jsonl(checked(refs['task1']))
        by = {doc['pmc_id']: doc for doc in base}
        assert len(base) == len(by) == len(inputs) == len(geometry) == len(task1) == 16
        assert set(by) == {doc['pmc_id'] for doc in inputs}
        assert sum(len(doc['entities']) for doc in base) == plan['entities'] == 1189
        assert sum(len(doc['association']) for doc in base) == plan['patients'] == 39
        for source in (geometry, task1):
            assert {doc['pmc_id'] for doc in source} == set(by)
            for doc in source:
                old = by[doc['pmc_id']]
                assert {k: v for k, v in doc.items() if k not in ('entities', 'association')} == {
                    k: v for k, v in old.items() if k not in ('entities', 'association')}
                assert ev.associations(doc).keys() == ev.associations(old).keys()
        old, supplied = pairs(base, ev), pairs(geometry, ev)
        assert len(old) == plan['baseline_relations'] == 524
        assert len(supplied) == plan['geometry_relations'] == 190
        support = {doc['pmc_id']: ev.document_concepts(doc, by[doc['pmc_id']], 'all', 'text') for doc in task1}
        accepted = {(pmc, pid, concept) for pmc, pid, concept in supplied if supported(concept, support[pmc])}
        added = accepted - old
        audit = [dict(pmc_id=pmc, patient_id=pid, concept=concept,
                      supported=supported(concept, support[pmc]), already_baseline=(pmc, pid, concept) in old,
                      added=(pmc, pid, concept) in added) for pmc, pid, concept in sorted(supplied)]
        put(out / 'support-audit.json', audit)
        predicted = add_relations(base, added, ev)
        valid = va.validate_predictions(inputs, predicted, va.read_hpo(checked(refs['hpo'])))
        put(out / 'format.json', valid)
        assert valid['valid']
        prediction_path = out / 'prediction.jsonl'
        with prediction_path.open('xb') as stream:
            if not added:
                stream.write(checked(refs['baseline']).read_bytes())
            else:
                for doc in predicted:
                    stream.write((json.dumps(doc, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        artifacts = {name: ref(out / name) for name in ('contracts.json', 'support-audit.json', 'format.json', 'prediction.jsonl')}
        put(out / 'freeze.json', dict(at_utc=now(), plan=ref(plan_path), artifacts=artifacts,
            added_pairs=sorted(added), added=len(added), geometry_relations=len(supplied),
            supported_geometry=len(accepted), baseline_relations=len(old), final_relations=len(old | added),
            docs=16, patients=39, entities=1189, nonassociation_objects_exact=True,
            baseline_relation_lists_prefix_exact=True, gold_read=False, NN_calls=0, fit_calls=0))
        return
    freeze = json.loads((out / 'freeze.json').read_text())
    assert freeze['plan'] == ref(plan_path)
    for item in freeze['artifacts'].values():
        checked(item)
    prior_report = json.loads(checked(refs['baseline_metrics']).read_text())
    prior = prior_report['metrics']
    assert prior_report['prediction_sha256'] == refs['baseline']['sha256']
    assert prior_report['gold_sha256'] == refs['gold']['sha256']
    assert prior['Score'] == plan['baseline_score']
    if not freeze['added']:
        assert checked(freeze['artifacts']['prediction.jsonl']).read_bytes() == checked(refs['baseline']).read_bytes()
        put(out / 'result.json', dict(status='BYTEEXACT_ZERO_INCREMENT_ZERO_GOLD_ZERO_V2', at_utc=now(),
            baseline=prior, retain=False, new_v2=0, old_v2=0, freeze=ref(out / 'freeze.json')))
        return
    put(out / 'evaluation-intent.json', dict(at_utc=now(), freeze=ref(out / 'freeze.json'), new_v2=1, old_v2=0))
    allow_gold = True
    gold = va.load_jsonl(checked(refs['gold']))
    prediction = va.load_jsonl(freeze['artifacts']['prediction.jsonl']['path'])
    report = ev.evaluate_report(gold, prediction, body_scope='all', no_id_doc='text')
    report.update(prediction_sha256=freeze['artifacts']['prediction.jsonl']['sha256'], gold_sha256=refs['gold']['sha256'])
    put(out / 'metrics.json', report)
    delta = {key: report['metrics'][key] - value for key, value in prior.items()}
    assert delta['menF1'] == delta['docF1'] == 0
    added = set(map(tuple, freeze['added_pairs']))
    tp = len(added & pairs(gold, ev))
    put(out / 'result.json', dict(status='COMPLETE_ONE_FIXED_F1_ENDPOINT', at_utc=now(), fold=1,
        baseline=prior, metrics=report['metrics'], delta=delta, added=len(added), added_TP=tp,
        added_FP=len(added)-tp, retain=delta['Score'] > 0, new_v2=1, old_v2=0,
        NN_calls=0, fit_calls=0, original_dev_read=False, F0_gold_read=False,
        development_not_blind=True, freeze=ref(out / 'freeze.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('mode', choices=['contracts', 'assemble', 'evaluate'])
    args = parser.parse_args()
    run(args.plan, args.mode)
