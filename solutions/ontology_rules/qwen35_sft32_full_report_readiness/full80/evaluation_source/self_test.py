#!/usr/bin/env python3
"""Bounded pure synthetic contracts; no real artifacts, gold, models or evaluator."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import time

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sft32_eval_synthetic', Path(__file__).with_name('evaluate_full80.py'))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def reject(action):
    try:
        action()
    except (ValueError, KeyError, FileExistsError):
        return
    raise AssertionError('Bad synthetic input accepted')


def main():
    started = time.monotonic()
    p = {'row_id': 'S:relations:0', 'pmc_id': 'S', 'code_map': {
        'C000': 'HP:0000001;HP:0000002', 'C001': 'NIL;literal', 'C002': 'NIL'},
        'patient_ids': ['P1', 'OII-1']}
    raw = {'row_id': p['row_id'], 'pmc_id': 'S', 'response': 'intentionally not parsed',
        'input_tokens': 100, 'output_tokens': 10, 'natural_eos': True, 'hit_token_limit': False,
        'removed_demonstrations': 0, 'batch_seconds': .1}
    c = {'row_id': p['row_id'], 'pmc_id': 'S', 'strict_complete': True, 'abstention': None,
        'candidate_patients': {'C000': ['P1'], 'C001': ['OII-1'], 'C002': []}, 'provenance': 'fresh'}
    mapping = e.canonical_gate(p, raw, c)
    require = e.require
    require(mapping == c['candidate_patients'], 'Canonical mapping changed')
    for field, value in [('candidate_patients', {'C000': ['P1']}),
                         ('candidate_patients', {'C000': ['unknown'], 'C001': [], 'C002': []}),
                         ('candidate_patients', {'C000': ['P1', 'P1'], 'C001': [], 'C002': []})]:
        broken = copy.deepcopy(c); broken[field] = value
        reject(lambda: e.canonical_gate(p, raw, broken))
    capped = dict(raw, natural_eos=False, hit_token_limit=True, output_tokens=1600)
    reject(lambda: e.canonical_gate(p, capped, c))
    invalid = dict(c, strict_complete=False, candidate_patients={}, abstention='ValueError: cap')
    require(e.canonical_gate(p, capped, invalid) is None, 'Invalid whole row did not abstain')
    reject(lambda: e.canonical_gate(p, capped, dict(invalid, candidate_patients={'C000': []})))
    reject(lambda: json.loads('{"C000":[],"C000":["P1"]}', object_pairs_hook=e.unique_object))
    old = ' {"pmc_id":"S", "pmid":null, "entities":[{"text":"\\u4e2d"}], "association" : [{"patient_id":"P1","phenotype":["OLD"]},{"patient_id":"OII-1","phenotype":[]}], "extra":"é"}\r\n'
    inputs = [{'pmc_id': 'S', 'patient': [{'patient_id': 'P1'}, {'patient_id': 'OII-1'}], 'full_text': []}]
    documents, stats = e.make_primary([old], inputs, [p], [mapping])
    alone, union, counts = e.assemble_two([old], documents)
    require(e.nonrelation_bytes(alone) == e.nonrelation_bytes(union) == e.nonrelation_bytes([old]),
            'Escaped Unicode/whitespace/CRLF nonrelation bytes changed')
    require(e.relation_map(json.loads(alone[0])) == {'P1': {'HP:0000001;HP:0000002'}, 'OII-1': {'NIL;literal'}},
            'Literal compounds/NIL incorrectly split or dropped')
    require(e.relation_map(json.loads(union[0]))['P1'] == {'OLD', 'HP:0000001;HP:0000002'}, 'Union differs')
    # The established public evaluator only splits exact HP compounds; output itself remains literal.
    hp_compound = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
    require(bool(hp_compound.fullmatch('HP:0000001;HP:0000002')) and
            not hp_compound.fullmatch('NIL;literal') and not hp_compound.fullmatch('HP:0000001;NIL'),
            'Compound boundary fixture differs')
    reject(lambda: e.make_primary([old], inputs, [p, p], [mapping, mapping]))
    reject(lambda: e.make_primary([old], inputs, [p], []))
    reject(lambda: e.make_primary([old], [dict(inputs[0], patient=[{'patient_id': 'P1'}])], [p], [mapping]))
    broken_p = copy.deepcopy(p); broken_p['code_map']['C002'] = broken_p['code_map']['C000']
    reject(lambda: e.make_primary([old], inputs, [broken_p], [mapping]))
    empty_documents, _ = e.make_primary([old], inputs, [p], [None])
    require(all(not a['phenotype'] for a in empty_documents[0]['association']), 'Invalid-row fallback appeared')
    for bad in ['{"association":[],"association":[]}', '{"a":1}', '{"association":{}}']:
        reject(lambda bad=bad: e.association_span(bad))
    require(e.retain({'micF1': .6, 'macF1': .5}, {'micF1': .5, 'macF1': .5}) and
            not e.retain({'micF1': .7, 'macF1': .4}, {'micF1': .5, 'macF1': .5}), 'Retention boundary differs')
    # A disabled plan with inaccessible references must fail before touching any of them.
    reject(lambda: e.reference_gate({'execution_ready': False}))
    # Run the real common-freeze function on synthetic files and inspect BOTH stored outputs.
    with tempfile.TemporaryDirectory(prefix='sft32-full-report-synthetic-') as temporary:
        root = Path(temporary); plan_path = root / 'synthetic-plan.json'; plan_path.write_text('{}')
        old_path = root / 'old.jsonl'; old_path.write_bytes(old.encode('utf-8'))
        target = root / 'paired'
        frozen, freeze_ref = e.freeze_pair(target, e.ref(plan_path), {'inference_references': {}}, {'old_baseline_pred': e.ref(old_path)}, {},
            [old], documents, {'rows': 1}, stats, [])
        freeze = e.read(freeze_ref['path'])
        require(set(freeze['endpoints']) == set(e.ENDPOINTS) and freeze['gold_read'] is False and
                freeze['new_quality_calls'] == 0 and all(Path(frozen[n]['prediction']['path']).exists() for n in e.ENDPOINTS),
                'Joint freeze did not contain both complete outputs before quality')
        reject(lambda: e.save(plan_path, {'overwrite': True}))
    # Static ordering guard executes no quality helper. It specifically covers H50 and first gold access.
    tree = ast.parse(Path(e.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    line = {}
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'freeze_pair': line['freeze'] = node.lineno
            if node.func.id == 'module' and node.args and isinstance(node.args[0], ast.Constant):
                if node.args[0].value == 'validate_predictions': line['validator'] = node.lineno
                if node.args[0].value == 'sft32_full_report_fixed_evaluator': line['evaluator'] = node.lineno
            if node.func.id == 'sha' and node.args and isinstance(node.args[0], ast.Subscript):
                if isinstance(node.args[0].slice, ast.Constant) and node.args[0].slice.value == 'gold_path':
                    line['first_gold_hash'] = node.lineno
    require(line['freeze'] < line['validator'] < line['evaluator'] < line['first_gold_hash'], 'Freeze/import/gold order differs')
    elapsed = time.monotonic() - started; require(elapsed < 60, 'Synthetic CPU budget exceeded')
    result = {'status': 'PASS_pure_synthetic_literal_coverage_abstention_pair_freeze_H50',
        'elapsed_CPU_seconds': elapsed, 'real_artifacts_or_gold_read': False,
        'model_tokenizer_or_quality_executed': False, 'new_quality_calls': 0,
        'control': e.ref(e.__file__), 'helpers': e.ref(e.helper_path), 'self_test': e.ref(__file__)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
