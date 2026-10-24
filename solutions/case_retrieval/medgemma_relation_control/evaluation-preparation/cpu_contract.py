#!/usr/bin/env python3
"""Small synthetic relation-union/byte-preservation and once-only gate checks."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'evaluate_medgemma.py'
spec = importlib.util.spec_from_file_location('med_cpu_contract', SOURCE)
endpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endpoint)


def main():
    started, cases = time.monotonic(), []
    def reject(name, function):
        try:
            function()
        except ValueError:
            cases.append(name)
        else:
            raise AssertionError(name)
    raw = '{ "pmc_id" : "synthetic", "entities" : [{"text":"\\u4e2d", "span": [0,1]}], "association" : [{"patient_id":"P","phenotype":["A","B"]}], "extra": "literal space" }\n'
    base = json.loads(raw)
    med = copy.deepcopy(base)
    med['association'][0]['phenotype'] = ['B', 'C']
    primary, union, counts = endpoint.assemble_two([raw], [med])
    assert json.loads(primary[0])['association'] == med['association']
    assert json.loads(union[0])['association'][0]['phenotype'] == ['A', 'B', 'C']
    assert counts == {'old': 2, 'med': 2, 'intersection': 1, 'union': 3}
    cases.append('overlap_union_is_literal_set_union')
    assert endpoint.nonrelation_bytes(primary) == endpoint.nonrelation_bytes(union) == endpoint.nonrelation_bytes([raw])
    assert '\\u4e2d' in primary[0] and '"extra": "literal space"' in union[0]
    cases.append('all_nonassociation_bytes_whitespace_escapes_exact')
    empty = copy.deepcopy(base)
    empty['association'][0]['phenotype'] = []
    assert json.loads(endpoint.assemble_two([raw], [empty])[1][0])['association'] == base['association']
    cases.append('empty_med_adds_nothing')
    bad = copy.deepcopy(med); bad['association'][0]['phenotype'] = ['B', 'B']
    reject('duplicate_phenotype_rejected', lambda: endpoint.assemble_two([raw], [bad]))
    bad = copy.deepcopy(med); bad['association'].append(copy.deepcopy(bad['association'][0]))
    reject('duplicate_patient_rejected', lambda: endpoint.assemble_two([raw], [bad]))
    bad = copy.deepcopy(med); bad['association'][0]['patient_id'] = 'other'
    reject('patient_coverage_change_rejected', lambda: endpoint.assemble_two([raw], [bad]))
    reject('duplicate_document_rejected', lambda: endpoint.assemble_two([raw, raw], [med, med]))
    bad = copy.deepcopy(med); bad['entities'][0]['text'] = 'changed'
    reject('entity_change_rejected', lambda: endpoint.assemble_two([raw], [bad]))
    bad = copy.deepcopy(med); bad['association'][0]['confidence'] = .5
    reject('extra_association_field_rejected', lambda: endpoint.assemble_two([raw], [bad]))
    reject('duplicate_top_level_key_rejected', lambda: endpoint.association_span('{"association":[],"association":[]}'))
    baseline = {'micF1': .5, 'macF1': .5}
    assert endpoint.retain({'micF1': .51, 'macF1': .5}, baseline)
    assert not endpoint.retain(baseline, baseline)
    assert not endpoint.retain({'micF1': .55, 'macF1': .49}, baseline)
    cases.append('strict_relation_sum_and_nondecreasing_macro')
    w = endpoint.utilities()
    with tempfile.TemporaryDirectory(prefix='med-eval-gate-') as temporary:
        gate = Path(temporary) / 'evaluation-reservation.json'
        w.save(gate, {'synthetic': True})
        try:
            w.save(gate, {'synthetic': 'second'})
        except FileExistsError:
            cases.append('once_only_reservation_refuses_duplicate')
        else:
            raise AssertionError('Duplicate reservation accepted')
    tree = ast.parse(SOURCE.read_text())
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    calls = [node for node in ast.walk(main_node) if isinstance(node, ast.Call)]
    freezes = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'save'
               and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'freeze_path']
    gold_reads = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'sha'
                  and node.args and isinstance(node.args[0], ast.Subscript)
                  and isinstance(node.args[0].slice, ast.Constant) and node.args[0].slice.value == 'gold_path']
    assert len(freezes) == len(gold_reads) == 1 and freezes[0].lineno < gold_reads[0].lineno
    assert endpoint.ENDPOINTS == ('primary-medgemma', 'secondary-union')
    commands = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'command']
    assert len(commands) == 3  # One parser site, one two-endpoint format loop, one two-endpoint v2 loop.
    evaluator = [node for node in commands if '--gold' in ast.unparse(node)]
    assert len(evaluator) == 1 and evaluator[0].lineno > gold_reads[0].lineno
    cases.append('AST_two_frozen_endpoints_precede_unique_v2_loop_and_gold')
    result = {'status': 'PASS', 'source_sha256': w.sha(SOURCE), 'harness_sha256': w.sha(__file__),
              'cases': cases, 'count': len(cases), 'seconds': time.monotonic() - started,
              'actual_gold_reads': 0, 'actual_parser_calls': 0, 'actual_quality_calls': 0,
              'actual_model_loads': 0, 'actual_network_calls': 0,
              'scope': 'Synthetic union and byte/gate tests plus source-order inspection; no full endpoint/model simulation'}
    w.save(ROOT / 'cpu-contract.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
