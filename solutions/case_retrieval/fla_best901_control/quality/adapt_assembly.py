"""Mechanically expose identity gates of frozen native assembly; policy stays exact."""
import ast
import copy
import hashlib
from collections import Counter
from common import NATIVE_SOURCE_SHA, checked, module, sha


def dump(node): return ast.dump(node, annotate_fields=True, include_attributes=False)


def adapted_tree(path):
    assert sha(path) == NATIVE_SOURCE_SHA
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assemble')
    node = copy.deepcopy(original)
    node.args.args += [ast.arg(arg='verify_fingerprint'), ast.arg(arg='verify_decisions')]
    replaced_fingerprint = 0; replaced_assertions = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fingerprint_native':
            n.func.id = 'verify_fingerprint'; replaced_fingerprint += 1
    expected_assertions = [
        "assert decisions == json.loads(refs['patient_saved_decisions'].read_text())",
        "assert dict(patient_stats) == manifests['patient']['codec_status_counts']"]
    replacements = {dump(ast.parse(s).body[0]) for s in expected_assertions}
    body = []
    for n in node.body:
        if isinstance(n, ast.Assert) and dump(n) in replacements:
            replaced_assertions += 1
            if replaced_assertions == 1:
                body.append(ast.parse("verify_decisions(decisions, patient_stats, manifests['patient'])").body[0])
        else: body.append(n)
    node.body = body
    assert replaced_fingerprint == 1 and replaced_assertions == 2
    # Invert the three identity-only changes and demand complete original AST equality.
    inverse = copy.deepcopy(node); inverse.args.args = inverse.args.args[:-2]
    for n in ast.walk(inverse):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_fingerprint':
            n.func.id = 'fingerprint_native'
    restored = []
    for n in inverse.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'verify_decisions':
            restored += [ast.parse(s).body[0] for s in expected_assertions]
        else: restored.append(n)
    inverse.body = restored
    assert dump(inverse) == dump(original), 'Any non-identity algorithm change is forbidden'
    ast.fix_missing_locations(node)
    proof = dict(original_function_AST_sha256=hashlib.sha256(dump(original).encode()).hexdigest(),
                 adapted_function_AST_sha256=hashlib.sha256(dump(node).encode()).hexdigest(),
                 inverse_AST_exact=True, changed_calls=1, old_identity_assertions_replaced=2,
                 relation_policy_or_order_changes=0)
    return node, proof


def run(plan, verified):
    source = checked(plan['native_assembly_source']); node, proof = adapted_tree(source)
    native = module('frozen_native901_assembly_pure', source)
    scope = dict(native.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source)+'[identity-parameters]', 'exec'), scope)
    refs = {k: checked(v) for k, v in plan['assembly_references'].items()}
    sources = {k: checked(v) for k, v in plan['pure_sources'].items()}
    patient_observations = {}
    def fingerprint(arm, actual_refs, count):
        evidence = verified['flows']['public-new' if arm == 'public' else arm]
        assert evidence['rows'] == count
        assert sha(actual_refs[arm + '_responses']) == evidence['responses']['sha256']
        assert sha(actual_refs[arm + '_prompts']) == evidence['prompts']['sha256']
        return evidence
    def decisions(observed, statuses, evidence):
        assert len(observed) == evidence['rows'] == 64
        assert [x['row_id'] for x in observed] == evidence['ordered_row_ids']
        assert dict(Counter(x['status'] for x in observed)) == {k:v for k,v in statuses.items() if v}
        assert set(statuses) == {'complete', 'partial', 'abstain'}
        patient_observations.update(status_counts=dict(statuses), decisions=observed,
                                    comparison_to_native_decisions=False)
    result = scope['assemble'](refs, native.helpers(sources), fingerprint, decisions)
    return result, proof, patient_observations
