"""Source-call tracing with stubbed scoring only; never evaluate predictions/gold."""
import ast
import json
import types
from common import B, ROUTE, S, checked, read, ref, save, verify_sources


def main():
    verify_sources()
    correction = read(ROUTE / 'evidence/F1-evaluation-count-addendum.json')
    for name in ('original_result_pointer', 'original_source', 'original_metrics', 'evaluator', 'original_prediction', 'original_freeze'):
        checked(correction[name])
    evaluator = ast.parse(checked(correction['evaluator']).read_text())
    report = next(n for n in evaluator.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_report')
    calls = []
    def fake_evaluate(*args, **kwargs):
        calls.append(dict(args=args, kwargs=kwargs))
        return {'metrics': {'stub': 0}}
    namespace = dict(evaluate=fake_evaluate, VERSION='source-trace-only', Any=object)
    exec(compile(ast.Module(body=[report], type_ignores=[]), 'original_report_stubbed', 'exec'), namespace)
    original_stub_report = namespace['evaluate_report']([], [], body_scope='all', no_id_doc='text')
    assert len(calls) == 5 and len(original_stub_report['sensitivity']) == 4
    actual_metrics = read(checked(correction['original_metrics']))
    assert set(actual_metrics['sensitivity']) == set(original_stub_report['sensitivity'])
    assert actual_metrics['metrics']['Score'] == correction['original_score']
    source = ast.parse((S / 'postprocess.py').read_text())
    evcalls = [n for n in ast.walk(source) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ev' and n.func.attr in ('evaluate', 'evaluate_report')]
    assert len(evcalls) == 1 and evcalls[0].func.attr == 'evaluate'
    kwargs = {n.arg: ast.literal_eval(n.value) for n in evcalls[0].keywords}
    assert kwargs == dict(body_scope='all', no_id_doc='text', wildcard_no_id=True, validate=True)
    calls.clear()
    expr = ast.Expression(body=evcalls[0]); ast.fix_missing_locations(expr)
    eval(compile(expr, 'F0_scoring_call_stubbed', 'eval'), dict(ev=types.SimpleNamespace(evaluate=fake_evaluate), gold=[], pred=[]))
    assert len(calls) == 1 and calls[0]['kwargs'] == kwargs
    metric_keys = {n.slice.value for n in ast.walk(source) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'delta' and isinstance(n.slice, ast.Constant)}
    assert metric_keys == {'menF1', 'docF1', 'Score'}
    evaluate = next(n for n in evaluator.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('evaluate', 'evaluate_report') for n in ast.walk(evaluate))
    compile(source, str(S / 'postprocess.py'), 'exec')
    save(S / 'evidence/call-count-contract.json', dict(status='PASS_STUB_ONLY_CALL_GRAPH_NO_QUALITY',
        postprocess=ref(S / 'postprocess.py'), source_manifest=ref(S / 'source-manifest-v2.json'),
        evaluator=correction['evaluator'], F1_count_correction=ref(ROUTE / 'evidence/F1-evaluation-count-addendum.json'),
        F1_underlying_calls_proved_by_stub=5, F1_fixed_sensitivity_calls=4,
        F0_evaluate_call_sites=1, F0_stub_invocations=1, F0_report_calls=0, F0_sensitivity_calls=0,
        F0_configuration=kwargs, original_metric_names_preserved=sorted(metric_keys),
        actual_evaluate_calls=0, gold_reads=0, NN=0, remote=0,
        prior_F0_source_history=ref(B / 'F0-source-preparation-v1-history/manifest.json')))
    print(json.dumps(ref(S / 'evidence/call-count-contract.json')))


if __name__ == '__main__': main()
