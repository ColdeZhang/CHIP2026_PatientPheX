"""Bind a separately authorized complete F0 run; no inference, gold or evaluation.

This source-only tool cannot invent a full58 budget/cache split or source SHA.
All unknowns must arrive in an explicit later owner-frozen binding document.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    template_path = base / 'evaluation-template.json'
    template = json.loads(template_path.read_text())
    binding_path = Path(args.binding).resolve()
    binding = json.loads(binding_path.read_text())
    assert template['execution_ready'] is False
    assert template['control_sha256'] == digest(base / 'evaluate_f0.py')
    assert set(binding) == {'template_sha256', 'endpoint_references', 'cached_rows',
                            'inference_contract', 'full_inference_authorization'}
    assert binding['template_sha256'] == digest(template_path)
    assert all(value is None for value in template['endpoint_references'].values())
    assert template['cached_rows'] is template['inference_contract'] is template['full_inference_authorization'] is None
    plan = copy.deepcopy(template)
    for name in ('endpoint_references', 'cached_rows', 'inference_contract', 'full_inference_authorization'):
        plan[name] = binding[name]
    plan['execution_ready'] = True
    plan['source_preparation_only'] = False
    spec = importlib.util.spec_from_file_location('frozen_F0_endpoint', base / 'evaluate_f0.py')
    endpoint = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = endpoint
    spec.loader.exec_module(endpoint)
    endpoints = plan['endpoint_references']
    common, endpoints, records = endpoint.reference_gate(
        plan, template_path, endpoints['responses']['path'], endpoints['inference_result']['path'])
    prompts = endpoint.rows(common['prompts']['path'])
    audit = endpoint.read(common['input_audit']['path'])
    # Same full-coverage/no-global-timeout/source/cache/canonical-row gate as execution.
    # No baseline assembly or evaluator/gold access occurs here.
    _, coverage, _ = endpoint.verify_inference(plan, common, endpoints, prompts, audit)
    for record in records:
        endpoint.checked(record)
    assert digest(template_path) == binding['template_sha256']
    plan['completed_binding_source'] = endpoint.ref(binding_path)
    plan['binding_coverage'] = coverage
    output = base / 'evaluation-plan.json'
    endpoint.save(output, plan)
    print(json.dumps({'status': 'COMPLETE_F0_RUN_BOUND_NO_QUALITY_EXECUTED',
                      'evaluation_plan': endpoint.ref(output), 'coverage': coverage}))


if __name__ == '__main__':
    main()
