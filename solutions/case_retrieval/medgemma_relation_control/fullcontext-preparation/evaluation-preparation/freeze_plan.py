#!/usr/bin/env python3
"""Bind complete endpoint files to pretraining choices; never open gold or score."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    template = json.loads((root / 'evaluation-template.json').read_text())
    source = root / 'evaluate_medgemma.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == template['control_sha256']
    spec = importlib.util.spec_from_file_location('frozen_med_endpoint', source)
    endpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(endpoint)
    w = endpoint.utilities()
    assert w.sha(__file__) == template['freeze_plan_sha256']
    assert template['policy'] == endpoint.POLICY and template['endpoint_paths'] == endpoint.DYNAMIC_PATHS
    assert args.formal_directory.resolve() == Path(template['formal_directory']).resolve()
    assert w.sha(root / 'pretraining-amendment.json') == template['pretraining_amendment']['sha256']
    for record in template['references'].values():
        w.checked(record)
    template['endpoint_references'] = {name: w.ref(args.formal_directory / relative)
                                       for name, relative in endpoint.DYNAMIC_PATHS.items()}
    endpoint.reference_gate(template, w)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    w.save(args.output_directory / 'run-plan.json', template)
    print(json.dumps({'plan': w.ref(args.output_directory / 'run-plan.json'), 'gold_read': False, 'quality_run': False}))


if __name__ == '__main__':
    main()
