#!/usr/bin/env python3
"""Bind only the one actual complete58 terminal into a NEW enabled plan, no quality."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sft32_full_report_F0_evaluation', Path(__file__).with_name('evaluate_full58.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
TERMINAL = {'responses': 'responses.jsonl', 'canonical': 'canonical.jsonl', 'summary': 'summary.json',
            'complete': 'complete.json', 'wrapper': 'wrapper-complete.json', 'gpu_release': 'gpu-release.json',
            'actual_input': 'actual-input-and-assets.json'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True)
    parser.add_argument('--terminal', required=True)
    parser.add_argument('--output-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    template_ref = evaluation.ref(args.template)
    template = evaluation.read(args.template)
    evaluation.reference_gate(template, require_ready=False)
    terminal = Path(args.terminal).resolve()
    # Exclusive fixed terminal marker applies across every possible output directory.
    evaluation.require(not (terminal / 'evaluation-started-full-report-v1.json').exists(),
                       'Quality attempt for this actual terminal already started')
    evaluation.require(not Path(args.output_dir).exists(), 'Quality output already exists')
    plan = copy.deepcopy(template)
    plan['actual_references'] = {key: evaluation.ref(terminal / name) for key, name in TERMINAL.items()}
    plan['terminal_dir'] = str(terminal)
    plan['output_dir'] = str(Path(args.output_dir).resolve())
    plan['execution_ready'] = True
    plan['binding'] = {'template': template_ref, 'binder': evaluation.ref(__file__),
        'at_utc': datetime.now(timezone.utc).isoformat(), 'gold_read': False, 'quality_calls': 0}
    common, inference, actual, _ = evaluation.reference_gate(plan)
    _, _, coverage, shard_refs = evaluation.verify_inference(common, inference, actual, terminal)
    plan['binding']['verified_coverage'] = coverage
    plan['binding']['raw_shards_verified'] = len(shard_refs)
    evaluation.checked(template_ref)
    evaluation.save(Path(args.output_plan), plan)
    print(evaluation.ref(args.output_plan))


if __name__ == '__main__':
    main()
