"""Bind only an actually completed epoch2; CPU metadata/hash operation, never launch."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from probe_contract import check_sources, completed_binding, fixed_inputs, ref
from raw_first import atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--training-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    assert template['execution_ready'] is False and template['adapter_binding'] is None
    assert args.output.parent.resolve() == args.template.parent.resolve() and not args.output.exists()
    check_sources(template, args.template.parent)
    rows, encoded, audits = fixed_inputs(template)
    binding = completed_binding(template, args.training_directory)
    plan = dict(template, status='FINAL_EPOCH2_BOUND_PENDING_PARENT_EXECUTION_NOTIFICATION',
                execution_ready=True, adapter_binding=binding, template_reference=ref(args.template),
                bound_utc=datetime.now(timezone.utc).isoformat())
    atomic(args.output, plan)
    atomic(args.output.with_name('binding-result.json'), dict(
        status='COMPLETE_FINAL_EPOCH2_BINDING_NO_INFERENCE', plan=ref(args.output), template=ref(args.template),
        binding=binding, input_rows=len(rows), selected_index=73, selected_tokens=len(encoded[73]['input_ids']),
        selected_ids_sha256=audits[73]['native_ids_sha256'], new_NN_calls=0, quality_calls=0,
        no_model_or_training_state_deserialization=True, execution_not_launched=True))
    print(json.dumps(dict(status='BOUND_NOT_LAUNCHED', plan=ref(args.output))))


if __name__ == '__main__':
    main()
