"""Bind real full64 H51 cost/memory to the already authorized shared budget."""
from pathlib import Path
import argparse
import json
import os
from activation import ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readiness', type=Path, required=True)
    args = parser.parse_args()
    root = args.readiness
    terminal = root / 'probe-v1/package/terminal'
    wrapper = json.loads((terminal / 'wrapper-complete.json').read_text())
    step = json.loads((terminal / 'step.json').read_text())
    prior = json.loads((root / 'cost-readiness.json').read_text())
    assert wrapper['returncode'] == 0 and not wrapper['timed_out'] and wrapper['target_has_no_process']
    assert wrapper['all12_backwards_and_one_update_verified']
    assert step['actual_forwards'] == step['actual_backwards'] == 12 and step['actual_optimizer_steps'] == 1
    assert step['final_inventory']['all_grad_finite'] and step['final_inventory']['frozen_base_no_gradient']
    assert step['max_supervision128_and_longest6462_completed']
    window = step['diagnostic_window_seconds']
    # Two real extremes are included, but this is a cost proxy, not a guarantee
    # of equal per-token cost or full-dataset memory feasibility.
    projected_pressure = window * (2 * 918058 / 52201)
    projected_load_and_fixed = 2 * (wrapper['elapsed_seconds'] - window)
    fixed_io_allowance = 1200  # Both complete SHM -> .100 exports together.
    update_save_contingency = 120  # Unmeasured allowance, not observed seconds.
    projected_training = max(projected_pressure + projected_load_and_fixed,
                             prior['token_proxy_twoepoch_seconds'])
    estimate = projected_training + fixed_io_allowance + update_save_contingency
    memory = step['memory']
    assert memory['free_bytes'] >= 10 * 1024**3
    assert step['combined_peak_reserved_bytes'] < 35 * 1024**3
    assert 0 < estimate <= 5400
    value = dict(status='ACTUAL_FULL64_H51_PASS_COST_SUPPORTS_CONDITIONAL_FRESH_FORMAL',
        probe_wrapper=ref(terminal / 'wrapper-complete.json'), probe_step=ref(terminal / 'step.json'),
        probe_complete=ref(terminal / 'complete.json'),
        probe_plan=ref(Path(__file__).resolve().parents[1] / 'probe/plan.json'),
        collection=ref(root / 'probe-v1/package/collection.json'), source=ref(__file__),
        prior_estimate=ref(root / 'cost-readiness.json'),
        real_probe_training_window_seconds=window, probe_tokens=52201,
        complete_twoepoch_tokens=2 * 918058,
        token_scaled_probe_training_seconds=projected_pressure,
        twice_observed_nontraining_overhead_proxy_seconds=projected_load_and_fixed,
        estimated_training_seconds=projected_training,
        two_complete_export_allowance_seconds=fixed_io_allowance,
        unmeasured_update_and_save_contingency_seconds=update_save_contingency,
        estimated_complete_campaign_seconds=estimate,
        formal_cost_fits_5400=True, hard_shared_budget_seconds=5400,
        actual_training_already_performed=False, estimate_not_guarantee=True,
        actual_peak_allocated_bytes=step['combined_peak_allocated_bytes'],
        actual_peak_reserved_bytes=step['combined_peak_reserved_bytes'],
        observed_last_free_bytes=memory['free_bytes'],
        no_claim_of_full265_backward_coverage=True, probe_state_discarded=True,
        fresh_epochs=2, rows_per_epoch=265, updates_per_epoch=34,
        total_updates=68, tail_microbatches_per_epoch=1, warmup_steps=4)
    path = root / 'formal-cost-decision.json'
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(reference=ref(path),estimated_seconds=estimate)))


if __name__ == '__main__':
    main()
