"""Bind the completed trained-adapter probe to one900+30 full80 plan; no ML calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def ref(path, remote=None):
    path = Path(path); data = path.read_bytes()
    return dict(path=str(remote or path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-collected', type=Path, required=True)
    parser.add_argument('--probe-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    terminal = json.loads((args.probe_collected / 'terminal.json').read_text())
    plan = json.loads(args.probe_plan.read_text())
    assert terminal['exit_code'] == 0 and terminal['timeout_triggered'] is False
    assert terminal['plan']['sha256'] == ref(args.probe_plan)['sha256']
    probe_remote = Path(terminal['plan']['path']).parent
    cache = {'terminal': ref(args.probe_collected / 'terminal.json', probe_remote / 'terminal.json'),
             'plan': terminal['plan']}
    for key, relative in [('result', 'actual/result.json'), ('ids', 'actual/row73.ids.json'),
                          ('raw', 'actual/row73.raw.json'), ('row', 'actual/row73.json')]:
        candidate = ref(args.probe_collected / relative, probe_remote / relative)
        assert {k: candidate[k] for k in ('bytes', 'sha256')} == terminal['files'][relative]
        cache[key] = candidate
    result = json.loads((args.probe_collected / 'actual/result.json').read_text())
    assert result['status'] == 'COMPLETE_SINGLE_FINAL_ADAPTER_LONGEST_FORMAT_COST_PROBE_NO_QUALITY'
    assert result['format_pass'] and result['normalized_valid'] and result['native_EOS_observed']
    assert result['adapter_binding'] == plan['adapter_binding']
    plan['sources'] = {p.name: {k: v for k, v in ref(p).items() if k != 'path'} for p in sorted(directory.glob('*.py'))}
    plan.update(status='FROZEN_FULL80_FINAL_ADAPTER_79FRESH_1CACHE',
        created_utc=datetime.now(timezone.utc).isoformat(), execution_ready=True,
        budget_seconds=900, grace_seconds=30, soft_stop_seconds=870,
        project_root=str(probe_remote.parents[2]), final_adapter_cache=cache,
        prior_probe_source_sha256=result['source']['sha256'],
        cached_indices=[73], new_indices=[i for i in range(80) if i != 73],
        RNG_policy='fresh_same_seed_per_row_not_continuous_RNG',
        policy='Exactly original80 pure F1 inputs,79 fresh final-adapter generations and one exact same-adapter row73 cache; reset seed20261024 per row, no old base-model cache.',
        format_gate='Full coverage required for quality handoff; original normalized codec retains invalid/missing CID abstention. Per-row strict/normalized/EOS/cap coverage disclosed; no semantic repair or negative fill.',
        failure_policy='One900+TERM30 absolute deadline including preflight/load. Softstop870 preserves partial/raw evidence. First failure/timeout preserved, no automatic retry or alternate inference settings.',
        scope='Pure-input F1 inference only, no labels, fitting, prediction assembly, quality, F0/dev or official action.',
        future_cache='Only this final-adapter row73 reused. Output rows preserve exact source identity and fresh/cache distinction.',
        authorization='Root authorized once900+TERM30 after actual trained-adapter longest probe. Parent owns source/plan push, fresh GPU1 gate and unique launch.',
        budget_basis=dict(probe_load_seconds=result['load_seconds'], probe_generation_seconds=result['generation_seconds'],
            probe_generated_tokens=result['generated_tokens'], probe_program_seconds=result['program_seconds'],
            limitation='One50-token output is not an upper bound for other rows.900-second cap fixed before full80; no implicit extension.'))
    assert plan['project_root'] == '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
    assert not args.output.exists()
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(ref(args.output)))


if __name__ == '__main__':
    main()
