"""Qwen3 original-G quotes: unchanged training backend plus explicit delta probe."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'formal'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission-expires-unix', required=True, type=float)
    parser.add_argument('--absolute-deadline-unix', required=True, type=float)
    args = parser.parse_args()
    start = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['route_id'] == 'qwen3-grounded-quotes-v1'
    assert args.mode in plan['authorized_phases'] and plan['GPU_authorized'] is True
    assert time.time() < args.admission_expires_unix < args.absolute_deadline_unix
    backend = Path(plan['backend_source_dir'])
    assert 'common' not in sys.modules and 'native' not in sys.modules
    sys.path.insert(0, str(backend))
    common = importlib.import_module('common')
    shared = importlib.import_module('native')
    assert Path(common.__file__).resolve() == backend / 'common.py'
    assert Path(shared.__file__).resolve() == backend / 'native.py'
    for item in plan['source_refs'].values():
        common.checked(item)
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    common.save(output / 'started.json', dict(at_utc=common.stamp(), pid=os.getpid(), mode=args.mode,
        plan=common.ref(args.plan), source=common.ref(__file__),
        admission_expires_unix=args.admission_expires_unix,
        absolute_deadline_unix=args.absolute_deadline_unix))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original phase deadline')))
    torch = None
    stage = 'complete_cpu_source_asset_gate'
    try:
        cpu = json.loads(common.checked(plan['cpu_result']).read_text())
        assert cpu['status'] == 'PASS_QWEN3_GROUNDED_126_NATIVE_CPU_NO_MODEL'
        assert cpu['all126_native_HF_IDs_decode_exact'] and cpu['all87_original_answers_EOS_labels_exact']
        assert cpu['counts']['train']['rows'] == 87 and cpu['counts']['held']['rows'] == 39
        for phase in ['train', 'held']:
            common.checked(cpu['counts'][phase]['encoded'])
        common.gpu_gate(plan, output)
        if args.mode == 'probe':
            assets = common.verify_assets(plan, full=True)
        else:
            prior = json.loads(common.checked(plan['probe_complete']).read_text())
            assert prior['formal_budget_gate'] and prior['fresh_state_discarded']
            assets = common.verify_assets(plan, full=False,
                previous=json.loads(common.checked(plan['probe_assets']).read_text()))
        common.save(output / 'verified-assets.json', assets)
        assert time.time() < args.admission_expires_unix
        assert time.time() < args.absolute_deadline_unix
        stage = 'model_load'
        torch, np, model, named, manifest = common.load_model(plan, output)
        stage = args.mode
        if args.mode == 'probe':
            from delta_probe import probe
            result = probe(plan, output, cpu, torch, model, named, manifest, common, shared)
        else:
            result = shared.train(plan, output, cpu, torch, np, model, named)
        result.update(at_utc=common.stamp(), elapsed_seconds=time.monotonic() - start,
            phase=args.mode, no_held_labels_or_quality=True, source=common.ref(__file__), plan=common.ref(args.plan))
        common.save(output / 'complete.json', result)
        common.sync_dir(output)
    except BaseException as exc:
        failure = dict(status='FAILED_NO_AUTOMATIC_RETRY', stage=stage, error=type(exc).__name__,
            message=str(exc)[:2000], elapsed_seconds=time.monotonic() - start)
        if torch is not None:
            failure['memory'] = common.memory(torch)
        common.save(output / 'failure.json', failure)
        common.sync_dir(output)
        raise


if __name__ == '__main__':
    main()
