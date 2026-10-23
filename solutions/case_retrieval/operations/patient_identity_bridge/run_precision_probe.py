#!/usr/bin/env python3
"""Launch/recover only the approved single CPU-reviewed A30 training probe.

Requires its frozen launch plan and already-pushed source. New starts are never
retried. If job.json exists, recover its managed ID instead of invoking launch.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--client-directory', type=Path, required=True)
    parser.add_argument('--io-directory', type=Path, required=True)
    args = parser.parse_args()
    base = args.directory.resolve()
    sys.path[:0] = [str(args.client_directory), str(args.io_directory)]
    import remote_io as remote
    plan = json.loads((base / 'probe-launch-plan.json').read_text())
    assert sha(__file__) == plan['launcher_sha256']
    assert sha(base / 'diagnostic-plan.json') == plan['training_plan_sha256']
    assert sha(base / 'diagnose_precision.py') == plan['trainer_sha256']
    assert sha(base / 'actual-cpu-contract.json') == plan['actual_cpu_contract_sha256']
    assert not (base / 'probe-job.json').exists(), 'Existing job: recover managed ID, never implicit restart'
    assert plan['hard_budget_seconds'] == 300 and plan['kill_grace_seconds'] == 30
    assert plan['argv'][:4] == ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', '300']
    remote.M = plan['mount_id']

    def save(name, obj):
        with (base / name).open('x') as handle:
            handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

    def readonly(name, arguments):
        for attempt in range(3):
            try:
                return remote.call(name, arguments)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    def check_gpu(output):
        code = ("import subprocess,time,json; apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True); "
                "assert not any(s.split(',')[0].strip()==" + repr(plan['gpu_uuid']) + " for s in apps.splitlines()),'GPU occupied; no process stopped'; "
                "devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free,memory.total','--format=csv,noheader'],text=True); "
                "target=next(s.split(',') for s in devices.splitlines() if s.split(',')[0].strip()==" + repr(plan['gpu_uuid']) + "); "
                "assert int(target[2].strip().split()[0])>=20000,'Insufficient free assigned GPU memory'; "
                "print(json.dumps({'gpu_uuid':" + repr(plan['gpu_uuid']) + ",'target_empty':True,'compute_apps':apps,'devices':devices,'server_epoch':time.time()}))")
        result = readonly('run_command', dict(command='python3 -c ' + shlex.quote(code), cwd=plan['cwd'], timeout_seconds=20))
        save(output, result)
        assert result['exit_code'] == 0

    source_check = ("from pathlib import Path;import hashlib,json; expected=" + repr(plan['remote_sources']) +
                    "; actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected};assert actual==expected;"
                    "assert not Path('diagnostic-v1').exists();print(json.dumps({'actual_sources':actual,'output_absent':True}))")
    checked = readonly('run_command', dict(command='python3 -c ' + shlex.quote(source_check), cwd=plan['cwd'], timeout_seconds=20))
    save('probe-source-preflight.json', checked)
    assert checked['exit_code'] == 0
    check_gpu('probe-gpu-preflight.json')
    # Single mutation, never automatically retried even on ambiguous transport.
    job = remote.call('start_process', dict(command=shlex.join(plan['argv']), cwd=plan['cwd'], env=plan['env']))
    save('probe-job.json', dict(job=job, launch_plan_sha256=sha(base / 'probe-launch-plan.json'),
                               local_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(started=job['id'], budget_seconds=300, grace_seconds=30)), flush=True)
    cursors = dict(stdout=0, stderr=0)
    while True:
        running = False
        for stream in cursors:
            result = readonly('read_process_output', dict(process_id=job['id'], stream=stream, cursor=cursors[stream], limit=20000))
            cursors[stream] = result['next_cursor']
            running |= result['running']
            if result['content']:
                with (base / ('probe.' + stream + '.log')).open('a') as handle:
                    handle.write(result['content'])
                if stream == 'stdout':
                    print(result['content'], flush=True)
        (base / 'probe-monitor.json').write_text(json.dumps(dict(job=job['id'], cursors=cursors,
            running=running, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2) + '\n')
        if not running:
            break
        time.sleep(10)
    check_gpu('probe-gpu-release.json')
    for name in ['fp16-diagnosis.json', 'diagnostic-complete.json']:
        result = readonly('read_file', dict(path=plan['cwd'] + '/diagnostic-v1/' + name, offset=0, limit=100000))
        assert hashlib.sha256(result['content'].encode()).hexdigest() == result['sha256']
        (base / ('actual-' + name)).write_text(result['content'])
    result = json.loads((base / 'actual-diagnostic-complete.json').read_text())
    assert result['status'] == 'complete_disposable_fp16_cause_and_bf16_probe' and len(result['bf16_steps']) == 4
    assert result['plan_sha256'] == plan['training_plan_sha256'] and result['source_sha256'] == plan['trainer_sha256']
    assert result['fp16_overflow']['found_inf'] > 0 and result['bf16_steps'][1]['classifier_changed']
    assert result['no_checkpoint_or_optimizer_saved'] and result['elapsed_seconds'] < 300
    save('probe-recovery-complete.json', dict(status='complete_released_no_formal_successor',
         managed_id=job['id'], actual_probe_sha256=sha(base / 'actual-diagnostic-complete.json'),
         actual_manifest_sha256=sha(base / 'actual-fp16-diagnosis.json'), release_sha256=sha(base / 'probe-gpu-release.json')))
    print('PROBE_COMPLETE_RELEASED_NO_FORMAL_START', flush=True)


if __name__ == '__main__':
    main()
