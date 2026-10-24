"""Independent bounded remote controller; no retry or timeout reset."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from common import ref, save, source_gate, stamp, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    source_gate(plan)
    root = Path(plan['run_root'])
    root.mkdir(exist_ok=False)
    save(root / 'driver-start.json', {'at_utc': stamp(), 'pid': os.getpid(), 'plan': ref(args.plan),
                                    'phase_budgets': {'reference': 900, 'probe': 600, 'formal': 1800}, 'grace': 30})
    stages = []
    phases=[('probe',600),('formal',1800)] if plan.get('inherited_reference') else [('reference',900),('probe',600),('formal',1800)]
    for phase, seconds in phases:
        if phase == 'formal':
            previous = json.loads((root / 'probe/complete.json').read_text())
            if not previous['formal_budget_gate']:
                save(root / 'closed-before-formal.json', {'reason': 'Measured estimate exceeds authorized1800', 'probe': previous})
                break
        start = time.monotonic()
        at = datetime.now(timezone.utc)
        if phase=='probe' and plan.get('probe_absolute_term_utc'):
            seconds=min(seconds,(datetime.fromisoformat(plan['probe_absolute_term_utc'])-at).total_seconds())
            if seconds<=0:
                save(root/'closed-before-probe.json',{'reason':'Original absolute probe window already expired','at_utc':stamp()})
                break
        deadline = start + seconds
        # timeout remains the child's immediate parent if this MCP controller
        # exits unexpectedly; it retains the original per-stage hard budget.
        command = ['timeout', '--signal=TERM', '--kill-after=30', str(seconds),
                   plan['python'], str(Path(__file__).parent / 'native.py'), phase, '--plan', str(Path(args.plan).resolve())]
        environment = dict(os.environ)
        environment.update(plan['environment'])
        environment['CUDA_VISIBLE_DEVICES'] = plan['gpu_uuid']
        with (root / (phase + '.stdout.log')).open('xb') as stdout, (root / (phase + '.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       env=environment, start_new_session=True)
            entry = {'phase': phase, 'pid': process.pid, 'started_at_utc': at.isoformat(),
                     'term_at_utc': (at + timedelta(seconds=seconds)).isoformat(),
                     'kill_at_utc': (at + timedelta(seconds=seconds + 30)).isoformat(),
                     'budget_seconds': seconds, 'command': command}
            save(root / (phase + '-child-start.json'), entry)
            update(root / 'driver-progress.json', entry)
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
                timeout = process.returncode in (124, 137)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=max(.001, deadline + 30 - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        entry.update({'finished_at_utc': stamp(), 'return_code': process.returncode,
                      'timed_out': timeout, 'elapsed_seconds': time.monotonic() - start,
                      'complete_file_exists': (root / phase / 'complete.json').is_file()})
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        entry['released_gpu'] = {'at_utc': stamp(), 'target_gpu': [line for line in gpu.splitlines() if plan['gpu_uuid'] in line],
                                 'target_apps': [line for line in apps.splitlines() if plan['gpu_uuid'] in line]}
        save(root / (phase + '-child-result.json'), entry)
        stages.append(entry)
        if process.returncode != 0 or timeout or not (root / phase / 'complete.json').exists():
            break
        if entry['released_gpu']['target_apps']:
            save(root / 'unexpected-live-GPU-owner.json', entry['released_gpu'])
            break
    success=len(stages)==len(phases) and all(row['return_code']==0 and not row['timed_out'] and row['complete_file_exists'] for row in stages)
    result = {'status': ('COMPLETE_RECOVERY_PROBE_AND_FORMAL' if plan.get('inherited_reference') else 'COMPLETE_ALL_THREE_PHASES') if success else 'STOPPED_WITH_ORIGINAL_EVIDENCE_NO_RETRY',
              'at_utc': stamp(), 'stages': stages, 'inherited_reference':plan.get('inherited_reference'),'no_quality_calls': True}
    save(root / 'driver-result.json', result)
    files = {}
    for directory, prefix in [(root, 'run'), (Path(plan['CPU_ready']['path']).parent, 'cpu'), (Path(__file__).parent, 'source')]:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.jsonl', '.gz', '.log'} and path.stat().st_size < 1048576:
                files[prefix + '/' + str(path.relative_to(directory))] = path
    files['plan.json'] = Path(args.plan)
    files['source/dpo_ops.py'] = Path(__file__).parent.parent / 'dpo_ops.py'
    inventory = {name: ref(path) for name, path in files.items()}
    save(root / 'small-terminal-inventory.json', inventory)
    with zipfile.ZipFile(root / 'small-terminal.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(root / 'small-terminal-inventory.json', 'inventory.json')
    save(root / 'small-terminal-receipt.json', {'archive': ref(root / 'small-terminal.zip'), 'members': len(files) + 1,
                                             'large_states_remain_at_native_checkpoint_locators': True})


if __name__ == '__main__':
    main()
