"""One bounded train or final80 process with persistent actual exit/release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    phase = plan['phase']; assert phase in ('train','infer')
    cap = 3900 if phase == 'train' else 600
    assert plan['cap_seconds'] == cap and plan['grace_seconds'] == 30
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    assert not any(Path(n).exists() for n in ['driver-started.json','wrapper-complete.json','stdout.log','stderr.log'])
    for name, item in plan['files'].items():
        path = Path(name); assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    output = Path(plan['output_dir'])
    assert not output.exists(), 'Never overwrite or automatically resume an existing phase'
    # Allocation occurs only in the child. There is no retry or fallback.
    command = ['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(cap)+'s']+plan['argv']
    started = time.time()
    save('driver-started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), command=command,
         phase=phase, plan_sha256=sha(args.plan), deadline_TERM_unix=started+cap,
         deadline_KILL_unix=started+cap+30, once_only=True))
    with Path('stdout.log').open('xb') as out, Path('stderr.log').open('xb') as err:
        done = subprocess.run(command, stdout=out, stderr=err, check=False)
    report = dict(at_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
         returncode=done.returncode, elapsed_seconds=time.time()-started,
         cap_seconds=cap, grace_seconds=30, output_dir=str(output), no_retry=True)
    endpoint_error = None
    if done.returncode == 0:
        try:
            if phase == 'train':
                complete_path = output/'training_complete.json'
                complete = json.loads(complete_path.read_text())
                assert complete['epochs'] == 2 and complete['global_step'] == 32 and not complete['smoke_only']
                final = output/'epoch-002'; metadata = json.loads((final/'checkpoint.json').read_text())
                assert metadata['completed_epochs'] == 2 and metadata['global_step'] == 32
                assert all((final/name).is_file() for name in metadata['files'])
                report['training_complete_sha256'] = sha(complete_path)
                report['final_checkpoint_metadata_sha256'] = sha(final/'checkpoint.json')
                report['final_checkpoint'] = str(final)
                report['final_state_persistent_export_pending'] = True
            else:
                path = output/'responses.jsonl'; manifest_path = output/'responses.manifest.json'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                manifest = json.loads(manifest_path.read_text())
                assert len(rows) == len({r['row_id'] for r in rows}) == 80
                assert manifest['strict_all80_EOS_and_PID_CID_schema'] and manifest['fallback_rows'] == 0
                assert manifest['output_sha256'] == sha(path)
                save(output/'complete.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
                     rows=80, responses_sha256=sha(path), manifest_sha256=sha(manifest_path),
                     inference_plan_sha256=sha(args.plan), no_quality=True, complete_coverage=True))
                report['inference_complete_sha256'] = sha(output/'complete.json')
        except Exception as error:
            endpoint_error = dict(type=type(error).__name__, message=str(error))
            report['endpoint_error'] = endpoint_error
    try:
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        memory = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total',
                                          '--format=csv,noheader,nounits'], text=True, timeout=10)
        report['gpu_release'] = dict(at_utc=datetime.now(timezone.utc).isoformat(),
            target=plan['gpu_uuid'], target_has_no_process=plan['gpu_uuid'] not in apps,
            target_memory=[line for line in memory.splitlines() if plan['gpu_uuid'] in line])
    except Exception as error:
        report['gpu_release_unknown'] = type(error).__name__
    save('wrapper-complete.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(done.returncode or (1 if endpoint_error else 0))


if __name__ == '__main__':
    main()
