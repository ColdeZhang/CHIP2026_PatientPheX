#!/usr/bin/env python3
"""Fixed two fresh CE arms, shared training budget, then final inference only.

No labels from heldout data or evaluator are opened. Each managed job is saved
before monitoring, starts are never retried, and an existing job blocks a new
launcher invocation. Recover the original stage ID after any interruption.
"""
from pathlib import Path
import argparse
import base64
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
    plan = json.loads((base / 'formal-paired-plan.json').read_text())
    assert sha(__file__) == plan['source_sha256']
    assert plan['arm_order'] == ['baseline', 'bridge'] and plan['shared_training_cap_seconds'] == 1500
    assert plan['per_arm_training_cap_seconds'] == 900 and plan['per_arm_inference_cap_seconds'] == 300
    assert plan['kill_grace_seconds'] == 30 and plan['unique_checkpoint_epoch'] == 4
    assert not (base / 'training-budget-start.json').exists() and not list(base.glob('*-job.json')), 'Existing run: recover original jobs, never restart'
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

    def fetch(remote_name, local_name):
        result = readonly('read_file', dict(path=plan['cwd'] + '/' + remote_name, offset=0, limit=1000000))
        assert hashlib.sha256(result['content'].encode()).hexdigest() == result['sha256']
        with (base / local_name).open('x') as handle:
            handle.write(result['content'])
        return json.loads(result['content']) if local_name.endswith('.json') else result['content']

    def gpu_state(name):
        code = ("import subprocess,time,json; apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True); "
                "assert not any(s.split(',')[0].strip()==" + repr(plan['gpu_uuid']) + " for s in apps.splitlines()),'GPU occupied; no process stopped'; "
                "devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free,memory.total','--format=csv,noheader'],text=True); "
                "target=next(s.split(',') for s in devices.splitlines() if s.split(',')[0].strip()==" + repr(plan['gpu_uuid']) + "); "
                "assert int(target[2].strip().split()[0])>=20000;print(json.dumps({'gpu_uuid':" + repr(plan['gpu_uuid']) +
                ",'target_empty':True,'compute_apps':apps,'devices':devices,'server_epoch':time.time()}))")
        result = readonly('run_command', dict(command='python3 -c ' + shlex.quote(code), cwd=plan['cwd'], timeout_seconds=20))
        save(name, result)
        assert result['exit_code'] == 0
        return json.loads(result['stdout'])

    def managed(stage, argv):
        job = remote.call('start_process', dict(command=shlex.join(argv), cwd=plan['cwd'], env=plan['env']))
        save(stage + '-job.json', dict(job=job, argv=argv, paired_plan_sha256=sha(base / 'formal-paired-plan.json'),
                                      local_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(stage=stage, started=job['id'])), flush=True)
        cursors = dict(stdout=0, stderr=0)
        while True:
            running = False
            for stream in cursors:
                result = readonly('read_process_output', dict(process_id=job['id'], stream=stream, cursor=cursors[stream], limit=20000))
                cursors[stream] = result['next_cursor']
                running |= result['running']
                if result['content']:
                    with (base / (stage + '.' + stream + '.log')).open('a') as handle:
                        handle.write(result['content'])
                    if stream == 'stdout':
                        print(result['content'], flush=True)
            (base / (stage + '-monitor.json')).write_text(json.dumps(dict(job=job['id'], cursors=cursors, running=running,
                at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2) + '\n')
            if not running:
                break
            time.sleep(15)
        gpu_state(stage + '-release.json')

    for role, reference in plan['local_references'].items():
        path = Path(reference['path'])
        assert path.stat().st_size == reference['bytes'] and sha(path) == reference['sha256'], 'Changed ' + role
    code = ("from pathlib import Path;import hashlib,json;expected=" + repr(plan['remote_sources']) +
            ";actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected};assert actual==expected;"
            "outs=" + repr(['formal-' + arm + '-v1' for arm in plan['arm_order']] + ['infer-' + arm + '-v1' for arm in plan['arm_order']]) +
            ";assert not any(Path(p).exists() for p in outs);print(json.dumps({'actual_sources':actual,'all_outputs_absent':True}))")
    checked = readonly('run_command', dict(command='python3 -c ' + shlex.quote(code), cwd=plan['cwd'], timeout_seconds=30))
    save('source-preflight.json', checked)
    assert checked['exit_code'] == 0
    start = gpu_state('training-initial-gpu.json')
    deadline = start['server_epoch'] + 1500
    save('training-budget-start.json', dict(server_epoch=start['server_epoch'], shared_deadline_epoch=deadline,
                                           shared_cap_seconds=1500, per_arm_cap_seconds=900, kill_grace_seconds=30))
    terminal = {}
    for arm in plan['arm_order']:
        if arm != 'baseline':
            gpu_state('train-' + arm + '-preflight.json')
        train_plan_name = 'formal-' + arm + '-plan.json'
        train_plan = json.loads((base / train_plan_name).read_text())
        assert train_plan['execution_mode'] == 'train' and train_plan['arm'] == arm and train_plan['max_updates'] == 1032
        argv = [plan['python'], 'budget_exec.py', '--deadline', str(deadline), '--max-seconds', '900', '--',
                plan['python'], '-u', 'train_encoded_ce.py', '--plan', train_plan_name, '--mode', 'train', '--output', 'formal-' + arm + '-v1']
        managed('train-' + arm, argv)
        relative = 'formal-' + arm + '-v1'
        complete = fetch(relative + '/training-complete.json', arm + '.training-complete.json')
        manifest = fetch(relative + '/run-manifest.json', arm + '.training-manifest.json')
        assert complete['status'] == 'complete_four_epochs' and complete['updates'] == 1032
        assert [r['epoch'] for r in complete['epochs']] == [1, 2, 3, 4]
        assert complete['source_sha256'] == plan['trainer_sha256'] and complete['plan_sha256'] == sha(base / train_plan_name)
        assert complete['elapsed_seconds'] < 900 and manifest['arm'] == arm and manifest['fresh_public_base']
        assert not manifest['old_gold_CE_or_probe_state_reused']
        hcode = ("from pathlib import Path;import json,hashlib;root=Path(" + repr(relative + '/model') +
                 ");names=sorted(p.name for p in root.iterdir());assert names==['config.json','model.safetensors'];"
                 "print(json.dumps({'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir()},"
                 "'training_complete_mtime_epoch':(root.parent/'training-complete.json').stat().st_mtime}))")
        hashed = readonly('run_command', dict(command='python3 -c ' + shlex.quote(hcode), cwd=plan['cwd'], timeout_seconds=30))
        save(arm + '.model-actual-hashes.json', hashed)
        assert hashed['exit_code'] == 0
        observed = json.loads(hashed['stdout'])
        assert observed['training_complete_mtime_epoch'] <= deadline, 'Shared training deadline exceeded'
        terminal[arm] = dict(training_complete=complete, training_manifest=manifest, model_files=observed['files'],
                             actual_complete_mtime_epoch=observed['training_complete_mtime_epoch'])
    # Inference has its own two fixed 300s budgets, after both training jobs.
    for arm in plan['arm_order']:
        relative = 'formal-' + arm + '-v1'
        inference_plan = dict(source_sha256=plan['inference_sha256'], arm=arm, hard_budget_seconds=300,
            runtime_versions=plan['runtime_versions'], gpu_uuid=plan['gpu_uuid'],
            training_complete=dict(path=relative + '/training-complete.json', sha256=sha(base / (arm + '.training-complete.json'))),
            training_manifest=dict(path=relative + '/run-manifest.json', sha256=sha(base / (arm + '.training-manifest.json'))),
            trainer_source_sha256=plan['trainer_sha256'], training_plan_sha256=sha(base / ('formal-' + arm + '-plan.json')),
            model_directory=relative + '/model', model_files=terminal[arm]['model_files'],
            encoded_file=plan['heldout_encoded_file'], encoded_sha256=plan['heldout_encoded_sha256'])
        name = 'inference-' + arm + '-plan.json'
        save(name, inference_plan)
        # Idempotency is intentionally not assumed for this mutation.
        result = remote.call('write_file', dict(path=plan['cwd'] + '/' + name,
            content=base64.b64encode((base / name).read_bytes()).decode(), encoding='base64', create_parents=True))
        gpu_state('infer-' + arm + '-preflight.json')
        argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', '300', plan['python'], '-u',
                'infer_encoded_ce.py', '--plan', name, '--output', 'infer-' + arm + '-v1']
        managed('infer-' + arm, argv)
        raw = fetch('infer-' + arm + '-v1/pair-scores.jsonl', arm + '.pair-scores.jsonl')
        complete = fetch('infer-' + arm + '-v1/inference-complete.json', arm + '.inference-complete.json')
        assert complete['status'] == 'complete_unique_final_epoch_inference' and complete['rows'] == 1166
        assert complete['output_sha256'] == sha(base / (arm + '.pair-scores.jsonl')) and complete['elapsed_seconds'] < 300
        assert complete['source_sha256'] == plan['inference_sha256'] and complete['plan_sha256'] == sha(base / name)
        assert complete['model_files'] == terminal[arm]['model_files'] and complete['no_gold_access']
        terminal[arm]['inference_complete'] = complete
    save('formal-complete.json', dict(status='complete_two_fresh_final_arms_no_evaluation',
         paired_plan_sha256=sha(base / 'formal-paired-plan.json'), arms=terminal,
         both_predictions_must_be_frozen_before_gold=True, official_submission=False))
    print('FORMAL_PAIR_COMPLETE_RELEASED_NO_EVALUATION', flush=True)


if __name__ == '__main__':
    main()
