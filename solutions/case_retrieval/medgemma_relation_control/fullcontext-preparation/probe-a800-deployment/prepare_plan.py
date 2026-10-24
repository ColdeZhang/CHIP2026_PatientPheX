"""Freeze deployment and owner-only single GPU launch plan, without launching."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import shlex
import tarfile

D = Path(__file__).resolve().parent
F = D.parent
E = F.parent
R = E.parents[1]
P = R / 'case_retrieval/qwen35-preparation/sft-meta1-fusion'
REMOTE = 'research/medgemma-relation-control/fullcontext-probe-a800-v1'
OLD = 'research/medgemma-relation-control/a800-preparation'
ROOT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (D/name).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    old = json.loads((E/'probe-a800/run-plan.json').read_text())
    local = {
        'probe-source/probe_fullcontext_longest.py': F/'probe-source/probe_fullcontext_longest.py',
        'probe-source/fullcontext_contract.py': F/'probe-source/fullcontext_contract.py',
        'construct.py': F/'construct.py', 'legacy_full_context.py': F/'legacy_full_context.py',
        'train.fullcontext.jsonl': F/'train.fullcontext.jsonl',
        'train32.input.jsonl': P/'public-training-candidate-preparation/train32.input.jsonl',
        'preparation.json': F/'preparation.json',
        'reference/summary.json': F/'actual-cpu/summary.json',
        'reference/row-token-audit.json': F/'actual-cpu/row-token-audit.json'}
    members = {n:{'local':str(p), 'bytes':p.stat().st_size, 'sha256':sha(p), 'remote':REMOTE+'/'+n}
               for n,p in local.items()}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, path in sorted(local.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    payload = gzip.compress(stream.getvalue(), compresslevel=6, mtime=0)
    bundle = D/'deployment-source-inputs.tar.gz'
    with bundle.open('xb') as handle:
        handle.write(payload)
    assert len(payload) <= 1550000, 'Keep this deployment a small compressed source/input transfer'
    reuse_names = ['train_medgemma32.py','medgemma_interface.py','qwen35_smoke.py','qwen35_relation_cost.py',
                   'train.jsonl','excluded.input.jsonl','preparation-manifest.json','expected-files.json']
    reused = {OLD+'/'+n:old['remote_source_files'][OLD+'/'+n] for n in reuse_names}
    env = dict(old['env'])
    env['PYTHONPATH'] = ROOT+'/'+OLD+':'+env['PYTHONPATH']
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    output = old['shm_root']+'/fullcontext-probe-longest-once'
    argv = [old['probe_argv'][0], '-u', REMOTE+'/probe-source/probe_fullcontext_longest.py',
            '--train', REMOTE+'/train.fullcontext.jsonl', '--original-train', OLD+'/train.jsonl',
            '--train-input', REMOTE+'/train32.input.jsonl', '--excluded-input', OLD+'/excluded.input.jsonl',
            '--data-manifest', OLD+'/preparation-manifest.json', '--fullcontext-manifest', REMOTE+'/preparation.json',
            '--model',old['model'],'--model-source-manifest',OLD+'/expected-files.json',
            '--cpu-reference',REMOTE+'/reference/summary.json','--cpu-row-audit',REMOTE+'/reference/row-token-audit.json',
            '--output-dir',output,'--gpu-uuid',old['gpu_uuid']]
    wrapped = ['/usr/bin/timeout','--signal=TERM','--kill-after=30','180']+argv
    save('source-bundle.json', {'status':'frozen_exact_small_source_inputs','namespace':REMOTE,'members':members,
                              'file_count':len(members),'total_member_bytes':sum(r['bytes'] for r in members.values()),
                              'bundle':{'path':str(bundle),'bytes':len(payload),'sha256':sha(bundle)},
                              'existing_A800_inputs_and_sources':reused,'no_weights_in_bundle':True})
    save('run-plan.json', {'status':'authorized_single_longest_diagnostic_owner_launch_pending',
                          'mount_id':old['mount_id'],'gpu_uuid':old['gpu_uuid'],'model':old['model'],
                          'namespace':REMOTE,'env':env,'cwd':'.','probe_argv':argv,'timeout_argv':wrapped,
                          'timeout_command':shlex.join(wrapped),'probe_output':output,
                          'remote_source_files':{r['remote']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in members.values()},
                          'reused_source_files':reused,'source_bundle_sha256':sha(D/'source-bundle.json'),
                          'source_only_plan_sha256':sha(F/'probe-source/source-plan.json'),
                          'original_training_cap_unchanged':8192,'explicit_new_diagnostic_cap':14000,
                          'expected_selection':{'row_id':'12628480:relations:8','full_tokens':12824,'prefix_tokens':12754,'supervised_tokens':70},
                          'GPU_budget_seconds':180,'TERM_grace_seconds':30,'absolute_deadline_UTC':None,
                          'fresh_empty_GPU_required':True,'owner_must_fresh_check_GPU_before_launch':True,
                          'CPU_delta_not_to_run':True,'fresh_base_no_adapter_resume':True,
                          'no_diagnostic_checkpoint_saved':True,'no_formal_training_no_quality':True,
                          'stop_policy':'First OOM/nonfinite/deadline stops; no retry or short-row fallback. UNKNOWN preserves known intent/job, never launches another.',
                          'probe_source_sha256':sha(F/'probe-source/probe_fullcontext_longest.py'),
                          'input_contract_sha256':sha(F/'probe-source/fullcontext_contract.py'),
                          'preparation_source_sha256':sha(__file__),
                          'asset_inventory_and_existing_copy_ack_reused':True,
                          'native_probe_retains_original_asset_load_guard':True,
                          'current_deployment_does_not_hash_or_copy_model_weights':True})
    print(json.dumps({'bundle_bytes':len(payload),'bundle_sha256':sha(bundle),
                      'run_plan_sha256':sha(D/'run-plan.json'),'files':len(members)}))


if __name__ == '__main__':
    main()
