"""Bind only an actual complete new epoch2 after both durable 11/22-step ACKs."""
import argparse
import copy
import time
from common import B, G, OLD_B, PROJECT, REMOTE, ROUTE, S, checked, read, ref, save, verify_sources


def validate_terminal(formal, trainplan):
    collection = read(formal / 'collection.json')
    assert collection['all_actual_SHA_verified']
    for item in collection['files'].values(): checked(item)
    wrapper = read(formal / 'terminal/wrapper-complete.json')
    assert wrapper['returncode'] == 0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper
    assert wrapper['all_epochs_persistent_and_readback_verified']
    assert set(wrapper['durability']) == {'1', '2'}
    complete_path = formal / 'terminal/output/training_complete.json'
    complete = read(complete_path)
    assert wrapper['training_complete_sha256'] == ref(complete_path)['sha256']
    assert complete['epochs'] == 2 and complete['global_step'] == 22 and complete['rows'] == 87
    assert complete['source_sha256'] == trainplan['trainer_sha256']
    assert complete['probe_state_used'] is False
    assert complete['both_epochs_file_directory_fsync_and_reopened_SHA_verified']
    for epoch in (1, 2):
        cp = formal / ('terminal/output/epoch-%03d/checkpoint.json' % epoch)
        ck = read(cp)
        ack = read(formal / ('terminal/output/durability-epoch-%03d.json' % epoch))
        assert ack['completed_epochs'] == ck['completed_epochs'] == epoch
        assert ack['global_step'] == ck['global_step'] == 11 * epoch
        assert ack['all_members_reopened_and_SHA_verified'] and ack['reopened_SHA_all_exact'] and ack['file_directory_fsync']
        assert ack['checkpoint_metadata_sha256'] == ref(cp)['sha256']
        assert ack == wrapper['durability'][str(epoch)]['receipt']
        assert ck['fingerprint']['source_sha256'] == trainplan['trainer_sha256']
        assert ck['smoke_only'] is False
        assert {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'} <= set(ck['files'])
    return collection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal-directory', type=__import__('pathlib').Path, default=B / 'formal-fresh-v1')
    ap.add_argument('--formal-plan', type=__import__('pathlib').Path, default=ROUTE / 'training/fresh_v1/formal-plan.json')
    ap.add_argument('--output', type=__import__('pathlib').Path, default=S / 'infer-plan.json')
    args = ap.parse_args()
    verify_sources()
    trainplan = read(args.formal_plan)
    assert trainplan['trainer_sha256'] == '43a902c49151c4f232c3402bf8b6ee5e05b55fe7ddc21cce62538b25da7cbc94'
    validate_terminal(args.formal_directory, trainplan)
    old = read(ROUTE / 'inference_helper/infer-plan.json')
    f0 = read(G / 'f0/infer-plan.json')
    result = read(B / 'F1-postprocess-v1/quality-v1/result.json')
    assert result['strict_positive'] and result['new_v2']==1 and result['old_v2']==0
    assert result['baseline']['Score']==0.701325377212172
    old['schema']='grounded-held47-inference-v1'; old['expected_rows']=47
    for name in ('held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json','native-CPU-result.json'):
        old['payload_sources'][name]=f0['payload_sources'][name]
    p = copy.deepcopy(old)
    sources = {name: item for name, item in old['payload_sources'].items()}
    for name in ('run.py', 'raw_store.py', 'infer_driver.py', 'admission.py', 'policy.json'):
        sources[name] = ref(S / name)
    sources['train.py'] = trainplan['payload_sources']['train.py']; checked(sources['train.py'])
    for name, member in [('checkpoint.json', 'output/epoch-002/checkpoint.json'),
                         ('durability-epoch-002.json', 'output/durability-epoch-002.json'),
                         ('training_complete.json', 'output/training_complete.json'),
                         ('training-wrapper-complete.json', 'wrapper-complete.json')]:
        sources[name] = ref(args.formal_directory / 'terminal' / member)
    for item in sources.values(): checked(item)
    names = {key: __import__('pathlib').Path(item['path']).name for key, item in old['assets'].items()}
    names['training_wrapper'] = 'training-wrapper-complete.json'
    names['admission'] = 'admission.py'
    identity = lambda item: {k: item[k] for k in ('bytes', 'sha256')}
    p['assets'] = {key: dict(path=REMOTE + '/' + name, **identity(sources[name])) for key, name in names.items()}
    p['payload_sources'] = sources
    p['files'] = {REMOTE + '/' + name: identity(item) for name, item in sources.items()}
    p.update(budget_seconds=2100, authorized_budget_seconds=2100, grace_seconds=30,
             authorization='Independent seed20261025; complete47 fresh normal-G greedy; 2100+30 shared absolute budget; one fixed downstream endpoint',
             output_dir=REMOTE + '/output', checkpoint_directory=trainplan['output_dir'] + '/epoch-002',
             adapter=trainplan['output_dir'] + '/epoch-002/adapter',
             operations_source=ref(S / 'remote.py'), collection_source=ref(S / 'collect.py'), binding_source=ref(__file__))
    for key in ('python', 'model', 'gpu_uuid', 'environment', 'storage_parent'): p[key] = trainplan[key]
    p['argv'] = [p['python'], '-u', REMOTE + '/run.py', '--plan', REMOTE + '/infer-plan.json']
    p['input_sources'] = dict(old['input_sources'], formal_collection=ref(args.formal_directory / 'collection.json'),
                             formal_plan=ref(args.formal_plan), helper_sources=ref(S / 'source-manifest.json'), F1_result=ref(B/'F1-postprocess-v1/quality-v1/result.json'), original_F0_plan=ref(G/'f0/infer-plan.json'))
    p['reuse_packages'] = [ref(args.formal_directory / 'package.json'), ref(OLD_B / 'native-cpu-v2/package.json')]
    p['reuse_actual_artifacts'] = {}
    for name, member in [('checkpoint.json', 'output/epoch-002/checkpoint.json'),
                         ('durability-epoch-002.json', 'output/durability-epoch-002.json'),
                         ('training_complete.json', 'output/training_complete.json'),
                         ('training-wrapper-complete.json', 'wrapper-complete.json')]:
        root = trainplan['output_dir'].rsplit('/output', 1)[0]
        p['reuse_actual_artifacts'][name] = dict(path=root + '/' + member, original_root=root,
            original_member=member, actual_collection=ref(args.formal_directory / 'collection.json'), **identity(sources[name]))
    now = time.time()
    p['admission'] = dict(schema='training-seed-F0-admission-v1', bound_at_unix=now,
        launch_not_after_unix=now + 300, deadline_TERM_unix=now + 2100, deadline_KILL_unix=now + 2130,
        immutable_before_start_intent=True, no_unknown_restart=True)
    from admission import validate
    validate(p)
    save(args.output, p)
    print(__import__('json').dumps(ref(args.output)))


if __name__ == '__main__': main()
