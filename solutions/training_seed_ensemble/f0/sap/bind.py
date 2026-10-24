"""Bind all new eligible queries to the original fixed 41099-row public bank."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import B, G, PROJECT, checked, read, ref, save, verify_sources

N = Path(__file__).resolve().parent
ROOT = PROJECT + '/training-seed-ensemble-v1/F0-Sap-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preparation-manifest', type=Path, default=B / 'F0-postprocess-v1/normalization-preparation/manifest.json')
    ap.add_argument('--output', type=Path, default=N / 'held-query-plan.json')
    args = ap.parse_args()
    verify_sources()
    contract = read(N / 'host-contract.json')
    for item in contract['sources'].values(): checked(item)
    target = read(checked(contract['target_assets']))
    for item in contract['new_sources'].values(): checked(item)
    prep = read(args.preparation_manifest)
    assert prep['schema'] == 'grounded-held-normalization-preparation-v1'
    assert prep['documents'] == 16 and prep['patients'] == 47 and prep['no_gold_fit_or_score']
    queries = read(checked(prep['queries']))
    assert len(queries) == prep['necessary_public_query_count']
    if not queries:
        print(json.dumps(dict(no_GPU_query_required=True, preparation=ref(args.preparation_manifest))))
        return
    estimate = 20 + .02 * len(queries)
    assert estimate <= 180, 'Complete new query set exceeds existing cost bound; never subset'
    collection = read(B / 'infer-F0-v1/collection.json')
    assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
    old = read(G / 'normalization/held-query-plan.json')
    p = copy.deepcopy(old)
    p['query_count'] = len(queries)
    p['cost_estimate_seconds'] = estimate
    p['authorization'] = 'Necessary complete F0 queries under unchanged train32/public41099 policy; independent 180+30 runtime and 300-second launch TTL'
    sources = dict(old['payload_sources'])
    for name in ('held_query.py', 'held_driver.py', 'sap_admission.py'): sources[name] = ref(N / name)
    sources['queries.json'] = prep['queries']
    for item in sources.values(): checked(item)
    identity = lambda item: {k: item[k] for k in ('bytes', 'sha256')}
    p['files'] = {path: item for path, item in old['files'].items() if not path.startswith(old['remote_root'] + '/')}
    p['files'].update({ROOT + '/' + name: identity(item) for name, item in sources.items()})
    p['payload_sources'] = sources
    p.update(remote_root=ROOT, output_dir=ROOT + '/output', queries=ROOT + '/queries.json',
             policy=ROOT + '/policy_v1.json', operations_source=ref(N / 'held_remote.py'),
             collection_source=ref(N / 'held_collect.py'), binding_source=ref(__file__))
    p['files'] = {ROOT + '/' + name: identity(item) for name, item in sources.items()}
    p['files'].update(target['files'])
    for key in ('model_dir', 'hpo', 'bank', 'labels', 'bank_manifest', 'python', 'gpu_uuid', 'expected_encoder_runtime'): p[key] = target[key]
    p['environment']['PYTHONPATH'] = ROOT + ':' + target['pythonpath']
    p['environment']['CUDA_VISIBLE_DEVICES'] = target['gpu_uuid']
    p['reuse_packages'] = []
    p['reuse_actual_artifacts'] = {}
    p['host_migration'] = dict(contract=ref(N / 'host-contract.json'), original_runtime=old['expected_encoder_runtime'], actual_target_runtime=target['expected_encoder_runtime'], all_model_bank_HPO_bytes_equal=True, bank_reencoded=False, policy_unchanged=True)
    p['argv'] = [p['python'], '-u', ROOT + '/held_query.py', '--plan', ROOT + '/held-query-plan.json']
    p['input_sources'] = dict(preparation=ref(args.preparation_manifest),
        original_public_bank_collection=old['input_sources']['original_public_bank_collection'],
        complete_inference=ref(B / 'infer-F0-v1/collection.json'), fixed_policy=prep['policy'],
        original_asset_plan=ref(G / 'normalization/held-query-plan.json'), target_asset_contract=ref(N / 'host-contract.json'))
    now = time.time()
    p['launch_admission'] = dict(schema='training-seed-Sap-admission-v1', bound_at_unix=now,
        launch_not_after_unix=now + 300, immutable_before_start_intent=True, no_unknown_restart=True)
    save(args.output, p)
    print(json.dumps(ref(args.output)))


if __name__ == '__main__': main()
