"""Synthetic metadata-only checks; never bind actual output or call MCP/models."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path

import bind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    bind.gold_guard()
    prep = dict(schema='grounded-sampled-normalization-preparation-v1',
        complete_all78=True, rows=78, documents=16, patients=39, seeds=bind.SEEDS,
        no_gold_fit_or_score=True, no_unknown_to_NIL=True, no_threshold_changes=True,
        shared_queries_only_exact_identical_strings=True,
        train32=dict(sha256=bind.TRAIN), input=dict(sha256=bind.INPUT),
        hpo=dict(sha256=bind.HPO), policy=dict(sha256=bind.PINS['policy_v1.json']),
        extraction=dict(SYNTHETIC_NO_PATH=True), necessary_public_query_count=2)
    queries = [dict(quote='SYNTHETIC alpha'), dict(quote='SYNTHETIC beta')]
    outcomes = []

    def case(name, patch=None, changed_queries=None, reject=False):
        p = copy.deepcopy(prep)
        if patch: p.update(patch)
        q = copy.deepcopy(queries if changed_queries is None else changed_queries)
        try:
            count = bind.validate_preparation_metadata(p, q)
        except (AssertionError, KeyError, TypeError):
            assert reject, name
            outcomes.append(dict(case=name, expected_rejection=True, passed=True))
        else:
            assert not reject, name
            outcomes.append(dict(case=name, accepted_count=count, passed=True))

    case('synthetic_complete78_train32')
    case('zero_necessary_queries', dict(necessary_public_query_count=0), [])
    case('partial_sampling', dict(complete_all78=False), reject=True)
    case('wrong_row_count', dict(rows=39), reject=True)
    case('single_seed', dict(seeds=[20261024]), reject=True)
    case('wrong_patients', dict(patients=36), reject=True)
    case('wrong_training_ancestry', dict(train32=dict(sha256='SYNTHETIC_WRONG')), reject=True)
    case('old_extraction_field', dict(extraction_manifest={}), reject=True)
    case('duplicate_query', changed_queries=[queries[0], queries[0]], reject=True)
    case('unsorted_query', changed_queries=list(reversed(queries)), reject=True)
    case('extra_query_key', changed_queries=[dict(quote='SYNTHETIC alpha', extra=True), queries[1]], reject=True)
    case('boolean_query_count', dict(necessary_public_query_count=True), reject=True)
    case('count_mismatch', dict(necessary_public_query_count=1), reject=True)
    case('policy_change', dict(policy=dict(sha256='SYNTHETIC_WRONG')), reject=True)
    large = [dict(quote=f'SYNTHETIC {i:05d}') for i in range(8001)]
    case('budget_overflow_requires_explicit_new_budget', dict(necessary_public_query_count=len(large)), large, reject=True)

    old = bind.N.parent / 'full64_inference'
    namespace_changes = []
    for new_name, old_name in [('remote.py', 'held_remote.py'), ('collect.py', 'held_collect.py')]:
        source = old / old_name
        target = bind.S / new_name
        before = source.read_bytes()
        previous = b'grounded-generation-full64-v1/normalization-dev36-v1'
        current = b'grounded-sampled-consensus-v1/normalization-F1-v1'
        assert before.count(previous) == 1
        assert target.read_bytes() == before.replace(previous, current)
        namespace_changes.append(dict(original=bind.ref(source), current=bind.ref(target),
                                      sole_change='One private namespace literal'))

    runtime = {name:bind.ref(bind.N/name) for name in bind.PINS}
    assert {name:item['sha256'] for name,item in runtime.items()} == bind.PINS
    host = bind.read(bind.S/'target-assets.json')
    assert host['mount'] == bind.MOUNT and host['gpu_uuid'] == bind.GPU
    assert all(
        host['expected_encoder_runtime'][k] == host['actual_same_encoder_on_target'][k]
        for k in host['expected_encoder_runtime'])
    # Actual small metadata only; no peer predictions, model/bank body or train labels.
    for item in host['proofs'].values(): bind.checked(item)
    bind.checked(host['inherited_asset_contract'])
    result = dict(schema='grounded-sampled-normalization-metadata-check-v1',
        status='PASS_SYNTHETIC_METADATA_AND_LOCAL_SOURCE_IDENTITIES_ONLY',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=bind.ref(__file__), binder=bind.ref(bind.S/'bind.py'), target_assets=bind.ref(bind.S/'target-assets.json'),
        cases=outcomes, copied_transport_exact_except_namespace=namespace_changes,
        unchanged_query_runtime=runtime, no_actual_preparation_or_plan_created=True,
        no_MCP_or_GPU_or_fit_or_bank_encoding_or_gold_or_quality=True,
        no_current_GPU_availability_or_complete78_claim=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    parent = args.output.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent
    print(json.dumps(dict(result=bind.ref(args.output), cases=len(outcomes), no_actual_query_plan=True)))


if __name__ == '__main__': main()
