"""Write one future complete F0 entry from existing artifacts, without inference or scoring.

This module does not run at import. Execution requires the actual complete94
collection, fixed normalization and closed quality endpoint; it never parses
model text, calls the normalizer/evaluator, or opens/hashes a gold body.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

S = Path(__file__).resolve().parents[1]
D = Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1')
F = D / 'F0'
Q = F / 'quality-F0-v1'
SEEDS = [20261024, 20261025]
BASE_SCORE = .7281956289113009
BASE_SHA = '0c37ee543876f957bbb04049d0be51a905565d6219559ebab5d66abd43bc3ac3'
METRICS = ['menF1', 'docF1', 'micF1', 'macF1', 'Score']


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path): return json.loads(Path(path).read_text())


def checked(item):
    assert ref(item['path']) == item
    return Path(item['path'])


def guard():
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(args[0])).name.endswith('.gold.jsonl'):
                raise PermissionError('F0 reporting cannot open/hash gold')
    sys.addaudithook(audit)


def save(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path); pending.unlink()
    parent = path.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent


def collection(directory, success_key):
    value = read(directory/'collection.json')
    assert value['all_actual_SHA_verified'] and value[success_key]
    assert value['child_exit_before_freeze']['no_running_process']
    wrapper = value['wrapper']
    assert wrapper['returncode'] == 0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper
    assert read(checked(value['files']['wrapper-complete.json'])) == wrapper
    return value


def mapped(collection_value, remote_ref):
    relative = Path(remote_ref['path']).relative_to(collection_value['original_remote']).as_posix()
    local = collection_value['files'][relative]
    assert all(local[key] == remote_ref[key] for key in ['bytes', 'sha256'])
    return checked(local)


def main():
    assert __debug__; guard()
    target = D/'F0-complete-review-entry.json'
    assert not target.exists() and not target.with_name(target.name+'.pending').exists()
    cpu = collection(D/'native-F0-cpu-v1', 'CPU_reported_success')
    c = collection(D/'infer-F0-A30-v1', 'inference_reported_success')
    assert cpu['wrapper']['all47_native_inputs_verified']
    assert c['wrapper']['complete94_actual_members_verified']
    g = read(checked(c['files']['output/complete.json']))
    assert c['wrapper']['complete_sha256'] == c['files']['output/complete.json']['sha256']
    assert g['status'] == 'COMPLETE_FRESH94_SAMPLED_QUOTE_NO_QUALITY'
    assert (g['rows'],g['documents'],g['patients_per_seed'],g['generated']) == (94,16,47,94)
    assert g['seeds'] == SEEDS and set(g['seed_completes']) == set(map(str,SEEDS))
    assert g['no_regeneration_or_answer_repair'] and len(g['generation_refs']) == len(g['raw_refs']) == 94
    assert g['plan']['sha256'] == ref(S/'f0/infer-plan.json')['sha256']
    assert g['source']['sha256'] == ref(S/'f0/run.py')['sha256']
    # Read persisted numeric generation metadata only; no raw response interpretation.
    generations = [read(mapped(c,item)) for item in g['generation_refs']]
    assert len({(r['seed'],r['row_id']) for r in generations}) == 94
    assert g['EOS_rows'] == sum(r['actual_last_token_eos'] for r in generations)
    assert g['cap_rows'] == sum(r['hit_token_limit'] for r in generations)
    per_seed = {}
    for seed in SEEDS:
        rows = [r for r in generations if r['seed'] == seed]
        sc = read(mapped(c,g['seed_completes'][str(seed)]))
        assert sc['status'] == 'COMPLETE_FRESH47_RAW_GENERATION_NO_QUALITY' and sc['rows'] == len(rows) == 47
        assert sc['seed'] == seed and sc['EOS_rows'] == sum(r['actual_last_token_eos'] for r in rows)
        assert sc['cap_rows'] == sum(r['hit_token_limit'] for r in rows)
        per_seed[str(seed)] = dict(rows=len(rows),EOS=sc['EOS_rows'],cap=sc['cap_rows'],
            output_tokens=sum(r['output_tokens'] for r in rows),native_seconds=sc['seconds'])
    ex = read(F/'extraction-v1/manifest.json')
    prep = read(F/'normalization-preparation-v1/manifest.json')
    norm = read(F/'normalized-v1/manifest.json')
    for stage in [ex,prep,norm]:
        assert stage['complete_all94'] and (stage['rows'],stage['documents'],stage['patients']) == (94,16,47)
        assert stage['seeds'] == SEEDS
    assert ex['collection'] == ref(D/'infer-F0-A30-v1/collection.json')
    assert prep['extraction'] == norm['extraction'] == ref(F/'extraction-v1/manifest.json')
    assert norm['preparation'] == ref(F/'normalization-preparation-v1/manifest.json')
    decisions = read(checked(ex['decisions']))
    decision_by_key = {(r['seed'],r['row_id']):r['decision'] for r in decisions}
    assert len(decisions) == len(decision_by_key) == 94
    cap_keys = {(r['seed'],r['row_id']) for r in generations if r['hit_token_limit']}
    assert all(decision_by_key[k]['status']=='abstain' and not decision_by_key[k]['anchors'] for k in cap_keys)
    names = ['native-F0-cpu-v1/collection.json','infer-F0-A30-v1/collection.json']
    names += ['F0/'+x+'/manifest.json' for x in ['extraction-v1','normalization-preparation-v1','normalized-v1']]
    count = prep['necessary_public_query_count']
    assert type(count) is int and count >= 0 and norm['necessary_public_queries'] == count
    nc = None; query_seconds = None
    if count:
        nc = collection(D/'normalization-F0-runtime-v1','query_reported_success')
        assert norm['query_collection'] == ref(D/'normalization-F0-runtime-v1/collection.json')
        assert norm['query_complete'] == nc['files']['output/complete.json']
        done = read(checked(norm['query_complete']))
        assert done['status'] == 'COMPLETE_FIXED_PUBLIC_HELD_QUERIES' and done['query_count'] == count
        assert done['public_bank_rows'] == 41099 and done['no_bank_reencoding'] and done['no_fit'] and done['no_quality']
        query_seconds = done['seconds']; names.append('normalization-F0-runtime-v1/collection.json')
    else:
        assert all(norm[k] is None for k in ['query_collection','query_complete','query_scores'])
    frozen = read(Q/'freeze.json'); result = read(Q/'result.json')
    qp = read(checked(frozen['plan']))
    assert frozen['formula'] == qp['formula'] == 'B4 UNION (S1 INTERSECT S2)'
    assert frozen['complete_all94'] and (frozen['documents'],frozen['patients'],frozen['entities']) == (16,47,947)
    assert frozen['nonphenotype_bytes_exact'] and frozen['baseline_relation_lists_prefix_exact']
    assert frozen['normalization'] == ref(F/'normalized-v1/manifest.json')
    assert frozen['baseline_relations'] == qp['expected_baseline_relations'] == 343
    assert qp['baseline_score'] == BASE_SCORE and qp['old_v2'] == result['old_v2'] == 0
    assert qp['inputs']['baseline']['sha256'] == BASE_SHA
    prior = read(checked(qp['inputs']['baseline_metrics']))
    assert prior['prediction_sha256'] == BASE_SHA and prior['metrics']['Score'] == BASE_SCORE
    assert result['baseline'] == prior['metrics'] and result['freeze'] == ref(Q/'freeze.json')
    assert read(checked(frozen['format']))['valid'] and checked(frozen['prediction']) == Q/'prediction.jsonl'
    terminal = read(Q/'evaluation-terminal.json')
    assert terminal['returncode'] == 0 and not terminal['timed_out'] and terminal['freeze'] == ref(Q/'freeze.json')
    assert terminal['unique_worker'] and terminal['budget_seconds'] == 30 and terminal['grace_seconds'] == 1
    if frozen['added']:
        assert result['status'] == 'COMPLETE_ONE_FIXED_ENDPOINT' and result['new_v2'] == 1
        metric = read(Q/'metrics.json')
        assert metric['metrics'] == result['metrics'] and metric['prediction_sha256'] == frozen['prediction']['sha256']
        assert result['added'] == frozen['added'] and result['added_TP'] + result['added_FP'] == result['added']
        scores = result['metrics']; delta = result['delta']; tp = result['added_TP']; fp = result['added_FP']
        assert all(delta[k] == scores[k]-prior['metrics'][k] for k in METRICS)
        assert result['keep'] == (scores['Score'] > BASE_SCORE)
        assert result['no_single_seed_quality'] and result['no_other_arm_or_seed_scan']
        names += ['F0/quality-F0-v1/'+x for x in ['metrics.json','evaluation-intent.json']]
        metrics_origin = 'one_existing_new_v2'
    else:
        assert result['status'] == 'NO_ADDED_KEYS_BASE_BYTEEXACT_ZERO_V2' and result['new_v2'] == 0
        assert result['keep'] is False and result['no_gold_read'] and frozen['no_added_prediction_byteexact']
        assert frozen['prediction']['sha256'] == BASE_SHA
        assert not (Q/'metrics.json').exists() and not (Q/'evaluation-intent.json').exists()
        scores = prior['metrics']; delta = {k:0.0 for k in METRICS}; tp = fp = 0
        metrics_origin = 'unchanged_baseline_reused_zero_v2'
    assert delta['menF1'] == delta['docF1'] == 0
    names += ['F0/quality-F0-v1/'+x for x in ['freeze.json','result.json','prediction.jsonl','format.json',
        'all-seed-keys.json','intersection-and-additions.json','evaluation-launch.json','evaluation-terminal.json']]
    sources = {}
    for directory in ['f0','postprocess_F0','normalization_F0','reporting']:
        for path in sorted((S/directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py','.json','.md'} and '__pycache__' not in path.parts:
                if directory == 'reporting' and path.name != 'f0_entry.py': continue
                sources[path.relative_to(S).as_posix()] = ref(path)
    accepted = read(S/'evidence/F1-final-review.json'); restored = read(S/'evidence/F1-actual-restore.json')
    assert accepted['accepted_F1_candidate'] and restored['all_SHA_fsync_reopen'] and restored['embedded_manifest_byteexact']
    checked(accepted['review']); checked(restored['root_unique_actual_restore'])
    entry = dict(schema='grounded-open-quote-two-sample-F0-review-entry-v1',
        status=('POSITIVE_F0_PENDING_INDEPENDENT_REVIEW' if result['keep'] else 'NONPOSITIVE_F0_PENDING_INDEPENDENT_REVIEW'),
        refs={name:ref(D/name) for name in names},source_refs=sources,metrics=scores,metrics_origin=metrics_origin,
        baseline=prior['metrics'],delta=delta,added=frozen['added'],added_TP=tp,added_FP=fp,
        keep_F0_candidate=result['keep'],keep_cross_fold_candidate=result['keep'] and accepted['accepted_F1_candidate'],
        new_v2=result['new_v2'],old_v2=0,formula=frozen['formula'],macro_has_no_extra_veto=True,
        actual_generation=dict(rows=g['rows'],documents=g['documents'],patients_per_seed=g['patients_per_seed'],
            EOS=g['EOS_rows'],cap=g['cap_rows'],whole_cap_row_abstained=True,per_seed=per_seed,
            wrapper_seconds=c['wrapper']['elapsed_seconds'],native_seconds=g['seconds'],
            max_allocated_bytes=max(r['max_gpu_allocated_bytes'] for r in generations),
            max_reserved_bytes=max(r['max_gpu_reserved_bytes'] for r in generations),
            output_tokens=sum(r['output_tokens'] for r in generations),raw_and_parsing_reexecuted=False),
        normalization=dict(necessary_queries=count,wrapper_seconds=nc['wrapper']['elapsed_seconds'] if nc else None,
            native_seconds=query_seconds,query_executed=bool(count),per_seed=norm['per_seed']),
        prior_F1_chain=dict(accepted_pointer=ref(S/'evidence/F1-final-review.json'),review=accepted['review'],
            restore_pointer=ref(S/'evidence/F1-actual-restore.json'),actual_root_restore=restored['root_unique_actual_restore']),
        no_single_seed_quality=True,no_other_arm_or_seed_scan=True,originaldev_gold_read=False,no_official_action=True,
        this_writer=dict(source=ref(__file__),no_model_no_parser_no_normalizer_no_evaluator=True,no_gold_body_read_or_hashed=True),
        limitations=['Held F0 confirmation after F1 method development; not originaldev or official acceptance',
            'Same train32 d52/native policy; A30 Torch2.7.1 differs from original greedy A100 Torch2.6; no cross-host bitwise claim',
            'All cap outputs retained and whole-row abstained without retry; all47 patients retained per seed'])
    save(target,entry); print(json.dumps(ref(target)))


if __name__ == '__main__': main()
