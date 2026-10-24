"""Read-only supply audit: existing selected GLiNER additions vs strong HGB.

No gold paths, fitting, neural forward, prediction generation, or scoring.
Training-window labels are never inspected; only pmc_id and file hashes bind
the already reviewed models to their target-excluding source sets.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import time
from datetime import datetime, timezone


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def ref(p, expected=None):
    p = pathlib.Path(p)
    value = dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)
    if expected is not None:
        assert value['sha256'] == expected, (p, expected, value['sha256'])
    return value


def read(p):
    return json.loads(pathlib.Path(p).read_text())


def lines(p):
    return [json.loads(s) for s in pathlib.Path(p).read_text().splitlines() if s]


def keyed(p):
    rows = lines(p)
    result = {str(r['pmc_id']): r for r in rows}
    assert len(result) == len(rows)
    return result


def key(e):
    return e['offset'], e['length'], e['identifier']


def overlap(a, b):
    return a['offset'] < b['offset'] + b['length'] and b['offset'] < a['offset'] + a['length']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--data', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    start = time.monotonic()
    r, s = a.root, a.root / 'sequence_crf'
    strong = r / 'ontology_rules/public-biobert-selector-control/formal-control'
    freeze_path = strong / 'complete-freeze.json'
    freeze_ref = ref(freeze_path, 'cf8c1a2820ca94099d6129e9c121eb11c1fd049e4ec35d89de725ef169408f82')
    frozen = read(freeze_path)
    prepared = s / 'gliner-medium-v2-pilot/a800/prepared'
    prep = read(prepared / 'manifest.json')
    folds = {f: set(keyed(a.data / f'fold-{f}/validation16.input.jsonl')) for f in range(4)}
    assert sum(map(len, folds.values())) == len(set.union(*folds.values())) == 64
    reports = []
    policies = []
    for f, link_name, train_name, window_name in [
        (0, 'gliner-link-outer0', 'gliner-outer0-a30', 'train48.jsonl'),
        (1, 'gliner-link-inner1', 'gliner-long10ep-a800', 'nested_train32.jsonl'),
    ]:
        link = s / link_name
        assembly_path = link / 'assembled/manifest.json'
        assembly = read(assembly_path)
        lm_path = link / 'manifest.json'
        lm = read(lm_path)
        ref(lm_path, assembly['inputs']['candidate_manifest'])
        tm_path = s / train_name / 'formal-fixed10ep/run_manifest.json'
        tm = read(tm_path)
        ref(tm_path, lm['inputs']['span_training_manifest'])
        ref(prepared / 'manifest.json', lm['inputs']['span_data_manifest'])
        window_ref = ref(prepared / window_name, tm['fingerprint']['train_sha256'])
        train_ids = {str(row['pmc_id']) for row in lines(prepared / window_name)}
        expected_train = folds[2] | folds[3] | (folds[1] if f == 0 else set())
        assert train_ids == expected_train == set(lm['training_document_ids'])
        assert train_ids.isdisjoint(folds[f])
        assert set(lm['input_document_ids']) == folds[f]
        assert tm['documents'] == len(train_ids)
        assert tm['completed_epochs'] == 10
        assert tm['fingerprint']['excluded_input_sha256'] == lm['inputs']['input']
        assert prep['files']['train' if f == 0 else 'nested_train'] == lm['inputs']['train']
        input_ref = ref(a.data / f'fold-{f}/validation16.input.jsonl', lm['inputs']['input'])
        assert assembly['inputs']['train'] == lm['inputs']['train']
        assert assembly['policy']['probability'] == .65
        assert assembly['policy']['similarity'] == .92
        policies.append((assembly['policy'], assembly['source_sha256']))
        base_path = link / 'assembled/baseline.jsonl'
        aug_path = link / 'assembled/augmented.jsonl'
        baseline_ref = ref(base_path, assembly['outputs']['baseline.jsonl'])
        augmented_ref = ref(aug_path, assembly['outputs']['augmented.jsonl'])
        strong_ref = frozen['folds'][f]['new_prediction']
        ref(strong_ref['path'], strong_ref['sha256'])
        old, augmented, current = keyed(base_path), keyed(aug_path), keyed(strong_ref['path'])
        assert set(old) == set(augmented) == set(current) == folds[f]
        counts = collections.Counter()
        detail = []
        for pmc in sorted(old):
            original = old[pmc]['entities']
            original_keys = {key(e) for e in original}
            all_aug = augmented[pmc]['entities']
            assert len({key(e) for e in all_aug}) == len(all_aug)
            assert all(e in all_aug for e in original)
            additions = [e for e in all_aug if key(e) not in original_keys]
            selected = current[pmc]['entities']
            selected_keys = {key(e) for e in selected}
            assert all(not overlap(e, old_e) for e in additions for old_e in original)
            assert all(not overlap(e, other) for i, e in enumerate(additions) for other in additions[:i])
            for e in additions:
                collisions = [v for v in selected if overlap(e, v)]
                if key(e) in selected_keys:
                    status = 'already_selected_exact_key'
                elif collisions:
                    status = 'same_boundary_other_id' if any(v['offset'] == e['offset'] and v['length'] == e['length'] for v in collisions) else 'other_overlap_blocked'
                else:
                    status = 'nonoverlap_new_key'
                counts[status] += 1
                detail.append(dict(pmc_id=pmc, entity=e, status=status,
                                   overlapping_keys=[list(key(v)) for v in collisions]))
        assert sum(counts.values()) == assembly['counts']['accepted']
        reports.append(dict(fold=f, counts=dict(counts), existing_gliner_additions=len(detail),
            strong_entities=sum(len(x['entities']) for x in current.values()),
            model_training_documents=sorted(train_ids), target_documents=sorted(folds[f]),
            training_target_intersection=0, actual_epochs=10, actual_steps=tm['total_steps'],
            actual_model_sha256=tm['epoch_results'][-1]['model_sha256'],
            resources=dict(link_manifest=ref(lm_path), assembly_manifest=ref(assembly_path),
                training_manifest=ref(tm_path), prepared_manifest=ref(prepared / 'manifest.json'),
                training_windows=window_ref, target_input=input_ref, old_baseline=baseline_ref,
                old_augmented=augmented_ref, strong_baseline=strong_ref),
            hgb_training_documents=48, gliner_training_documents=len(train_ids),
            source_policy=assembly['policy'], details=detail))
    assert policies[0] == policies[1]
    output = dict(status='input_only_existing_supply_checked_no_quality',
        at_utc=datetime.now(timezone.utc).isoformat(), source=ref(__file__),
        strong_freeze=freeze_ref, folds=reports,
        supply_definition='Only previously accepted GLiNER additions (old augmented minus old CRF/Sap base); never re-add all old CRF entities.',
        history_scope='Sequence notes plus ontology owner confirmation: no prior three-public HGB+proper-nested plus old selected GLiNER-only nonoverlap control found. Earlier CRF/Sap unions are different.',
        limitations=['F0 GLiNER trained48; F1 trained32, both exclude their target16; not a same-training-size comparison.',
          'Both resources and decoder were previously developed; this input-only audit is not a new blind evaluation.',
          'Only F0/F1 existing supply is audited; no claim of fourfold ready models.',
          'No new prediction, gold metrics, fit, neural forward, original dev, or model transfer.'],
        elapsed_seconds=time.monotonic()-start)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(report=ref(a.output), folds=[dict(fold=x['fold'], counts=x['counts']) for x in reports]), ensure_ascii=False))


if __name__ == '__main__':
    main()
