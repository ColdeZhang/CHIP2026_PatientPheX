"""Check all immutable pool/request joins without model, labels, or fit."""
import argparse
import collections
import json
from pathlib import Path
import time

import features as f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-manifest', required=True)
    ap.add_argument('--request-summary', required=True)
    ap.add_argument('--hpo', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    started = time.monotonic()
    assert f.sha(a.hpo) == f.HPO_SHA
    names = f.preferred_names(a.hpo)
    assert len(names) == 19120
    summary = json.loads(Path(a.request_summary).read_text())
    request_ref = summary['artifacts']['atomic_request']
    requests = {}
    for row in f.read_rows(f.checked(request_ref)):
        key = row['text'], row['hpo_id']
        assert key not in requests and names.get(row['hpo_id']) == row['name']
        assert len(row['pair_key']) == 64
        requests[key] = row
    manifest = json.loads(Path(a.prepared_manifest).read_text())
    counts, observed, files, fold_counts = collections.Counter(), collections.Counter(), [], []
    for fold in manifest['folds']:
        fc = {'fold': fold['fold']}
        for split in ['train', 'input']:
            spec = fold[split]
            path = f.checked(spec)
            files.append(spec)
            sc = collections.Counter()
            seen_docs = set()
            for row in f.read_rows(path):
                assert row['pmc_id'] not in seen_docs
                seen_docs.add(row['pmc_id'])
                assert set(row['input']) == {'pmc_id', 'pmid', 'patient', 'full_text'}
                # Explicitly do not inspect row['base'] or any label field.
                keys = set()
                for c in row['candidates']:
                    key = c['offset'], c['length'], c['identifier']
                    assert key not in keys and not set(f.FEATURES) & c.keys()
                    keys.add(key)
                    atoms = f.atomic_ids(c['identifier'], names)
                    sc['rows'] += 1
                    sc['compound_rows'] += len(atoms) > 1
                    if not atoms:
                        sc['unencoded_rows'] += 1
                        assert c['identifier'] == '-1', 'unexpected real-pool sentinel'
                    for hid in atoms:
                        pair = c['text'], hid
                        assert pair in requests, ('missing_raw_pair', pair)
                        observed[pair] += 1
                        sc['atomic_requests'] += 1
            sc['documents'] = len(seen_docs)
            assert sc['documents'] == (48 if split == 'train' else 16)
            assert sc['rows'] == fold['new_' + split + '_rows']
            counts.update(sc)
            fc[split] = dict(sc)
        fold_counts.append(fc)
    assert counts['rows'] == 747178 == manifest['total']['new_rows']
    assert observed.keys() == requests.keys()
    assert all(count == requests[key]['pool_occurrences'] for key, count in observed.items())
    assert len(requests) == summary['unique_raw_surface_atomic_id_pairs']
    result = dict(status='PASS_REAL_POOLS_AND_REQUESTS_INPUT_ONLY',
                  source_sha256=f.sha(__file__), features_sha256=f.sha(f.__file__),
                  prepared_manifest={'path': str(Path(a.prepared_manifest).resolve()),
                                     'sha256': f.sha(a.prepared_manifest)},
                  request_summary={'path': str(Path(a.request_summary).resolve()),
                                   'sha256': f.sha(a.request_summary)},
                  request=request_ref, pool_files=files, folds=fold_counts,
                  counts=dict(counts), unique_raw_pairs=len(requests),
                  no_missing_valid_pairs=True, no_mixed_valid_invalid_compound=True,
                  old_candidates_not_modified=True, original_dev_read=False,
                  no_base_or_gold_fields_inspected=True, no_model_loaded=True,
                  no_forward=True, no_fit=True, no_quality_result=True,
                  seconds=time.monotonic()-started)
    with Path(a.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'counts', 'unique_raw_pairs', 'seconds']}))


if __name__ == '__main__':
    main()
