"""Append the existing public PhenoBERT pair means to frozen original Bioformer pools."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phenobert_pair_control'))
import features as pair_features
FEATURES = pair_features.FEATURES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def checked(item):
    path = Path(item['path'])
    assert sha(path) == item['sha256']
    return path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def norm(text):
    return ' '.join(text.lower().split()).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True)
    args = ap.parse_args(); start = time.monotonic(); plan = read(args.plan)
    assert sha(__file__) == plan['source_sha256']
    assert Path(pair_features.__file__).resolve() == checked(plan['feature_source']).resolve()
    old = read(checked(plan['old_prepared_manifest']))
    pair_cache = pair_features.load(plan['pair_cache'])
    out = Path(plan['output_dir']); out.mkdir(exist_ok=False)
    folds, total, unique = [], Counter(), {}
    assert [f['fold'] for f in old['folds']] == [0, 1, 2, 3]
    heldout = set()
    for fold in old['folds']:
        assert len(fold['train_ids']) == len(set(fold['train_ids'])) == 48
        assert len(fold['input_ids']) == len(set(fold['input_ids'])) == 16
        assert not set(fold['train_ids']) & set(fold['input_ids'])
        assert not heldout & set(fold['input_ids']); heldout.update(fold['input_ids'])
        assert len(fold['nested_input_ids']) == 3 and all(len(x) == 16 for x in fold['nested_input_ids'])
        assert [d for ids in fold['nested_input_ids'] for d in ids] == fold['train_ids']
        target = out / f'fold{fold["fold"]}'; target.mkdir()
        result = dict(fold); counts = {}
        for role in ['train', 'input']:
            documents = rows(checked(fold[role]))
            assert [d['pmc_id'] for d in documents] == fold[role+'_ids']
            stat = Counter(); path = target / (role+'.pool.jsonl.gz')
            with path.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
                    with io.TextIOWrapper(zipped, encoding='utf8') as stream:
                        for index, document in enumerate(documents):
                            assert document['origin_index'] == (index//16 if role == 'train' else None)
                            assert set(document['input']) == {'pmc_id', 'pmid', 'full_text', 'patient'}
                            candidates, keys = [], set()
                            for candidate in document['candidates']:
                                assert not set(FEATURES) & candidate.keys()
                                assert not any('_top3_' in key for key in candidate)
                                key = candidate['offset'], candidate['length'], candidate['identifier']
                                assert key not in keys; keys.add(key)
                                pair = candidate['text'], candidate['identifier']
                                added = pair_cache.features(candidate)
                                observed = sum(added.values()) > 0
                                unique[pair] = observed
                                new = dict(candidate, **added)
                                assert {k:v for k,v in new.items() if k not in FEATURES} == candidate
                                assert all(type(new[k]) is type(v) for k,v in candidate.items())
                                candidates.append(new); stat['rows'] += 1; stat['observed'] += observed
                                stat['missing'] += not observed
                                if not observed:
                                    stat['missing_nil' if '-1' in candidate['identifier'].split(';') else 'missing_not_current_branch'] += 1
                            stream.write(json.dumps(dict(document, candidates=candidates), ensure_ascii=False)+'\n')
            assert stat['rows'] == fold['old_'+role+'_rows']
            result[role] = ref(path); counts[role] = dict(stat); total.update(stat)
        result['pair_counts'] = counts; folds.append(result)
    assert len(heldout) == 64
    result = dict(status='Original Bioformer pools exact; only three public PhenoBERT pair fields appended',
                  source_sha256=sha(__file__), plan_sha256=sha(args.plan), old_prepared_manifest=plan['old_prepared_manifest'],
                  pair_cache=plan['pair_cache'], pair_feature_visits=dict(pair_cache.counts), folds=folds, total=dict(total),
                  unique_pairs=len(unique), unique_observed=sum(unique.values()), unique_missing=sum(not v for v in unique.values()),
                  new_features=FEATURES, top3_new_keys=False, top3_six_features=False,
                  old_keys_fields_types_order_exact=True, strict_source_groups=16,
                  estimated_four_fit_infer_seconds=plan['estimated_four_fit_infer_seconds'],
                  estimate_basis=plan['estimate_basis'], no_gold_read=True, no_GPU=True,
                  seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['total','unique_pairs','unique_observed','unique_missing','seconds']}))


if __name__ == '__main__':
    main()
