"""Append uniform raw prehybrid keys, preserving every old object unchanged."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

FEATURES = [f'raw_{m}_{f}' for m in ('pubmed', 'bioformer') for f in ('observed', 'p')]
NATIVE_FEATURES = [prefix + field for prefix in ('public_', 'bioformer_')
                   for field in ('present', 'score', 'neural', 'dictionary', 'abbreviation', 'origin_unknown')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    return dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)


def checked(spec):
    p = Path(spec['path'])
    assert sha(p) == spec['sha256']
    return p


def key(c):
    return c['offset'], c['length'], c['identifier']


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    start = time.monotonic()
    plan = json.loads(Path(a.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    parent = json.loads(checked(plan['old_prepared_manifest']).read_text())
    raw = rows(checked(plan['raw_selected']))
    assert len(raw) == 7397
    supplied = defaultdict(list)
    for record in raw:
        assert record['observations'] and set(record['observations']) <= {'pubmed', 'bioformer'}
        for cell in record['observations'].values():
            assert cell['original_rank'] == 1 and float(cell['native_probability_5dp']) > .95
            assert .95 < cell['probability'] <= 1
        supplied[record['pmc_id']].append(record)
    out = Path(plan['output_dir'])
    out.mkdir(exist_ok=False)
    census = json.loads(checked(plan['census']).read_text())['policies']['raw_top1_original95_valid']
    folds, total_old, total_new = [], 0, 0
    for previous in parent['folds']:
        fold = dict(previous)
        target = out / f"fold{fold['fold']}"
        target.mkdir()
        for role in ('train', 'input'):
            documents = rows(checked(previous[role]))
            new_count = old_count = 0
            for doc in documents:
                original = doc['candidates']
                old_keys = {key(c) for c in original}
                crf_spans = {key(c)[:2] for c in original if c['crf_span']}
                added = []
                for raw_record in supplied[doc['pmc_id']]:
                    if key(raw_record) in old_keys:
                        continue
                    c = {k:raw_record[k] for k in ('offset', 'length', 'identifier', 'text')}
                    matches = [p for p in doc['input']['full_text'] if p['offset'] <= c['offset'] and
                               c['offset'] + c['length'] <= p['offset'] + len(p['text'])]
                    assert len(matches) == 1
                    p = matches[0];left = c['offset'] - p['offset']
                    assert p['text'][left:left+c['length']] == c['text']
                    c.update(probability=.5, similarity=0., exact=False, is_base=False, crf=False,
                             crf_span=key(c)[:2] in crf_spans, semantic_source=False,
                             **{name:False for name in NATIVE_FEATURES})
                    for model in ('pubmed', 'bioformer'):
                        cell = raw_record['observations'].get(model)
                        c[f'raw_{model}_observed'] = cell is not None
                        c[f'raw_{model}_p'] = float(cell['probability']) if cell else 0.
                    added.append(c)
                assert all(not set(FEATURES) & set(c) for c in original)
                added.sort(key=key)
                doc['candidates'] = original + added
                assert doc['candidates'][:len(original)] == original
                assert len({key(c) for c in doc['candidates']}) == len(doc['candidates'])
                old_count += len(original)
                new_count += len(doc['candidates'])
            expected = next(x for x in census['folds'] if x['fold'] == fold['fold'] and x['role'] == role)
            assert old_count == expected['old_rows'] and new_count == expected['merged_rows']
            dest = target / f'{role}.pool.jsonl.gz'
            payload = ''.join(json.dumps(doc, ensure_ascii=False) + '\n' for doc in documents).encode()
            dest.write_bytes(gzip.compress(payload, mtime=0))
            fold[role] = ref(dest)
            fold[f'new_{role}_rows'] = new_count
            total_old += old_count
            total_new += new_count
        folds.append(fold)
    assert total_old == 244794 and total_new == 252014
    result = dict(status='prepared label-free C pools', source_sha256=sha(__file__), plan_sha256=sha(a.plan),
        old_prepared_manifest=plan['old_prepared_manifest'], raw_selected=plan['raw_selected'], census=plan['census'],
        folds=folds, old_rows=total_old, new_rows=total_new, added_rows=total_new-total_old,
        old_fields_types_keys_order_exact=True, new_features=FEATURES,
        feature_semantics='Only appended raw keys have observed/p columns. Old candidate objects and native canonical fields are untouched. Missing raw columns in feature expansion mean not supplied as an appended raw candidate, not public posterior zero.',
        estimated_four_fit_infer_seconds=70., estimate_basis='Actual old50.21s scaled1.03 plus conservative source features/IO allowance; still only an estimate.',
        no_new_NN=True, no_quality=True, seconds=time.monotonic()-start)
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('old_rows','new_rows','added_rows','seconds')}))


if __name__ == '__main__':
    main()
