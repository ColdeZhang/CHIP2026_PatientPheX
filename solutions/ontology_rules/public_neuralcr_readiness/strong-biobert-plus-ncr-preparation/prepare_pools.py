"""Input-only NCR increment over all eight immutable strong BioBERT pools.

No imports of model/training code, gold, posterior files, or evaluators.
Only main() reads the plan and actual artifacts; merge() supports synthetic use.
"""
import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time


FEATURES = ['ncr_present', 'ncr_score']
PUBLIC_FEATURES = [p + x for p in ('public_', 'bioformer_')
                   for x in ('present', 'score', 'neural', 'dictionary',
                             'abbreviation', 'origin_unknown')]
BIOBERT_FEATURES = ['biobert_' + x for x in
                    ('present', 'score', 'neural', 'dictionary', 'abbreviation', 'origin_unknown')]
OLD_FEATURES = PUBLIC_FEATURES + BIOBERT_FEATURES
BASE_MANIFEST_SHA256 = '0ba8dffdf4fd69d32765cd2de4878c64f6d9e95e20ab7d910d15239bd66206ce'
BASE_FIT_SOURCE_SHA256 = 'ae870b823abaae2cc50809e633f9b4f5b0ecf13a5515a846e022b284d4309253'
BASE_POOL_ROWS = 246804
PREB_POOL_ROWS = 244794
RAW_SPAN_UPPER_BOUND = 5341
CACHE_STATUS = 'complete_NCR_current_canonical_from_verified_raw_cache_no_quality'
CANONICAL_FIELDS = {'pmc_id', 'offset', 'length', 'identifier', 'text',
                    'ncr_present', 'ncr_score'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item):
    require(isinstance(item, dict) and all(k in item for k in ('path', 'sha256')),
            'Invalid artifact reference')
    path = Path(item['path'])
    require(path.is_absolute() and sha(path) == item['sha256'],
            'Artifact SHA/path mismatch: ' + str(path))
    if 'bytes' in item:
        require(path.stat().st_size == item['bytes'], 'Artifact size mismatch: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def sources(plan):
    refs = plan['sources']
    require(isinstance(refs, dict) and bool(refs), 'Plan must bind source SHA values')
    own = Path(__file__).resolve()
    require(any(Path(path).resolve() == own for path in refs), 'Plan omits this preparation source')
    for path, digest in refs.items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Source SHA mismatch: ' + str(path))


def key(candidate):
    return candidate['offset'], candidate['length'], candidate['identifier']


def fullkey(candidate):
    return (candidate['pmc_id'],) + key(candidate)


def defaults():
    return {'ncr_present': False, 'ncr_score': 0.0}


def same_json(current, original):
    """Check JSON values, exact Python types, and mapping/list order recursively."""
    if type(current) is not type(original):
        return False
    if isinstance(original, dict):
        return list(current) == list(original) and all(same_json(current[k], original[k]) for k in original)
    if isinstance(original, list):
        return len(current) == len(original) and all(same_json(a, b) for a, b in zip(current, original))
    return current == original


def reject_old_posterior(candidate):
    require('selector_probability' not in candidate, 'Old selector posterior is not an input feature')


def base_missing_defaults():
    # Same values/types as the strong BioBERT merger for missing source evidence.
    return {name: False for name in PUBLIC_FEATURES} | {
        name: (0.0 if name == 'biobert_score' else False) for name in BIOBERT_FEATURES}


def same_old(current, previous):
    require(len(current) >= len(previous), 'Old candidate prefix was shortened')
    for new, old in zip(current, previous):
        reject_old_posterior(old)
        reject_old_posterior(new)
        require(not set(FEATURES) & set(old) and set(new) == set(old) | set(FEATURES),
                'Unexpected old-prefix feature schema')
        projected = {name: value for name, value in new.items() if name not in FEATURES}
        require(same_json(projected, old), 'Old-prefix values/types/key order changed')


def check_canonical(candidate):
    require(isinstance(candidate, dict) and set(candidate) == CANONICAL_FIELDS,
            'Canonical NCR row must have exactly the seven frozen fields')
    require(isinstance(candidate['pmc_id'], str) and bool(candidate['pmc_id']), 'Invalid canonical document ID')
    require(type(candidate['offset']) is int and candidate['offset'] >= 0 and
            type(candidate['length']) is int and candidate['length'] > 0,
            'Invalid canonical coordinates')
    require(isinstance(candidate['identifier'], str) and
            re.fullmatch(r'HP:[0-9]{7}', candidate['identifier']), 'Invalid canonical HPO ID format')
    require(isinstance(candidate['text'], str) and len(candidate['text']) == candidate['length'],
            'Canonical text/length mismatch')
    require(candidate['ncr_present'] is True and type(candidate['ncr_score']) is float and
            math.isfinite(candidate['ncr_score']) and 0.0 <= candidate['ncr_score'] <= 1.0,
            'Canonical NCR source must be observed with a finite float score in [0,1]')


def merge(previous, supplied, doc):
    """Copy old prefix; update only NCR fields; append new exact keys in source order."""
    require(previous['pmc_id'] == doc['pmc_id'], 'Pool/input document mismatch')
    old = previous['candidates']
    for candidate in old:
        reject_old_posterior(candidate)
        require(not set(FEATURES) & set(candidate), 'Input pool already has NCR fields')
        require(all(name in candidate for name in OLD_FEATURES), 'Missing frozen public-source feature')
    out = [dict(candidate, **defaults()) for candidate in old]
    indexed = {key(candidate): candidate for candidate in out}
    require(len(indexed) == len(out), 'Duplicate old candidate key')
    crf_spans = {key(candidate)[:2] for candidate in old if candidate['crf_span']}
    supplied_keys = set()
    for candidate in supplied:
        check_canonical(candidate)
        require(candidate['pmc_id'] == doc['pmc_id'], 'Canonical row belongs to another document')
        identity = key(candidate)
        require(identity not in supplied_keys, 'Duplicate canonical key within document')
        supplied_keys.add(identity)
        offset, length = identity[:2]
        passages = [p for p in doc['full_text'] if p['offset'] <= offset and
                    offset + length <= p['offset'] + len(p['text'])]
        require(len(passages) == 1, 'Canonical span does not belong to exactly one original passage')
        passage = passages[0]
        left = offset - passage['offset']
        require(passage['text'][left:left + length] == candidate['text'], 'Canonical span text mismatch')
        if identity not in indexed:
            target = {name: candidate[name] for name in ('offset', 'length', 'identifier', 'text')}
            # Keep the previous public-source merge's explicit missing-source values/types.
            target.update(probability=.5, similarity=0., exact=False, is_base=False,
                          crf=False, crf_span=identity[:2] in crf_spans, semantic_source=False,
                          **base_missing_defaults(), **defaults())
            indexed[identity] = target
            out.append(target)
        target = indexed[identity]
        require(target['text'] == candidate['text'], 'Shared key has different original text')
        target.update({name: candidate[name] for name in FEATURES})
    same_old(out, old)
    require(len({key(candidate) for candidate in out}) == len(out), 'Duplicate merged candidate key')
    for candidate in out:
        reject_old_posterior(candidate)
        require(type(candidate['ncr_present']) is bool and type(candidate['ncr_score']) is float,
                'NCR feature types changed')
    return dict(previous, candidates=out)


def validate_base_metadata(base):
    require(base['status'] == 'prepared_input_only_no_fit_or_quality' and
            list(base['new_features']) == BIOBERT_FEATURES and
            base['old_rows'] == PREB_POOL_ROWS and base['new_rows'] == BASE_POOL_ROWS,
            'Expected the completed strong BioBERT eight-pool manifest')
    folds = base['folds']
    require([fold['fold'] for fold in folds] == [0, 1, 2, 3], 'Expected exactly four frozen folds')
    held = [set(fold['input_ids']) for fold in folds]
    universe = set().union(*held)
    require(len(universe) == 64 and sum(len(ids) for ids in held) == 64 and
            all(len(ids) == 16 for ids in held), 'Base held folds are not disjoint 16-document sets')
    for fold in folds:
        train_ids, input_ids = fold['train_ids'], fold['input_ids']
        nested = fold['nested_input_ids']
        require(len(train_ids) == len(set(train_ids)) == 48 and len(input_ids) == 16 and
                set(train_ids) == universe - set(input_ids), 'Base 48/16 split changed')
        require(len(nested) == 3 and all(len(ids) == len(set(ids)) == 16 for ids in nested) and
                [did for ids in nested for did in ids] == train_ids,
                'Base nested 32-to-16 origin order changed')
        require(fold['old_held_candidates_exact'] is True and
                isinstance(fold['old_fit_manifest'], dict) and 'sha256' in fold['old_fit_manifest'],
                'Missing inherited preB generator metadata anchor')
    require(sum(fold['new_train_rows'] + fold['new_input_rows'] for fold in folds) == BASE_POOL_ROWS,
            'Strong BioBERT base count must use new_*, not historical preB old_*')
    require(sum(fold['old_train_rows'] + fold['old_input_rows'] for fold in folds) == PREB_POOL_ROWS,
            'Inherited historical preB counts changed')
    return universe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan = read(args.plan)
    plan_sha = sha(args.plan)
    sources(plan)
    require(plan['base_prepared_manifest']['sha256'] == BASE_MANIFEST_SHA256,
            'Plan must use the immutable strong BioBERT eight-pool manifest')
    base = read(checked(plan['base_prepared_manifest']))
    universe = validate_base_metadata(base)
    require([item['fold'] for item in plan['folds']] == [0, 1, 2, 3], 'Missing strong base-fit manifest refs')
    cm = read(checked(plan['cache_manifest']))
    require(cm['status'] == CACHE_STATUS and cm['canonical'] == plan['canonical'],
            'Canonical rows are not bound to the completed verified NCR cache manifest')
    require(cm['raw_spans'] == RAW_SPAN_UPPER_BOUND and
            isinstance(cm['refs'], dict) and bool(cm['refs']), 'Missing NCR raw-cache/source lineage')
    for role, item in cm['refs'].items():
        require(isinstance(item, dict) and 'path' in item and 'sha256' in item and
                re.fullmatch(r'[0-9a-f]{64}', item['sha256']), 'Invalid NCR source ref: ' + str(role))
    canonical = rows(checked(plan['canonical']))
    require(len(canonical) <= RAW_SPAN_UPPER_BOUND, 'Canonical count exceeds native raw-span sanity bound')
    supplied = defaultdict(list)
    canonical_keys = set()
    for candidate in canonical:
        check_canonical(candidate)
        identity = fullkey(candidate)
        require(identity not in canonical_keys and candidate['pmc_id'] in universe,
                'Duplicate canonical fullkey or document outside frozen train64')
        canonical_keys.add(identity)
        supplied[candidate['pmc_id']].append(candidate)
    output = Path(plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    folds = []
    total_base = total_new = total_matching = total_source = 0
    observed_documents = {}
    for prior in base['folds']:
        fold = copy.deepcopy(prior)
        fold['base_fold'] = copy.deepcopy(prior)
        fold['base_pool_refs'] = {role: prior[role] for role in ('train', 'input')}
        fold['base_fit_manifest'] = plan['folds'][prior['fold']]['base_fit_manifest']
        base_fit = read(checked(fold['base_fit_manifest']))
        require(base_fit['status'] == 'fit_and_input_scores_complete_no_decode_or_quality' and
                base_fit['fold'] == prior['fold'] and base_fit['source_sha256'] == BASE_FIT_SOURCE_SHA256 and
                base_fit['prepared_manifest'] == plan['base_prepared_manifest'] and
                base_fit['old_fit_manifest'] == prior['old_fit_manifest'], 'Strong base-fit lineage mismatch')
        require(base_fit['training_examples'] == prior['new_train_rows'] and
                base_fit['input_candidates'] == prior['new_input_rows'] and
                list(base_fit['new_features']) == BIOBERT_FEATURES and
                set(BIOBERT_FEATURES) <= set(base_fit['features']), 'Strong fitted counts/features mismatch')
        fold['base_train_rows'] = prior['new_train_rows']
        fold['base_input_rows'] = prior['new_input_rows']
        directory = output / ('fold' + str(prior['fold']))
        directory.mkdir()
        for role in ('train', 'input'):
            before = rows(checked(prior[role]))
            require(len(before) == (48 if role == 'train' else 16) and
                    [row['pmc_id'] for row in before] == prior[role + '_ids'],
                    'Pool document count/order differs from frozen manifest')
            after = []
            nbase = nnew = nmatching = nsource = 0
            for row in before:
                require(set(row) == {'pmc_id', 'input', 'base', 'candidates', 'origin_index'} and
                        set(row['input']) == {'pmc_id', 'pmid', 'patient', 'full_text'} and
                        row['pmc_id'] == row['input']['pmc_id'], 'Unexpected input-only pool row schema')
                did = row['pmc_id']
                expected_origin = (next(i for i, ids in enumerate(prior['nested_input_ids']) if did in ids)
                                   if role == 'train' else None)
                require(row['origin_index'] == expected_origin, 'Pool origin_index changed')
                if did in observed_documents:
                    require(same_json(row['input'], observed_documents[did]), 'Input differs between folds')
                else:
                    observed_documents[did] = row['input']
                previous_keys = {key(candidate) for candidate in row['candidates']}
                current = merge(row, supplied[did], row['input'])
                require(same_json({k: v for k, v in current.items() if k != 'candidates'},
                                  {k: v for k, v in row.items() if k != 'candidates'}),
                        'Pool row metadata/base/input changed')
                after.append(current)
                nbase += len(row['candidates'])
                nnew += len(current['candidates'])
                nsource += len(supplied[did])
                nmatching += sum(key(candidate) in previous_keys for candidate in supplied[did])
            require(nbase == prior['new_' + role + '_rows'] and nnew >= nbase and
                    nnew - nbase + nmatching == nsource,
                    'Merged exact-key row accounting mismatch')
            target = directory / (role + '.pool.jsonl.gz')
            data = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in after).encode('utf-8')
            with target.open('xb') as handle:
                handle.write(gzip.compress(data, mtime=0))
            fold[role] = ref(target)
            fold['new_' + role + '_rows'] = nnew
            fold['added_' + role + '_rows'] = nnew - nbase
            fold['matching_' + role + '_rows'] = nmatching
            fold['source_' + role + '_rows'] = nsource
            fold[role + '_row_ratio'] = nnew / nbase
            total_base += nbase
            total_new += nnew
            total_matching += nmatching
            total_source += nsource
        folds.append(fold)
    require(set(observed_documents) == universe and total_base == BASE_POOL_ROWS,
            'Final base-pool coverage/count mismatch')
    require(total_source == 4 * len(canonical) and total_new - total_base + total_matching == total_source and
            total_new - total_base <= 4 * RAW_SPAN_UPPER_BOUND,
            'Four-fold exact-key/source multiplicity sanity bound failed')
    sources(plan)
    require(sha(args.plan) == plan_sha, 'Plan changed during preparation')
    checked(plan['base_prepared_manifest'])
    checked(plan['cache_manifest'])
    checked(plan['canonical'])
    result = {
        'status': 'prepared_input_only_no_fit_or_quality',
        'source_sha256': sha(__file__), 'plan_sha256': plan_sha,
        'base_prepared_manifest': plan['base_prepared_manifest'],
        'canonical': plan['canonical'], 'cache_manifest': plan['cache_manifest'],
        'cache_source_refs': cm['refs'], 'raw_native_spans': cm['raw_spans'],
        'canonical_unique_keys': len(canonical), 'canonical_documents_with_rows': len({c['pmc_id'] for c in canonical}),
        'input_documents': len(universe), 'folds': folds, 'new_features': FEATURES,
        'preserved_source_features': OLD_FEATURES, 'base_cache_refs': base['cache'],
        'base_preB_prepared_manifest': base['old_prepared_manifest'],
        'base_rows': total_base, 'new_rows': total_new, 'added_rows': total_new - total_base,
        'matching_rows': total_matching, 'source_rows_across_eight_pools': total_source,
        'maximum_train_row_ratio': max(fold['train_row_ratio'] for fold in folds),
        'base_candidate_projection_values_types_order_exact': True,
        'base_input_base_origin_index_and_lineage_unchanged': True,
        'base_BioBERT_prefix_preserved': True, 'no_old_selector_posteriors': True,
        'new_key_order': 'First occurrence in frozen unique canonical NCR cache; original candidate prefix retained.',
        'score_semantics': 'Supplied observed NCR current-key float score, already max of actual native float32 values upstream; no new threshold, rounding, NN or overlap rule.',
        'sanity_bound': {'maximum_added_pool_rows': 4 * RAW_SPAN_UPPER_BOUND,
                         'reason': 'Each canonical key occurs once in each of four outer folds; not a row-ratio budget or trimming policy.'},
        'no_new_NN': True, 'no_labels_read': True, 'no_fit': True,
        'no_prediction_or_quality': True, 'seconds': time.monotonic() - started
    }
    write(output / 'manifest.json', result)
    print(json.dumps({name: result[name] for name in
                      ('status', 'base_rows', 'new_rows', 'added_rows', 'matching_rows', 'seconds')}))


if __name__ == '__main__':
    main()
