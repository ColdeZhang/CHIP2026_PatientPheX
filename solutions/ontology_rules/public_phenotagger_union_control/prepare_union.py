"""Input-only old .95 union native .5 public candidates, with observed evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check(item):
    assert sha(item['path']) == item['sha256'], item['path']
    return Path(item['path'])


def key(row):
    return str(row['pmc_id']), row['offset'], row['length'], row['hpo_id']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    for p, digest in plan['sources'].items():
        assert sha(p) == digest, p
    assert plan['method']['pool'] == 'old095_union_new05'
    for spec in plan['public_inputs'].values():
        for item in spec.values():
            check(item)
    old_manifest = json.loads(check(plan['public_inputs']['old095']['manifest']).read_text())
    assert old_manifest['public_model_training_on_task'] is False
    assert old_manifest['model_weight_sha256'] == plan['model_weight_sha256']
    old_prepared = json.loads(check(plan['old_prepared_manifest']).read_text())
    assert old_prepared['output_sha256'] == plan['old_prepared']['sha256']
    assert old_prepared['public_model_task_fit'] is False
    assert old_prepared['model_weight_sha256'] == plan['model_weight_sha256']
    census = json.loads(check(plan['public_inputs']['new05']['summary']).read_text())
    assert census['plan_sha256'] == plan['public_inputs']['new05']['plan']['sha256']
    assert census['original095_byteexact_before05'] and census['quality_labels_read'] is False
    assert census['proposed05']['canonical_sha256'] == plan['public_inputs']['new05']['canonical']['sha256']
    assert census['proposed05']['provenance_sha256'] == plan['public_inputs']['new05']['provenance']['sha256']
    docs = read(check(plan['input64']))
    docmap = {str(d['pmc_id']): d for d in docs}
    assert len(docmap) == len(docs) == 64
    assert all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs)
    assert old_manifest['input_sha256'] == census['input_sha256'] == plan['input64']['sha256']
    sys.path.insert(0, plan['ontology_source'])
    import run as base
    _, allowed, _, _ = base.parse_hpo(check(plan['hpo']))
    allowed.discard('HP:0000118')
    pool = {}
    arm_keys = []
    flags = ['raw_neural_match', 'raw_dictionary_match', 'abbrev_inferred']
    for arm in ['old095', 'new05']:
        spec = plan['public_inputs'][arm]
        rows = read(check(spec['canonical']))
        sidecar = read(check(spec['provenance']))
        provenance = {key(r): r for r in sidecar}
        assert len(provenance) == len(sidecar)
        keys = set()
        for r in rows:
            k = key(r)
            assert k not in keys and k in provenance
            keys.add(k)
            assert r['hpo_id'] in allowed
            doc = docmap[k[0]]
            passage = doc['full_text'][r['passage_index']]
            lo, hi = r['local_start'], r['local_end']
            assert 0 <= lo < hi <= len(passage['text'])
            assert r['text'] == passage['text'][lo:hi]
            assert r['offset'] == passage['offset']+lo and r['length'] == hi-lo
            assert math.isfinite(r['score']) and 0 <= r['score'] <= 1
            p = provenance[k]
            assert all(type(p[f]) is bool for f in flags) and any(p[f] for f in flags)
            assert r['raw_hpo_id'] in p['raw_hpo_ids']
            assert r['score'] in p['actual_hybrid_scores']
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in p['actual_hybrid_scores'])
            assert all(type(c) is int and 0 <= c < 17266 for c in p['matched_neural_head_columns'])
            if k not in pool:
                pool[k] = {'representative': r, 'arms': [], 'provenance': {
                    **dict.fromkeys(flags, False), 'raw_hpo_ids': [],
                    'matched_neural_head_columns': [], 'actual_hybrid_scores': []}}
            item = pool[k]
            for field in ['passage_index', 'local_start', 'local_end', 'text']:
                assert item['representative'][field] == r[field]
            # Retain a real observed row, using old-first stable tie handling.
            if r['score'] > item['representative']['score']:
                item['representative'] = r
            item['arms'].append(arm)
            for f in flags:
                item['provenance'][f] |= p[f]
            for f in ['raw_hpo_ids', 'matched_neural_head_columns', 'actual_hybrid_scores']:
                item['provenance'][f] = sorted(set(item['provenance'][f]) | set(p[f]))
        assert keys == set(provenance)
        arm_keys.append(keys)
    old, new = arm_keys
    assert len(old) == 6064 and len(new) == 8218
    assert len(old & new) == 6061 and len(old | new) == 8221
    assert old <= pool.keys() and new <= pool.keys()
    prepared = {did: [] for did in docmap}
    merged_rows, merged_flags = [], []
    for k, item in pool.items():
        r, p = item['representative'], item['provenance']
        score = max(p['actual_hybrid_scores'])
        assert score == r['score']  # Never synthesize a neural posterior.
        merged_rows.append(r)
        merged_flags.append({'pmc_id': k[0], 'offset': k[1], 'length': k[2],
                             'hpo_id': k[3], **p, 'observed_in': item['arms']})
        prepared[k[0]].append(dict(offset=k[1], length=k[2], text=r['text'], identifier=k[3],
            public_present=True, public_score=score, public_neural=p['raw_neural_match'],
            public_dictionary=p['raw_dictionary_match'], public_abbreviation=p['abbrev_inferred'],
            public_origin_unknown=False, public_raw_hpo_ids=p['raw_hpo_ids'],
            public_matched_neural_head_columns=p['matched_neural_head_columns']))
    # Check old adapter semantics against the independently frozen old pool.
    prior = read(check(plan['old_prepared']))
    before = {(str(d['pmc_id']), c['offset'], c['length'], c['identifier']): c
              for d in prior for c in d['candidates']}
    assert set(before) == old
    changed = Counter()
    for did, cs in prepared.items():
        for c in cs:
            k = did, c['offset'], c['length'], c['identifier']
            if k not in before:
                continue
            b = before[k]
            for name in ['public_neural', 'public_dictionary', 'public_abbreviation']:
                assert not b[name] or c[name]
                changed[name] += b[name] != c[name]
            assert c['public_score'] >= b['public_score']
            changed['public_score'] += c['public_score'] != b['public_score']
    out = Path(plan['output_dir'])/'prepared'
    out.mkdir(parents=True, exist_ok=False)
    values = {'candidates.jsonl': [{'pmc_id': did, 'candidates': cs} for did, cs in prepared.items()],
              'canonical-entities.jsonl': merged_rows, 'source-provenance.jsonl': merged_flags}
    for name, rows in values.items():
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    manifest = dict(source_sha256=sha(__file__), plan_sha256=sha(a.plan),
        public_model_task_fit=False, model_weight_sha256=plan['model_weight_sha256'],
        input_path=plan['input64']['path'], input_sha256=plan['input64']['sha256'],
        hpo_sha256=plan['hpo']['sha256'], output_sha256=sha(out/'candidates.jsonl'),
        documents=64, rows=len(pool), unique_keys=len(pool), annotations_read=False,
        method='Explicit old095 union new05 candidate-and-provenance increment; flags OR/observed score max.',
        old_keys_retained=len(old), new_only=len(new-old), old_only_retained=len(old-new),
        previous_key_feature_changes=dict(changed), parents=plan['public_inputs'],
        outputs={n: {'sha256': sha(out/n), 'bytes': (out/n).stat().st_size} for n in values})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
