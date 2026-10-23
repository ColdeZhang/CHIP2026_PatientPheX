#!/usr/bin/env python3
"""Learn patient-concept attribution from document-isolated OOF candidates.

Preparation emits training-only coverage diagnostics before any model is fit.
Candidate occurrence features never read annotations or identity strings.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
import calibrate_multi as multi

base = multi.single.baseline


def keys(entity):
    return [entity['text']] if entity['identifier'] == '-1' else entity['identifier'].split(';')


def gold_links(doc):
    result = set()
    for row in doc.get('association', []):
        for value in row.get('phenotype', []):
            values = value.split(';') if re.fullmatch(r'HP:\d{7}(?:;HP:\d{7})+', value) else [value]
            result.update((row['patient_id'], v) for v in values)
    return result


def reduce_occurrences(items):
    """Symmetric set evidence: no chosen occurrence depends on gold."""
    out = {'log_occurrences': math.log1p(len(items))}
    numerical = set().union(*(set(k for k, v in x.items() if isinstance(v, (bool, int, float))) for x in items))
    categories = Counter((k, str(v)) for x in items for k, v in x.items() if isinstance(v, str))
    for k in numerical:
        values = [float(x.get(k, 0.)) for x in items]
        out[k + '_max'] = max(values)
        out[k + '_min'] = min(values)
        out[k + '_mean'] = sum(values) / len(values)
    for (k, v), count in categories.items():
        out[k + '=' + v] = count / len(items)
    return out


def document_rows(doc, baseline, raw, crf, sequence):
    assert 'entities' not in doc and 'association' not in doc
    candidates = multi.pool(doc, baseline, raw, crf)
    anchors = base.patient_anchors(doc)
    states = multi.single.enhance.active_by_passage(doc, anchors)
    grouped = defaultdict(list)
    support = defaultdict(list)
    for c in candidates:
        pi, passage, start, end = multi.single.enhance.location(doc, c['offset'], c['offset'] + c['length'])
        pstart = int(passage.get('offset', 0)); text = passage['text']
        left, right = base.clause_bounds(text, start, end)
        clause = text[left:right]
        local = [a for a in anchors if pstart + left <= a[0] < pstart + right]
        previous = [a for a in anchors if a[0] <= c['offset']]
        mention = base.Mention(start, end, base.token_key(c['text']), tuple(c['identifier'].split(';')), 0)
        assigned = base.assign_patients(doc, passage, mention, anchors, states.get(pi))
        common = multi.single.features(doc, c, baseline)
        common.update(crf=c['crf'], crf_span=c['crf_span'], semantic_source=c['semantic_source'])
        for concept in keys(c):
            support[concept].append({'candidate': c, 'rule_patients': assigned,
                'local_patients': {a[2] for a in local}, 'negated': bool(common['negated'])})
        for patient in doc.get('patient', []):
            pid = patient['patient_id']
            own = [a for a in anchors if a[2] == pid]
            other = [a for a in anchors if a[2] != pid]
            distance = min([abs(a[0] - c['offset']) for a in own] or [100000])
            other_distance = min([abs(a[0] - c['offset']) for a in other] or [100000])
            feats = dict(common)
            feats.update({'pair_' + k: v for k, v in sequence.pair_features(doc, c, patient).items()})
            feats.update(rule_selected=pid in assigned, local_anchor=any(a[2] == pid for a in local),
                other_local_anchor=any(a[2] != pid for a in local),
                nearest_preceding=bool(previous and previous[-1][2] == pid),
                state_patient=states.get(pi) == pid, has_expanded_anchor=bool(own),
                expanded_log_distance=math.log1p(distance), expanded_nearest=distance <= other_distance,
                expanded_margin=max(-5., min(5., math.log1p(other_distance) - math.log1p(distance))),
                negated_for_target=bool(common['negated'] and (distance <= other_distance)),
                explicit_cohort=bool(re.search(r'\b(?:both|all|these|our) (?:affected )?(?:patients|siblings|individuals|children)\b', clause, re.I)))
            for concept in keys(c):
                grouped[(pid, concept)].append(feats)
    rows = [{'pmc_id': doc['pmc_id'], 'patient_id': pid, 'concept': concept,
             'features': reduce_occurrences(items)} for (pid, concept), items in sorted(grouped.items())]
    return rows, support, anchors


def diagnostics(gold, support, anchors):
    links = gold_links(gold)
    predicted = {(pid, concept) for concept, occurrences in support.items()
                 for occurrence in occurrences for pid in occurrence['rule_patients']}
    positive_entities = defaultdict(list)
    for e in gold.get('entities', []):
        if str(e.get('note', '')).upper() != 'NO':
            for k in keys(e): positive_entities[k].append(e)
    result = Counter(documents=1, gold_relations=len(links), rule_tp=len(predicted & links),
                     rule_fp=len(predicted - links), rule_fn=len(links - predicted))
    for pid, concept in links:
        occurrences = support.get(concept, [])
        if not occurrences:
            result['gold_no_candidate_concept'] += 1
            continue
        result['gold_candidate_concept_available'] += 1
        exact = {(e['offset'], e['length']) for e in positive_entities.get(concept, [])}
        result['gold_exact_positive_span_available'] += any(
            (o['candidate']['offset'], o['candidate']['length']) in exact for o in occurrences)
        own = [a for a in anchors if a[2] == pid]
        result['gold_target_no_document_anchor'] += not bool(own)
        result['gold_target_no_local_clause_anchor'] += not any(pid in o['local_patients'] for o in occurrences)
        result['gold_all_occurrences_negated'] += all(o['negated'] for o in occurrences)
        if (pid, concept) not in predicted:
            result['gold_available_but_rule_missed'] += 1
            result['gold_rule_missed_assigned_other'] += any(o['rule_patients'] - {pid} for o in occurrences)
    return result


def load_sequence(path):
    spec = importlib.util.spec_from_file_location('sequence_frozen', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write_gzip(path, rows):
    with gzip.open(path, 'wt') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_gzip(path):
    with gzip.open(path, 'rt') as f: return [json.loads(line) for line in f]


def prepare(a):
    started = time.monotonic(); out = Path(a.work_dir); out.mkdir(parents=True, exist_ok=True)
    sequence = load_sequence(a.sequence_source)
    training = multi.keyed(a.train); train_ids = set(training)
    inference = multi.keyed(a.input)
    assert not train_ids & set(inference)
    specs = json.loads(Path(a.training_manifest).read_text())
    rows, seen, stats, origins = [], set(), Counter(), []
    for spec in specs:
        gold = multi.keyed(spec['gold']); inputs = multi.keyed(spec['input'])
        assert set(gold) == set(inputs) and not seen & set(inputs)
        seen |= set(inputs)
        generator_ids = set(multi.keyed(spec['generator_train']))
        assert generator_ids <= train_ids and not generator_ids & set(inputs)
        manifest = json.loads(Path(spec['candidate_manifest']).read_text())
        assert manifest['inputs']['train'] == base.sha256(spec['generator_train'])
        assert manifest['inputs']['input'] == base.sha256(spec['input'])
        assert manifest['output']['sha256'] == base.sha256(spec['candidates'])
        bases = multi.keyed(spec['base']); raw = multi.single.read_candidates(spec['candidates'])
        crfs = multi.checked_crf(spec['crf'], spec['crf_manifest'], spec['generator_train'], spec['input'], train_ids)
        for did, doc in inputs.items():
            assert gold[did] == training[did]
            current, support, anchors = document_rows(doc, bases[did], raw[did], crfs[did], sequence)
            truth = gold_links(gold[did])
            for row in current:
                row['target'] = (row['patient_id'], row['concept']) in truth
            rows.extend(current); stats.update(diagnostics(gold[did], support, anchors))
        origins.append({k: base.sha256(spec[k]) for k in spec})
    assert seen == train_ids
    write_gzip(out / 'training-rows.jsonl.gz', rows)
    stats['training_examples'] = len(rows); stats['training_positive'] = sum(row['target'] for row in rows)
    bases = multi.keyed(a.base); raw = multi.single.read_candidates(a.candidates)
    crfs = multi.checked_crf(a.crf, a.crf_manifest, a.train, a.input, train_ids)
    input_rows = []
    for did, doc in inference.items():
        current, _, _ = document_rows(doc, bases[did], raw[did], crfs[did], sequence)
        input_rows.extend(current)
    write_gzip(out / 'input-rows.jsonl.gz', input_rows)
    (out / 'training-diagnostics.json').write_text(json.dumps(dict(stats), indent=2) + '\n')
    manifest = {'config': vars(a), 'source_sha256': base.sha256(__file__),
        'dependencies': {m.__name__: base.sha256(m.__file__) for m in [multi, multi.single, base, multi.single.enhance]},
        'sequence_source_sha256': base.sha256(a.sequence_source), 'training_origins': origins,
        'input_sha256': {k: base.sha256(getattr(a, k)) for k in
            ['train', 'input', 'base', 'candidates', 'crf', 'crf_manifest', 'entity_prediction', 'training_manifest']},
        'output_sha256': {p.name: base.sha256(p) for p in out.glob('*-rows.jsonl.gz')},
        'training_diagnostics': dict(stats), 'input_pairs': len(input_rows),
        'input_annotations_used': False, 'seconds': time.monotonic() - started}
    (out / 'preparation.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'training_diagnostics': dict(stats), 'input_pairs': len(input_rows), 'seconds': manifest['seconds']}))


def fit(a):
    out = Path(a.work_dir); prep = json.loads((out / 'preparation.manifest.json').read_text())
    assert prep['source_sha256'] == base.sha256(__file__)
    assert prep['input_sha256']['entity_prediction'] == base.sha256(a.entity_prediction)
    for name, value in prep['output_sha256'].items(): assert base.sha256(out / name) == value
    train = read_gzip(out / 'training-rows.jsonl.gz'); inputs = read_gzip(out / 'input-rows.jsonl.gz')
    assert not {r['pmc_id'] for r in train} & {r['pmc_id'] for r in inputs}
    assert all('target' not in r for r in inputs)
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform([r['features'] for r in train])
    model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=5., learning_rate=.06, random_state=20261023, early_stopping=False)
    started = time.monotonic(); model.fit(x, [r['target'] for r in train])
    scores = model.predict_proba(vectorizer.transform([r['features'] for r in inputs]))[:, 1]
    predictions = base.read_jsonl(a.entity_prediction)
    by_doc = defaultdict(list)
    for r, p in zip(inputs, scores): by_doc[r['pmc_id']].append((r, float(p)))
    results = {}
    for policy in ['entities', 'pool']:
        output = []
        for pred in predictions:
            allowed = {k for e in pred['entities'] for k in keys(e)}
            links = {r['patient_id']: set() for r in pred['association']}
            for row, probability in by_doc[pred['pmc_id']]:
                if probability >= .5 and (policy == 'pool' or row['concept'] in allowed):
                    links[row['patient_id']].add(row['concept'])
            output.append(dict(pred, association=[{'patient_id': pid, 'phenotype': sorted(values)} for pid, values in links.items()]))
        p = out / policy / 'pred.jsonl'; p.parent.mkdir(exist_ok=True)
        p.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
        results[policy] = {'path': str(p), 'sha256': base.sha256(p)}
    manifest = {'source_sha256': base.sha256(__file__), 'preparation_sha256': base.sha256(out / 'preparation.manifest.json'),
        'estimator': model.get_params(), 'features': len(vectorizer.feature_names_), 'threshold': .5,
        'entity_outputs_unchanged': True, 'input_annotations_used': False, 'results': results,
        'training_seconds': time.monotonic() - started}
    (out / 'fit.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['prepare', 'fit'])
    for name in ['work-dir', 'entity-prediction']:
        ap.add_argument('--' + name, required=True)
    for name in ['train', 'input', 'training-manifest', 'sequence-source', 'base', 'candidates', 'crf', 'crf-manifest']:
        ap.add_argument('--' + name)
    a = ap.parse_args()
    if a.phase == 'prepare': prepare(a)
    else: fit(a)


if __name__ == '__main__': main()
