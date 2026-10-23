#!/usr/bin/env python3
"""Freeze the 384 existing boundary-missing FN atoms as diagnostic queries.

These gold-indexed records are diagnostic queries, never candidates or
predictions.  Matching and overlap come from the frozen train64 FN source.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

FN_PLAN_SHA = '802dbee3472cdaba41d323eb6e05bd0fe365251ae7df2b7d37a73d655faeac2f'
FN_SUMMARY_SHA = 'ec0ee958a888843204febbecb9f4cb37b44f83688f267c42d9bc965b5f8bbebc'
OLD_PREPARER_SHA = '2e6f514e31c531587241dc314449937515efa678d7e10c2247e7c63ed3976802'
MECHANISMS = ('pool_missing_exact_span_overlap_only', 'pool_missing_any_overlapping_span')
EXPECTED = dict(zip(MECHANISMS, (269, 115)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked(item):
    path = Path(item['path'])
    assert path.is_absolute() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes'], str(path)
    return path


def rows(path):
    result = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [str(row['pmc_id']) for row in result]
    assert len(ids) == len(set(ids)), str(path)
    return result


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boundary_atoms(gold_units, predicted_units, candidates, fn):
    """Use the frozen NIL consumption, then split only spans absent from pool."""
    pool_spans = {fn.span(candidate) for candidate in candidates}
    for span, gold_atoms in sorted(gold_units.items()):
        missed, _ = fn.unmatched(gold_atoms, predicted_units.get(span, set()))
        if not missed or span in pool_spans:
            continue
        mechanism = MECHANISMS[0] if any(fn.overlap(span, other) for other in pool_spans) else MECHANISMS[1]
        for literal_id in sorted(missed):
            yield span, literal_id, sorted(gold_atoms), mechanism


def query_record(index, fold, document, span, literal_id, gold_atoms, mechanism):
    offset, length = span
    passages = [(i, passage) for i, passage in enumerate(document['full_text'])
                if passage['offset'] <= offset and
                offset + length <= passage['offset'] + len(passage['text'])]
    assert len(passages) == 1, (document['pmc_id'], span)
    passage_index, passage = passages[0]
    local = offset - passage['offset']
    text = passage['text'][local:local + length]
    assert length > 0 and len(text) == length
    assert any(entity['offset'] == offset and entity['length'] == length and
               entity['text'] == text and str(entity.get('note', '')).upper() != 'NO' and
               literal_id in entity['identifier'].split(';') for entity in document['entities'])
    return dict(query_index=index, fold=fold, pmc_id=str(document['pmc_id']),
                offset=offset, length=length, passage_index=passage_index,
                passage_offset=passage['offset'], local_start=local, local_end=local + length,
                text=text, gold_literal_id=literal_id, gold_span_atomic_ids=gold_atoms,
                mechanism=mechanism)


def denominators(queries):
    mechanisms = Counter(query['mechanism'] for query in queries)
    by_kind = Counter('unlinked' if query['gold_literal_id'] == '-1' else 'linked' for query in queries)
    return dict(query_atoms=len(queries), documents=len({query['pmc_id'] for query in queries}),
                unique_spans=len({(query['pmc_id'], query['offset'], query['length']) for query in queries}),
                by_kind={kind: by_kind[kind] for kind in ('linked', 'unlinked')},
                mechanisms={name: mechanisms[name] for name in MECHANISMS},
                mechanisms_by_kind={name: {
                    kind: sum(query['mechanism'] == name and
                              ('unlinked' if query['gold_literal_id'] == '-1' else 'linked') == kind
                              for query in queries) for kind in ('linked', 'unlinked')}
                    for name in MECHANISMS})


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fn-plan', 'fn-summary', 'output', 'summary'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sys.dont_write_bytecode = True
    output, summary_path = Path(args.output).resolve(), Path(args.summary).resolve()
    assert output != summary_path and not output.exists() and not summary_path.exists()
    assert sha(args.fn_plan) == FN_PLAN_SHA and sha(args.fn_summary) == FN_SUMMARY_SHA
    plan = json.loads(Path(args.fn_plan).read_text())
    previous = json.loads(Path(args.fn_summary).read_text())
    assert previous['plan_sha256'] == FN_PLAN_SHA
    assert plan['cut'] == previous['cut'] == .3
    assert previous['source_sha256'] == plan['source_sha256'] == sha(plan['source_path'])
    assert [fold['fold'] for fold in plan['folds']] == [0, 1, 2, 3]
    for item in plan['references'].values():
        checked(item)
    old_preparer = Path(__file__).resolve().parents[1] / 'public_phenotagger_top3_diagnostic' / 'prepare_queries.py'
    assert sha(old_preparer) == OLD_PREPARER_SHA
    prior_documents = json.loads(checked(previous['document_counts']).read_text())
    assert len(prior_documents) == 64
    prior_by_id = {str(row['pmc_id']): row for row in prior_documents}
    assert len(prior_by_id) == 64
    evaluator_path = checked(plan['references']['evaluator'])
    sys.path.insert(0, str(evaluator_path.parent))
    evaluator = module_from_path('frozen_boundary_evaluator', evaluator_path)
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    fn = module_from_path('frozen_boundary_fn', plan['source_path'])
    fixtures = fn.fixtures(evaluator)
    gold = {str(doc['pmc_id']): doc for doc in rows(plan['references']['train64']['path'])}
    assert len(gold) == 64
    queries, folds, seen = [], [], set()
    pooled_counts = {name: Counter() for name in ('mention', 'document')}
    pooled_metrics = defaultdict(Counter)
    for fold in plan['folds']:
        scores = rows(plan['references'][fold['scores_key']]['path'])
        predictions = rows(plan['references'][fold['prediction_key']]['path'])
        assert len(scores) == len(predictions) == 16
        assert [str(row['pmc_id']) for row in scores] == [str(row['pmc_id']) for row in predictions]
        local_queries = []
        fold_counts = {name: Counter() for name in pooled_counts}
        fold_metrics = defaultdict(Counter)
        for scored, predicted in zip(scores, predictions):
            did = str(predicted['pmc_id'])
            assert did in gold and did not in seen
            seen.add(did)
            document = gold[did]
            counts, metrics = fn.analyze_document(document, predicted, scored['candidates'], evaluator, plan['cut'])
            assert prior_by_id[did]['fold'] == fold['fold']
            assert counts == prior_by_id[did]['counts']
            assert metrics == prior_by_id[did]['aggregates']
            for name, values in counts.items():
                fold_counts[name].update(values)
            for name, values in metrics.items():
                fold_metrics[name].update(values)
            gold_units = evaluator.positive_units(document, document, 'all')
            predicted_units = evaluator.positive_units(predicted, document, 'all')
            for span, literal_id, gold_atoms, mechanism in boundary_atoms(
                    gold_units, predicted_units, scored['candidates'], fn):
                query = query_record(len(queries), fold['fold'], document, span, literal_id, gold_atoms, mechanism)
                queries.append(query)
                local_queries.append(query)
        prior_fold = next(item for item in previous['folds'] if item['fold'] == fold['fold'])
        assert fold_counts == prior_fold['counts'] and fold_metrics == prior_fold['aggregates']
        stats = denominators(local_queries)
        for name in MECHANISMS:
            assert stats['mechanisms'][name] == fold_metrics['mention_fn_mechanisms'][name]
            for kind in ('linked', 'unlinked'):
                assert stats['mechanisms_by_kind'][name][kind] == fold_metrics['mention_fn_' + kind][name]
        folds.append(dict(fold=fold['fold'], **stats, original_summary_categories_exact=True))
        for name, values in fold_counts.items():
            pooled_counts[name].update(values)
        for name, values in fold_metrics.items():
            pooled_metrics[name].update(values)
    assert seen == gold.keys() == prior_by_id.keys()
    assert pooled_counts == previous['counts'] and pooled_metrics == previous['aggregates']
    assert len(queries) == 384
    assert len({(q['pmc_id'], q['offset'], q['length'], q['gold_literal_id']) for q in queries}) == 384
    stats = denominators(queries)
    assert stats['mechanisms'] == EXPECTED
    assert stats['by_kind'] == {'linked': 337, 'unlinked': 47}
    assert [item['query_atoms'] for item in folds] == [98, 92, 91, 103]
    references = dict(fn_plan=ref(args.fn_plan), fn_summary=ref(args.fn_summary),
                      fn_source=ref(plan['source_path']), legacy_query_preparer=ref(old_preparer),
                      document_counts=previous['document_counts'], frozen_inputs=plan['references'])
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(scope='gold-indexed diagnostic queries only, not predictions or candidates',
                   source_sha256=sha(__file__), fn_plan_sha256=FN_PLAN_SHA,
                   fn_summary_sha256=FN_SUMMARY_SHA, references=references,
                   count=len(queries), queries=queries)
    write_json(output, payload)
    summary = dict(status='384 original boundary-missing FN atoms frozen',
                   completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   source=ref(__file__), arguments=vars(args), references=references,
                   queries=ref(output), input_documents=len(gold), denominators=stats, folds=folds,
                   frozen_fn_source_fixtures=fixtures, original_document_counts_and_aggregates_exact=True,
                   original_pooled_counts_and_aggregates_exact=True, original_summary_categories_exact=True,
                   matching='Frozen unmatched: exact literal atoms first; then one remaining gold NIL consumes one remaining prediction atom. No canonicalization.',
                   query_unit='One unmatched literal gold atom at its original exact span; compounds stay separate atoms.',
                   boundary_definition='Full frozen candidate pool has no exact span; frozen half-open overlap distinguishes the two original categories.',
                   originaldev_read=False, new_fit=False, new_NN=False, GPU_used=False,
                   new_predictions_written=False, new_candidates_written=False, new_method_selected=False,
                   seconds=time.monotonic() - started)
    write_json(summary_path, summary)
    print(json.dumps(dict(queries=ref(output), summary=ref(summary_path),
                         denominators=stats, seconds=summary['seconds'])))


if __name__ == '__main__':
    main()
