#!/usr/bin/env python3
"""Aggregate only: explain frozen train64 Task1 FN without new predictions/fits."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atom_ids(c):
    return set(c['identifier'].split(';'))


def span(c):
    return c['offset'], c['length']


def overlap(a, b):
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def unmatched(truth, guess):
    """Exact v2 one-to-one set consumption, including exact -1 before wildcard."""
    truth, guess = set(truth), set(guess)
    exact = truth & guess
    truth -= exact
    guess -= exact
    if '-1' in truth and guess:
        truth.remove('-1')
        guess.remove(sorted(guess)[0])
    return truth, guess


def nil_support(candidate_ids, gold_ids):
    # Linked gold atoms are consumed before the gold NIL can use leftovers.
    return '-1' in candidate_ids or bool(candidate_ids - (gold_ids - {'-1'}))


def greedy_trace(candidates, cut):
    order = sorted(range(len(candidates)), key=lambda i: (
        -candidates[i]['selector_probability'], -candidates[i]['length'], candidates[i]['offset']))
    selected, blockers = [], {}
    for i in order:
        c = candidates[i]
        p = c['selector_probability']
        assert math.isfinite(p) and 0 <= p <= 1
        if p < cut:
            continue
        blocker = next((j for j in selected if overlap(span(c), span(candidates[j]))), None)
        if blocker is None:
            selected.append(i)
        else:
            blockers[i] = blocker
    entities = [({k: candidates[i][k] for k in ['identifier', 'offset', 'length', 'text']}
                 | {'type': 'Phenotype', 'note': None}) for i in selected]
    return sorted(entities, key=lambda e: e['offset']), set(selected), blockers, {i:r for r,i in enumerate(order)}


def fixtures(evaluator):
    s = (10, 5)
    cases = [({'-1', 'H1'}, {'H1'}, 1, 0, 1),
             ({'-1', 'H1'}, {'H1', 'H2'}, 2, 0, 0),
             ({'-1', 'H1'}, {'-1'}, 1, 0, 1),
             ({'-1', 'H1'}, {'H2'}, 1, 0, 1),
             ({'H1', 'H2'}, {'H2', 'H3'}, 1, 1, 1)]
    for truth, pred, tp, fp, fn in cases:
        remaining, extra = unmatched(truth, pred)
        assert evaluator.mention_counts({s:truth}, {s:pred}) == (tp,fp,fn)
        assert len(remaining) == fn and len(extra) == fp
    assert not nil_support({'H1'}, {'-1','H1'})
    assert nil_support({'H1','H2'}, {'-1','H1'})
    assert nil_support({'-1'}, {'-1','H1'})
    assert unmatched({'HP:OLD'}, {'HP:CURRENT'}) == ({'HP:OLD'}, {'HP:CURRENT'})
    assert evaluator.document_concepts({'entities':[{'identifier':'-1','text':'Finding'}]}, {}, 'all','text') == {('unlinked_text','Finding')}
    assert evaluator.document_concepts({'entities':[{'identifier':'HP:1','text':'Finding'}]}, {}, 'all','text') == {('hpo','HP:1')}
    return {'one_to_one_cases': len(cases), 'nil_supply_cases': 3,
            'raw_id_no_canonicalization': True, 'document_unlinked_literal_text': True}


def analyze_document(gold, pred, candidates, evaluator, cut):
    decoded, selected, blockers, rank = greedy_trace(candidates, cut)
    assert decoded == pred['entities'], 'trace must reproduce frozen .3 entities exactly'
    gu = evaluator.positive_units(gold,gold,'all')
    pu = evaluator.positive_units(pred,gold,'all')
    counts = {'mention': dict(zip(('tp','fp','fn'),evaluator.mention_counts(gu,pu))),
              'document': dict(zip(('tp','fp','fn'),evaluator.compare_sets(
                  evaluator.document_concepts(gold,gold,'all','text'),
                  evaluator.document_concepts(pred,gold,'all','text'))))}
    by_span, concepts = defaultdict(list), defaultdict(list)
    ids = [atom_ids(c) for c in candidates]
    for i,c in enumerate(candidates):
        by_span[span(c)].append(i)
        for key in evaluator.document_concepts({'entities':[c]},gold,'all','text'):
            concepts[key].append(i)
    metrics = defaultdict(Counter)
    metrics['denominators'].update(documents=1, candidate_records=len(candidates),
        candidate_spans=len(by_span), selected_entity_records=len(pred['entities']),
        gold_mention_units=sum(map(len,gu.values())), gold_document_concepts=counts['document']['tp']+counts['document']['fn'])
    metrics['denominators']['mixed_linked_nil_gold_spans'] = sum('-1' in v and len(v)>1 for v in gu.values())
    for s, gs in gu.items():
        missed, _ = unmatched(gs, pu.get(s,set()))
        for identifier in gs:
            kind = 'unlinked' if identifier == '-1' else 'linked'
            metrics['gold_mention_units_by_kind'][kind] += 1
            if identifier not in missed:
                metrics['mention_tp_by_kind'][kind] += 1
                continue
            exact = by_span.get(s,[])
            support = [i for i in exact if (nil_support(ids[i],gs) if identifier == '-1' else identifier in ids[i])]
            if not support:
                if exact:
                    mechanism = ('pool_missing_nil_residual_supply_same_span_linked_consumption'
                                 if identifier == '-1' else 'pool_missing_correct_id_same_span_wrong_id')
                elif any(overlap(s,t) for t in by_span):
                    mechanism = 'pool_missing_exact_span_overlap_only'
                else:
                    mechanism = 'pool_missing_any_overlapping_span'
                metrics['mention_fn_top_level']['pool_support_absent'] += 1
            else:
                assert not (set(support) & selected)
                eligible = [i for i in support if candidates[i]['selector_probability'] >= cut]
                if not eligible:
                    mechanism = 'pool_support_present_all_below_cut'
                    metrics['mention_fn_top_level']['below_cut'] += 1
                else:
                    assert all(i in blockers for i in eligible)
                    mechanism = 'pool_support_at_cut_rejected_by_greedy'
                    metrics['mention_fn_top_level']['greedy_conflict'] += 1
                    i = min(eligible,key=rank.get)
                    geometry = 'same_span_competing_id' if span(candidates[blockers[i]]) == s else 'overlap_different_boundary'
                    metrics['greedy_blocker_geometry'][geometry] += 1
            metrics['mention_fn_mechanisms'][mechanism] += 1
            metrics['mention_fn_'+kind][mechanism] += 1
    missing_concepts = evaluator.document_concepts(gold,gold,'all','text') - evaluator.document_concepts(pred,gold,'all','text')
    for key in missing_concepts:
        support = concepts.get(key,[])
        if not support:
            mechanism = 'candidate_concept_absent'
        else:
            assert not (set(support)&selected)
            eligible = [i for i in support if candidates[i]['selector_probability'] >= cut]
            mechanism = ('candidate_concept_present_at_cut_greedy' if eligible else 'candidate_concept_present_all_below_cut')
            assert all(i in blockers for i in eligible)
        metrics['document_fn_mechanisms'][mechanism] += 1
        metrics['document_fn_'+('unlinked' if key[0]=='unlinked_text' else 'linked')][mechanism] += 1
    assert sum(metrics['mention_fn_mechanisms'].values()) == counts['mention']['fn']
    assert sum(metrics['mention_tp_by_kind'].values()) == counts['mention']['tp']
    assert sum(metrics['document_fn_mechanisms'].values()) == counts['document']['fn']
    return counts,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
    started=time.monotonic();signal.alarm(115)
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());out=Path(plan['output_directory'])
    assert not (out/'summary.json').exists() and not (out/'document-counts.json').exists()
    assert plan['cut']==.3 and plan['cpu_hardcap_seconds']==120
    for ref in plan['references'].values():
        path=Path(ref['path']);assert sha(path)==ref['sha256'] and path.stat().st_size==ref['bytes'], str(path)
    assert sha(__file__)==plan['source_sha256']
    evaluation=Path(plan['references']['evaluator']['path'])
    sys.path.insert(0,str(evaluation.parent))
    spec=importlib.util.spec_from_file_location('frozen_evaluator',evaluation)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    fixture_result=fixtures(evaluator)
    frozen=json.loads(Path(plan['references']['calibration_results']['path']).read_text())
    assert frozen['selected_cut']==plan['cut']
    expected=next(c for c in frozen['curves'] if c['cut']==plan['cut'])
    golds={d['pmc_id']:d for d in read_jsonl(plan['references']['train64']['path'])};assert len(golds)==64
    pooled_counts={k:Counter() for k in ('mention','document')};pooled_metrics=defaultdict(Counter)
    folds=[];doc_counts=[];seen=set()
    for fold in plan['folds']:
        scores=read_jsonl(plan['references'][fold['scores_key']]['path'])
        preds=read_jsonl(plan['references'][fold['prediction_key']]['path'])
        assert len(scores)==len(preds)==16
        assert [d['pmc_id'] for d in scores]==[d['pmc_id'] for d in preds]
        fc={k:Counter() for k in pooled_counts};fm=defaultdict(Counter)
        for scored,pred in zip(scores,preds):
            did=pred['pmc_id'];assert did not in seen;seen.add(did)
            counts,metrics=analyze_document(golds[did],pred,scored['candidates'],evaluator,plan['cut'])
            for name,values in counts.items():fc[name].update(values)
            for name,values in metrics.items():fm[name].update(values)
            doc_counts.append({'pmc_id':did,'fold':fold['fold'],'counts':counts,'aggregates':metrics})
        prior=next(x for x in expected['folds'] if x['fold']==fold['fold'])
        for name in fc:
            assert dict(fc[name])=={k:prior['counts'][name][k] for k in ('tp','fp','fn')}
            pooled_counts[name].update(fc[name])
        for name,values in fm.items():pooled_metrics[name].update(values)
        assert fm['denominators']['selected_entity_records']==prior['entity_records']
        folds.append({'fold':fold['fold'],'counts':fc,'aggregates':fm})
    assert seen==set(golds)
    for name in pooled_counts:
        assert dict(pooled_counts[name])=={k:expected['pooled']['counts'][name][k] for k in ('tp','fp','fn')}
    assert pooled_metrics['denominators']['selected_entity_records']==expected['pooled']['entity_records']
    documents=out/'document-counts.json';documents.write_text(json.dumps(doc_counts,indent=2)+'\n')
    summary={'status':'completed aggregate-only diagnostic','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(__file__),'plan_sha256':sha(plan_path),'cut':plan['cut'],
        'counts':pooled_counts,'aggregates':pooled_metrics,'folds':folds,
        'seconds':time.monotonic()-started,'fixtures':fixture_result,
        'frozen_predictions_entity_exact':True,'frozen_selected_curve_counts_exact':True,
        'document_counts':{'path':str(documents),'sha256':sha(documents),'bytes':documents.stat().st_size},
        'originaldev_read':False,'new_fit':False,'new_threshold_selection':False,'new_predictions_written':False,
        'limitations':plan['limitations']}
    assert summary['seconds']<120
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'counts':pooled_counts,'aggregates':pooled_metrics,'seconds':summary['seconds']}))


if __name__=='__main__':main()
