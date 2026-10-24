"""Authorized aggregate posthoc F1 diagnosis; no evaluator, new scores or output predictions."""
from pathlib import Path
from collections import defaultdict
import argparse,json,gzip,hashlib,datetime,sys,re
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from evaluate import associations  # frozen literal compound expansion ONLY; never evaluate
R=Path('/workspace/patientphex-runs/round2-20261023');D0=R/'research/patient-conditioned-spans-v1';G=R/'research/patient-span-geometry-v1'

def ref(p):
    p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def pairs(rows):return {(d['pmc_id'],str(pid),v) for d in rows for pid,vs in associations(d).items() for v in vs}
def concepts(rows):
    out=defaultdict(set);spans=defaultdict(lambda:defaultdict(set));linked=defaultdict(lambda:defaultdict(set));nil=defaultdict(set)
    for d in rows:
        for e in d['entities']:
            if str(e.get('note') or '').upper()=='NO':continue
            span=(e['offset'],e['length'])
            for ident in e['identifier'].split(';'):
                value=e['text'] if ident=='-1' else ident
                out[d['pmc_id']].add(value);spans[d['pmc_id']][span].add(value)
                if ident=='-1':nil[d['pmc_id']].add(span)
                else:linked[d['pmc_id']][span].add(ident)
    return out,spans,linked,nil

def main(output):
    O=Path(output);O.mkdir(parents=True,exist_ok=False);plan=json.loads((G/'collected/source/plan.json').read_text())
    paths={'gold':Path(plan['quality_refs']['gold']['path']),'B471':Path(plan['baseline']['path']), 'strong_task1':R/'research/lightgbm-selector-readiness/formal-control/fold1/lightgbm.nested.jsonl','B507':R/'research/parallel-relation-fusion-20261024/actual-v1/two_of_three.prediction.jsonl','B524':R/'research/four-source-fusion-20261024/F1-v1/prediction.jsonl','evaluator_source':Path('/workspace/CHIP2026_PatientPheX/scripts/evaluate.py')}
    assert ref(paths['strong_task1'])['sha256']=='8cb61ba94fab8de4223acafeece83ef5a243954312d008073a42e5ba80398d6f'
    assert ref(paths['B507'])['sha256']=='4967e8a65cab39e74d78d03ba6811e7b7dd19d561c3f87e25289b2dc603b2ce1'
    assert ref(paths['gold'])==plan['quality_refs']['gold']
    routes={'old8':(D0/'assembled-v2/primary-B-union-new.pred.jsonl',D0/'quality-v1/result.json',D0/'collected/formal-v1/normalization/selected-spans.json.gz'), 'old24':(D0/'duration24-v1/assembled-v1/primary-B-union-new.pred.jsonl',D0/'duration24-v1/quality-v1/result.json',D0/'duration24-v1/collected/run/normalization/selected-spans.json.gz'), 'geometry24':(G/'assembled-v1/primary-B-union-new.pred.jsonl',G/'quality-v1/result.json',G/'collected/run/normalization/selected-spans.json.gz')}
    intent={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'root-authorized aggregate diagnosis of three already evaluated F1 controls; known development gold, no new prediction/Score/fit/GPU','references':{k:ref(p) for k,p in paths.items()},'route_refs':{k:{n:ref(p) for n,p in zip(['prediction','existing_quality','selected_span_evidence'],vs)} for k,vs in routes.items()},'source':ref(__file__)}
    (O/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
    base=pairs(read(paths['B471']));b507=pairs(read(paths['B507']));b524=pairs(read(paths['B524']));assert len(base)==471 and len(b507)==507 and len(b524)==524 and base<=b507<=b524
    strong,_,_,_=concepts(read(paths['strong_task1']));additions={};support={}
    for name,(pred,qpath,selected) in routes.items():
        predpairs=pairs(read(pred));assert base<=predpairs;new=predpairs-base;additions[name]=new
        # No gold used to define support or intersections.
        supported={x for x in new if x[2] in strong[x[0]]}
        support[name]={'new_relations':len(new),'strong_task1_document_concept_supported':len(supported),'strong_task1_document_concept_unsupported':len(new-supported),'overlap_B507':len(new&b507),'overlap_B524':len(new&b524),'overlap_B524_minus_B507':len(new&(b524-b507))}
    (O/'support-no-gold.json').write_text(json.dumps({'no_gold_used':True,'counts':support,'concept_rule':'NO excluded; compound entity identifiers split; NIL uses exact entity text; document-level only, not patient evidence'},indent=2)+'\n')
    goldrows=read(paths['gold']);gold=pairs(goldrows);gold_doc=defaultdict(set)
    for pmc,pid,v in gold:gold_doc[pmc].add(v)
    entity_doc,entity_spans,linked_spans,nil_spans=concepts(goldrows);result={}
    for name,(pred,qpath,selected) in routes.items():
        new=additions[name];tp=new&gold;fp=new-gold;other={x for x in fp if x[2] in gold_doc[x[0]]};absent=fp-other;in_entities={x for x in absent if x[2] in entity_doc[x[0]]}
        evidence=defaultdict(set)
        for s in json.loads(gzip.decompress(selected.read_bytes())):
            for pid in s['selected_patient_ids']:
                for c in s['identifiers']:evidence[(s['pmc_id'],str(pid),c)].add((s['offset'],s['length']))
        assert new<=set(evidence)
        exact=set();mismatch=set();linked_wrong=set();nil_only=set();matching_alt=set();all_exact_wrong=set();multiple=set()
        for key in fp:
            pmc,pid,c=key;es={span for span in evidence[key] if span in entity_spans[pmc]}
            if len(evidence[key])>1:multiple.add(key)
            if not es:continue
            exact.add(key)
            wrong={s for s in es if c not in entity_spans[pmc][s]}
            if wrong:mismatch.add(key)
            if es-wrong:matching_alt.add(key)
            if wrong==es:all_exact_wrong.add(key)
            if any(linked_spans[pmc].get(s) and c not in linked_spans[pmc][s] and re.fullmatch(r'HP:\d{7}',c) for s in es):linked_wrong.add(key)
            if any(s in nil_spans[pmc] and not linked_spans[pmc].get(s) for s in es):nil_only.add(key)
        q=json.loads(qpath.read_text())['endpoints']['primary-B-union-new'];assert (len(new),len(tp),len(fp))==(q['added'],q['added_TP'],q['added_FP'])
        result[name]={'new':len(new),'TP':len(tp),'FP':len(fp),'FP_other_gold_patient_same_literal_concept':len(other),'FP_literal_concept_absent_all_gold_patient_relations':len(absent),'absent_relation_but_present_gold_entity_concept':len(in_entities),'absent_relation_and_absent_gold_entity_concept':len(absent-in_entities),'span_evidence':{'FP_with_any_exact_positive_gold_span':len(exact),'FP_with_any_exact_span_literal_concept_mismatch':len(mismatch),'FP_with_exact_linked_gold_span_wrong_HPO':len(linked_wrong),'FP_with_any_exact_NIL_only_gold_span':len(nil_only),'FP_with_any_exact_matching_concept_alternative':len(matching_alt),'FP_all_exact_supporting_spans_literal_mismatch':len(all_exact_wrong),'FP_with_multiple_supporting_spans':len(multiple),'missing_source_mapping':0},'non_gold_support':support[name]}
    out={'status':'COMPLETE_AGGREGATE_EXISTING_F1_DIAGNOSIS_NO_NEW_V2','counts':result,'new_quality_calls':0,'new_predictions':0,'new_model_calls':0,'known_F1_development_gold':True,'limitations':['relation TP uses frozen evaluator literal compound expansion; no HPO hierarchy or text normalization','concept absent from gold relationships does not prove unmentioned, nonphenotypic or wrongly normalized; entity concept detail only annotation coverage','exact gold span is document evidence, not proof of patient assignment; multiple span contributors may disagree','any-span mismatch and matching-alternative counters can overlap; all_exact_supporting_spans_mismatch excludes cases with a matching exact alternative but says nothing about unmatched spans','NIL exact-span wildcard mention credit differs from literal relation concept match; linked-HPO mismatch reported separately','support and B507/B524 overlaps use all additions without gold or patient selection; no output filtering or new Score'], 'intent':ref(O/'intent.json'),'no_gold_support':ref(O/'support-no-gold.json')}
    (O/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
