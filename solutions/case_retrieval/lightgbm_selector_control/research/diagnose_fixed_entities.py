"""Gold-aware development diagnostics on frozen outputs; no fit/predict/evaluate."""
from pathlib import Path
import collections, hashlib, json, time

L=Path(__file__).resolve().parent
F=L/'full64-dev-control'
paths={'prediction':F/'lightgbm.a179-entities-only.jsonl','scores':F/'selector-scores.jsonl',
       'gold':Path('/workspace/patientphex-data/day1/dev16.gold.jsonl'),'metrics':F/'metrics.json'}
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return {x['pmc_id']:x for x in map(json.loads,p.read_text().splitlines())}
def units(row):
    out=collections.defaultdict(set)
    for x in row['entities']:
        if str(x.get('note') or '').upper()!='NO':out[x['offset'],x['length']].update(x['identifier'].split(';'))
    return out
def overlaps(a,b):return a[0]<b[0]+b[1] and b[0]<a[0]+a[1]
def split_match(truth,guess):
    common=truth&guess;t=truth-common;p=guess-common
    matched=len(common)
    if '-1' in t and p:t=t-{'-1'};p=p-{sorted(p)[0]};matched+=1
    return matched,t,p
started=time.perf_counter()
gold=load(paths['gold']);pred=load(paths['prediction']);scores=load(paths['scores'])
assert set(gold)==set(pred)==set(scores) and len(gold)==16
miss=collections.Counter();wrong=collections.Counter();coverage=collections.Counter();sources=collections.Counter();docs=[]
total_tp=total_fn=total_fp=0
for pmc,g in gold.items():
    truth=units(g);guess=units(pred[pmc]);pool=collections.defaultdict(dict)
    for c in scores[pmc]['candidates']:
        span=(c['offset'],c['length'])
        for concept in c['identifier'].split(';'):
            old=pool[span].get(concept)
            if old is None or c['selector_probability']>old['selector_probability']:pool[span][concept]=c
    dm=collections.Counter();dw=collections.Counter();ctp=0
    for span in set(truth)|set(guess):
        matched,missing,extra=split_match(set(truth.get(span,set())),set(guess.get(span,set())))
        total_tp+=matched;total_fn+=len(missing);total_fp+=len(extra)
        for concept in missing:
            choices=pool.get(span,{})
            applicable=list(choices.values()) if concept=='-1' else ([choices[concept]] if concept in choices else [])
            if applicable:
                c=max(applicable,key=lambda c:c['selector_probability'])
                kind='correct_candidate_below_fixed_cut' if c['selector_probability']<.3 else 'correct_candidate_at_or_above_cut_not_selected'
                for flag in ['is_base','crf','semantic_source','public_present','bioformer_present','biobert_present']:
                    if c.get(flag):sources[flag]+=1
            elif choices:kind='exact_span_present_wrong_concepts_only'
            elif any(overlaps(span,s) for s in pool):kind='no_exact_span_but_overlapping_candidates'
            else:kind='no_overlapping_candidate'
            dm[kind]+=1;miss[kind]+=1
        for concept in extra:
            if truth.get(span):kind='gold_positive_exact_span_wrong_or_extra_concept'
            elif any(overlaps(span,s) for s in truth):kind='overlaps_positive_gold_different_span'
            else:kind='no_overlapping_positive_gold'
            dw[kind]+=1;wrong[kind]+=1
    # Optimistic coverage, ignoring probabilities/compatibility/unary checks.
    for span, concepts in truth.items():
        m,t,_=split_match(set(concepts),set(pool.get(span,{})))
        coverage['gold_units']+=len(concepts);coverage['available_candidate_units']+=m
        coverage['absent_candidate_units']+=len(t)
        ctp+=m
    docs.append(dict(pmc_id=pmc,missing=dict(dm),extra=dict(dw),candidate_available_units=ctp))
expected=json.loads(paths['metrics'].read_text())['counts']['mention']
assert (total_tp,total_fp,total_fn)==(expected['tp'],expected['fp'],expected['fn'])
result=dict(status='fixed_original_dev_gold_aware_error_inventory_no_model_or_new_score',
 sources={k:ref(p) for k,p in paths.items()},source=ref(Path(__file__)),
 exact_cached_counts={'tp':total_tp,'fp':total_fp,'fn':total_fn},
 missing_partition=dict(miss),extra_partition=dict(wrong),candidate_optimistic_coverage=dict(coverage),
 missed_correct_candidate_source_memberships=dict(sources),per_document=docs,
 limits=['Candidate coverage is optimistic and not realizable Score; ignores incompatible and negated candidates.',
         'Gold used only for explicitly authorized development error diagnosis after frozen result; no parameter/output changes.',
         'Source memberships overlap and are not exclusive error causes; boundary and concept categories are descriptive.'],
 seconds=time.perf_counter()-started)
out=L/'fixed-entity-error-inventory.json'
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['exact_cached_counts','missing_partition','extra_partition','candidate_optimistic_coverage','seconds']}))
