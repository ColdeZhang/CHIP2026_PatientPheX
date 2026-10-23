#!/usr/bin/env python3
"""Post-hoc error attribution on a labeled validation set; never fits a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import joblib
import run
# Model checkpoints made by the standalone runner use __main__ for this class.
Retriever=run.Retriever
Candidate=run.Candidate

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['model','input','gold','pred','output']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--mention-threshold',type=float,default=.60)
    args=ap.parse_args()
    saved=joblib.load(args.model);retriever,model,_=saved['model']
    gold={str(d['pmc_id']):d for d in run.load_jsonl(args.gold)}
    pred={str(d['pmc_id']):d for d in run.load_jsonl(args.pred)}
    counts=Counter();examples=[]
    for doc in run.load_jsonl(args.input):
        gid=str(doc['pmc_id']);gd=gold[gid];pd=pred[gid]
        candidates=retriever.candidates(doc);run.score_candidates(doc,candidates,retriever,model)
        bykey={(c.start,c.end,c.hid):c for c in candidates}
        selected={(e['offset'],e['offset']+e['length'],h) for e in pd['entities'] for h in run.ids(e)}
        goldkeys=run.gold_mentions(gd,retriever.alt)
        for e in gd['entities']:
            if str(e.get('note','')).upper()=='NO':continue
            for raw in run.ids(e):
                hid=retriever.alt.get(raw,raw);key=(e['offset'],e['offset']+e['length'],hid)
                if key in selected:continue
                c=bykey.get(key)
                if c is None:reason='missing_retrieval_candidate'
                elif run.nearby(doc,c)[3]:reason='negation_filter'
                elif c.probability<args.mention_threshold:reason='below_mention_threshold'
                else:reason='overlap_or_id_selection'
                counts['false_negative_'+reason]+=1
                if len(examples)<8:examples.append({'pmc_id':gid,'text':e['text'],'identifier':hid,'reason':reason})
        for key in selected-goldkeys:
            if any(key[:2]==g[:2] for g in goldkeys):reason='wrong_id_same_span'
            elif any(key[0]<g[1] and g[0]<key[1] for g in goldkeys):reason='overlapping_wrong_boundary'
            else:reason='no_gold_overlap'
            counts['false_positive_'+reason]+=1
        for key in selected:
            if key in bykey:
                c=bykey[key]
                counts['selected_exact' if c.exact else 'selected_fuzzy']+=1
                if key in goldkeys:counts['true_positive_exact' if c.exact else 'true_positive_fuzzy']+=1
                if not c.exact:
                    same_stem=' '.join(map(run.stem,c.normalized.split()))==' '.join(map(run.stem,c.matched.split()))
                    subtype='morphology' if same_stem else 'approximate'
                    counts['selected_'+subtype]+=1
                    if key in goldkeys:counts['true_positive_'+subtype]+=1
        present={h if h!='-1' else e['text'] for e in pd['entities'] for h in run.ids(e)}
        gs=run.gold_associations(gd,retriever.alt);ps={str(x['patient_id']):set(x['phenotype']) for x in pd['association']}
        for key,values in gs.items():
            for concept in values-ps.get(key,set()):
                counts['relation_fn_concept_present' if concept in present else 'relation_fn_concept_absent']+=1
    result={'kind':'posthoc_validation_diagnosis_not_used_for_training','counts':dict(counts),'examples':examples,
      'note':'Counts use strict split-HPO span matching; -1 matching and duplicate assumptions may differ slightly from the shared evaluator.'}
    run.write_json(args.output,result);print(json.dumps(result['counts'],ensure_ascii=False))
if __name__=='__main__':main()
