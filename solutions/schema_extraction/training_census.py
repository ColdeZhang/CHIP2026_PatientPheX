"""Train32-only structural feasibility; no invented mention-pair labels or fitting."""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from prepare import build

def identifiers(value,text=None):
    if value=='-1':return [text]
    return value.split(';') if re.fullmatch(r'HP:\d{7}(;HP:\d{7})*',value) else [value]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--upstream',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    p=Path(args.train);assert hashlib.sha256(p.read_bytes()).hexdigest()=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    source=Path(args.upstream)/'gliner2/processing/word_splitter.py';spec=importlib.util.spec_from_file_location('frozen_schema_splitter',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);splitter=module.WhitespaceTokenSplitter()
    docs=[json.loads(x) for x in p.read_text().splitlines() if x];assert len(docs)==32;counts=Counter();rows=[];association_fields=set()
    for raw in docs:
        doc=build({k:v for k,v in raw.items() if k not in ['entities','association']});tokens=list(splitter(doc['text'],lower=False));starts={a:i for i,(t,a,b) in enumerate(tokens)};ends={b:i+1 for i,(t,a,b) in enumerate(tokens)};windows=[];begin=0
        while begin<len(tokens):
            end=min(begin+384,len(tokens));windows.append((tokens[begin][1],tokens[end-1][2]))
            if end==len(tokens):break
            begin+=320
        counts['documents']+=1;counts['patients']+=len(doc['patients']);counts['windows']+=len(windows)
        byconcept=defaultdict(set);positive=[];bad=[]
        for i,entity in enumerate(raw['entities']):
            counts['entity_annotations']+=1
            if str(entity.get('note') or '').upper()=='NO':counts['explicit_NO_entity_annotations']+=1;continue
            counts['positive_entity_annotations']+=1
            mapped=[]
            for s in doc['passages']:
                a=int(entity['offset'])-s['offset']+s['start'];b=a+int(entity['length'])
                if s['start']<=a<b<=s['end'] and doc['text'][a:b]==entity['text'] and len(entity['text'])==entity['length']:mapped.append((a,b))
            if not mapped:counts['invalid_positive_source_spans']+=1;bad.append(i);continue
            a,b=mapped[0];positive.append((a,b));aligned=a in starts and b in ends;counts['native_word_aligned_positive_annotations' if aligned else 'positive_annotations_inside_native_words']+=1
            covered=any(s<=a<b<=e for s,e in windows);counts['window_covered_positive_annotations' if covered else 'positive_annotations_cross_all_windows']+=1
            if aligned and covered:
                for value in identifiers(entity['identifier'],entity['text']):byconcept[value].add((a,b))
        patient_spans={q['patient_id']:{tuple(span) for mention in q['mentions'] for span in mention['spans'] if span[0] in starts and span[1] in ends} for q in doc['patients']}
        gold=set()
        for association in raw['association']:
            association_fields.update(association)
            for value in association['phenotype']:
                gold.update((str(association['patient_id']),x) for x in identifiers(value,value))
        bag_sizes=[]
        for patient,concept in sorted(gold):
            counts['document_positive_relation_pairs']+=1
            if concept not in byconcept:counts['positive_relation_without_aligned_observed_concept_span']+=1;bag_sizes.append(0);continue
            counts['positive_relation_with_aligned_observed_concept_span']+=1
            candidates={(a,b,c,d) for a,b in patient_spans[patient] for c,d in byconcept[concept]
                        if any(s<=min(a,c) and max(b,d)<=e for s,e in windows)}
            bag_sizes.append(len(candidates))
            if candidates:counts['positive_relation_with_same_window_candidate_bag']+=1
            else:counts['positive_relation_without_same_window_candidate_bag']+=1
            if len(candidates)>1:counts['positive_relation_with_ambiguous_multi_pair_bag']+=1
            counts['structural_candidate_pairs_not_exact_relation_gold']+=len(candidates)
        window_counts=[len({(a,b) for a,b in positive if s<=a<b<=e}) for s,e in windows]
        counts['windows_over_64_positive_span_targets']+=sum(n>64 for n in window_counts)
        rows.append(dict(pmc_id=doc['pmc_id'],windows=len(windows),max_positive_span_targets=max(window_counts,default=0),positive_relation_pairs=len(gold),max_structural_bag_pairs=max(bag_sizes,default=0),invalid_positive_entity_indices=bad))
    report=dict(kind='TRAIN32_STRUCTURAL_FEASIBILITY_ONLY_NO_NEW_QUALITY_NO_FIT',train_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),counts=dict(counts),association_fields=sorted(association_fields),exact_head_tail_relation_annotation_fields_present=False,
                caveat='Co-window candidate bags only show structural availability. Their pairs are not labeled relation evidence; no pair-wise training dataset is produced.',
                native_splitter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
