"""One frozen input-space audit, followed only by literal relation set accounting."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from typing import Any


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size, sha256=sha(p))
def check(r):
    p=Path(r['path']); assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],str(p); return p

def read(r): return json.loads(check(r).read_text())
def rows(r):
    p=check(r); op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt') as f: return [json.loads(x) for x in f if x.strip()]
def write(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def extract(r,names):
    p=check(r); tree=ast.parse(p.read_text()); nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names: nodes.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets): nodes.append(n)
    env={'Path':Path,'re':re,'Any':Any}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
    assert all(n in env for n in names)
    return env

def key(e): return (str(e['identifier']),int(e['offset']),int(e['length']),e['text'])
def count_categories(records): return dict(sorted(Counter(x['category'] for x in records).items()))

def main(plan_path):
    started=time.monotonic();signal.alarm(120);plan=json.loads(Path(plan_path).read_text())
    assert sha(__file__)==plan['source']['sha256']
    out=Path(plan['output_directory']);out.mkdir(exist_ok=False)
    write(out/'start.json',dict(at_utc=now(),plan=ref(plan_path)))
    inv=read(plan['inventory']);baseline=read(plan['baseline']);read(plan['definitions']);read(plan['counter_contract'])
    old=extract(inv['common']['old_association_keys_and_HPO_parser'],{'parse_hpo','association_keys'})
    terms,allowed,aliases,version=old['parse_hpo'](check(inv['common']['HPO']));allowed.discard('HP:0000118')
    def canon(x):
        if not isinstance(x,str) or re.fullmatch(r'HP:[0-9]{7}',x) is None:return None
        v=aliases.get(x,x);return v if v in allowed else None
    compound=re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
    def atoms(e):
        v=str(e['identifier'])
        if v=='-1':return [('text',e['text'])]
        return [('hpo' if re.fullmatch(r'HP:[0-9]{7}',a) else 'other',a) for a in (v.split(';') if compound.fullmatch(v) else [v])]
    native={};native_scores={}
    for model in ['pubmed','bioformer','biobert']:
        data=rows(inv['common'][model+'_canonical']);s=defaultdict(list)
        for x in data:
            c=canon(x['hpo_id']);assert c==x['hpo_id'];s[(str(x['pmc_id']),c)].append(float(x['score']))
        native[model]=set(s);native_scores[model]=s
    full=[];aggregate=[];runtime=[]
    for f in inv['folds']:
        fold=f['fold'];inputs=rows(f['input']);ids=[str(x['pmc_id']) for x in inputs];idset=set(ids)
        patients={str(x['pmc_id']):{str(y['patient_id']) for y in x['patient']} for x in inputs}
        selected=rows(f['Task1_selected_entities_file']);pools=rows(f['Task1_input_pool']);old_docs=rows(f['old_entities'])
        assert [str(x['pmc_id']) for x in selected]==ids==[str(x['pmc_id']) for x in pools]==[str(x['pmc_id']) for x in old_docs]
        old_map={};O=set();old_hpo=set();old_exceptions=Counter()
        for doc in old_docs:
            d=str(doc['pmc_id']); m=defaultdict(set)
            for e in doc['entities']:
                kind='text' if str(e['identifier'])=='-1' else 'hpo'
                for val in old['association_keys'](e):
                    m[val].add((kind,val));O.add((d,kind,val))
                    c=canon(val) if kind=='hpo' else None
                    if c:old_hpo.add((d,c))
                    else:old_exceptions[kind]+=1
            old_map[d]=m
        old_pairs=set();pub_pairs=set();ambiguous=[];promptcounts={}
        for label,r in [('old',f['old_prompts']),('public2',f['prompts'])]:
            prs=rows(r);promptcounts[label]=len(prs)
            for p in prs:
                d=str(p['pmc_id']);assert d in idset and set(map(str,p['patient_ids']))<=patients[d]
                for val in p['code_map'].values():
                    if label=='old':
                        mapped=old_map[d].get(val,set());assert mapped
                        if len(mapped)!=1:ambiguous.append((d,val));continue
                        kind,v=next(iter(mapped))
                        old_pairs.update((d,str(pid),kind,v) for pid in p['patient_ids'])
                    else:
                        assert canon(val)==val
                        pub_pairs.update((d,str(pid),'hpo',val) for pid in p['patient_ids'])
        assert not ambiguous,'ambiguous typed raw code_map'
        old_doc={(d,k,v) for d,p,k,v in old_pairs};assert old_doc==O,(fold,'old prompt!=old entityspace',len(old_doc-O),len(O-old_doc))
        legal_raw=old_pairs|pub_pairs
        legal_canon={(d,p,canon(v)) for d,p,k,v in legal_raw if k=='hpo' and canon(v)}
        legal_doc={(d,c) for d,p,c in legal_canon};pub_doc={(d,v) for d,p,k,v in pub_pairs}
        public_difference={(str(d),v) for d,k,v in read(f['public_difference_concepts']) if k=='hpo'}
        assert pub_doc==public_difference
        expected_pub={(d,c) for d,c in native['pubmed']|native['bioformer'] if d in idset}-old_hpo
        assert pub_doc==expected_pub,(fold,'public canonical difference mismatch',len(pub_doc-expected_pub),len(expected_pub-pub_doc))
        T=set();source_flags=defaultdict(set);selected_raw=defaultdict(set);exceptions=Counter();selected_mentions=0
        flags=['is_base','semantic_source','crf','crf_span','public_present','public_neural','public_dictionary','public_abbreviation','bioformer_present','bioformer_neural','bioformer_dictionary','bioformer_abbreviation','biobert_present','biobert_neural','biobert_dictionary','biobert_abbreviation']
        for doc,pool in zip(selected,pools):
            d=str(doc['pmc_id']);lookup=defaultdict(list)
            for c in pool['candidates']:lookup[key(c)].append(c)
            for e in doc['entities']:
                selected_mentions+=1;cs=lookup[key(e)];assert cs,(fold,'selected exact key missing')
                for kind,val in atoms(e):
                    c=canon(val) if kind=='hpo' else None
                    if not c:exceptions[kind]+=1;continue
                    t=(d,c);T.add(t);selected_raw[t].add(val)
                    for candidate in cs:
                        source_flags[t].update(k for k in flags if bool(candidate.get(k,False)))
        missing=T-legal_doc
        records=[]
        for d,c in sorted(missing):
            mask=[m for m in native if (d,c) in native[m]]
            category='public2_present_but_not_prompted' if ('pubmed'in mask or 'bioformer'in mask) else ('BioBERT_only_native' if 'biobert'in mask else 'nonpublic_or_other')
            observations={m:dict(max_score=max(native_scores[m][(d,c)]),any_above_native_neural_threshold=any(s>.95 for s in native_scores[m][(d,c)]),all_hybrid_at_or_below_095=all(s<=.95 for s in native_scores[m][(d,c)])) for m in mask}
            records.append(dict(pmc_id=d,hpo=c,category=category,public_supply_mask=mask,selected_exact_source_flags=sorted(source_flags[(d,c)]),selected_raw_identifiers=sorted(selected_raw[(d,c)]),native_hybrid_observations=observations,patient_ids=sorted(patients[d])))
        sourcecounts={k:sum(k in x['selected_exact_source_flags'] for x in records) for k in flags}
        nativeobservations={m:dict(present=sum(m in x['native_hybrid_observations'] for x in records),all_hybrid_at_or_below_095=sum(x['native_hybrid_observations'].get(m,{}).get('all_hybrid_at_or_below_095',False) for x in records)) for m in native}
        patient_gaps={(d,p,c) for d,c in T for p in patients[d]}-legal_canon
        doc_missing_pairs={(x['pmc_id'],p,x['hpo']) for x in records for p in x['patient_ids']}
        a=dict(fold=fold,documents=len(ids),patients=sum(map(len,patients.values())),selected_entity_records=selected_mentions,selected_canonical_document_hpo=len(T),selected_noncanonical_counts=dict(exceptions),old_typed_document_concepts=len(O),old_canonical_document_hpo=len(old_hpo),public2_new_document_hpo=len(pub_doc),legal_canonical_document_hpo=len(legal_doc),legal_canonical_patient_hpo=len(legal_canon),missing_selected_document_hpo=len(missing),missing_selected_patient_hpo=len(doc_missing_pairs),additional_patient_specific_gap=len(patient_gaps-doc_missing_pairs),categories=count_categories(records),selected_exact_source_flags_nonexclusive=sourcecounts,native_hybrid_observations=nativeobservations,prompt_rows=promptcounts,old_entity_noncanonical_observations=dict(old_exceptions),old_entity_prompt_equivalence=True,public2_native_difference_prompt_equivalence=True)
        aggregate.append(a);full.append(dict(fold=fold,missing_concepts=records,canonical_legal_pairs=sorted(legal_canon),raw_legal_pairs=sorted(legal_raw),all_selected_document_hpo=sorted(T)))
        runtime.append((f,ids,patients,records,legal_raw,legal_canon,T,doc_missing_pairs,patient_gaps))
    write(out/'input-only-spaces.json',full)
    write(out/'input-only-freeze.json',dict(at_utc=now(),plan=ref(plan_path),spaces=ref(out/'input-only-spaces.json'),folds=aggregate,gold_read=False,definitions=plan['definitions']))
    # Only after all input-only keys and categories are durably frozen do we open gold.
    evaluation=extract(inv['common']['evaluator_v2_normalization_only'],{'COMPOUND_HPO','associations'})
    associations=evaluation['associations'];metrics=read(baseline['existing_metrics_ref']);quality=[]
    write(out/'gold-accounting-start.json',dict(at_utc=now(),input_only_freeze=ref(out/'input-only-freeze.json'),new_evaluator_calls=0))
    for f,ids,patients,records,legal_raw,legal_canon,T,missing_pairs,patient_gaps in runtime:
        fold=f['fold'];gp=Path(f['gold_unread']['path']);assert sha(gp)==f['gold_unread']['historical_sha256_from_existing_plan']
        gold=[json.loads(x) for x in gp.read_text().splitlines() if x.strip()]
        pred=rows(baseline['folds'][fold]['current_relation_prediction'])
        assert [str(x['pmc_id']) for x in gold]==ids==[str(x['pmc_id']) for x in pred]
        for g in gold:assert {str(x['patient_id']) for x in g['patient']}==patients[str(g['pmc_id'])]
        def relations(docs):
            return {(str(x['pmc_id']),str(pid),val) for x in docs for pid,vals in associations(x).items() for val in vals}
        G=relations(gold);P=relations(pred);FN=G-P
        observed=dict(tp=len(G&P),fp=len(P-G),fn=len(FN));expected=metrics['folds'][fold]['counts_after']['relation']
        assert observed=={k:expected[k] for k in observed},(fold,observed,expected)
        linkedG={x for x in G if re.fullmatch(r'HP:[0-9]{7}',x[2])};linkedFN=linkedG-P
        rawlegal={(d,p,v) for d,p,k,v in legal_raw}
        cg={(d,p,canon(c)) for d,p,c in linkedG if canon(c)};cp={(d,p,canon(c)) for d,p,c in P if canon(c)}
        def opportunity(pairs):
            hits=pairs&linkedG;fnhits=pairs&linkedFN
            return dict(candidate_patient_hpo=len(pairs),literal_gold_pairs=len(hits),literal_reference_FN_pairs=len(fnhits),literal_reference_FN_document_hpo=len({(d,c) for d,p,c in fnhits}),literal_reference_FN_documents=len({d for d,p,c in fnhits}),canonical_gold_pairs=len(pairs&cg),canonical_reference_FN_pairs=len(pairs&(cg-cp)))
        cats={}
        for category in sorted({x['category'] for x in records}):
            pairs={(x['pmc_id'],p,x['hpo']) for x in records if x['category']==category for p in x['patient_ids']};cats[category]=opportunity(pairs)
        quality.append(dict(fold=fold,baseline_relation_counts_crosscheck=observed,reused_baseline_score=metrics['folds'][fold]['new_metrics']['Score'],gold_all_pairs=len(G),gold_literal_linked_hpo_pairs=len(linkedG),reference_FN_literal_linked_hpo_pairs=len(linkedFN),literal_gold_pairs_outside_actual_prompt_space=len(G-rawlegal),literal_linked_gold_pairs_outside_actual_prompt_space=len(linkedG-rawlegal),literal_reference_FN_outside_actual_prompt_space=len(FN-rawlegal),literal_linked_reference_FN_outside_actual_prompt_space=len(linkedFN-rawlegal),selected_doc_missing=opportunity(missing_pairs),all_selected_patient_gap=opportunity(patient_gaps),by_input_only_category=cats,canonical_gold_linked_pairs=len(cg),canonical_reference_FN_linked_pairs=len(cg-cp),gold=ref(gp)))
    result=dict(status='COMPLETE_AGGREGATE_DIAGNOSTIC_ONLY',at_utc=now(),source=ref(__file__),plan=ref(plan_path),input_only_freeze=ref(out/'input-only-freeze.json'),input_counts=aggregate,relationship_counts=quality,seconds=time.monotonic()-started,new_v2_calls=0,new_model_calls=0,new_fit_calls=0,original_dev_gold_read=False,official_B_unchanged=True,limitations=plan['limitations'])
    write(out/'result.json',result);signal.alarm(0)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],folds=[dict(fold=a['fold'],missing=a['missing_selected_document_hpo'],categories=a['categories']) for a in aggregate],coverage=[dict(fold=q['fold'],selected=q['selected_doc_missing']) for q in quality])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)
