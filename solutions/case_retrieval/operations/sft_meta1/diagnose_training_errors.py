#!/usr/bin/env python3
"""CPU diagnosis of frozen F1 OOF SFT32 only; no outer0/originaldev labels."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import argparse,ast,hashlib,importlib.util,json,re,sys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def marker_ids(text):
    return {p for token in re.findall(r'\[PATIENT ([^:\]\n]+):',text) for p in token.split('/')}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--output-subdir',default='training-error-diagnostic');a=ap.parse_args();b=a.directory.resolve();out=b/a.output_subdir;out.mkdir()
    r=Path('/workspace/patientphex-runs/round2-20261023');d=Path('/workspace/patientphex-data');gold_path=d/'round2-inner4/fold-1/validation16.gold.jsonl';input_path=d/'round2-inner4/fold-1/validation16.input.jsonl';candidates=r/'sequence_crf/relation-sft-fold0/inner1/candidates.jsonl'
    assert sha(b/'run.py')=='41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0' and sha(b/'prepare_pair_rag.py')=='16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97'
    assert sha(b/'meta.responses.jsonl')==json.loads((b/'meta-responses-freeze.json').read_text())['raw_sha256']
    sys.path.insert(0,str(b));import run as base;import prepare_pair_rag as pp
    _,allowed,alt,_=base.parse_hpo(str(d/'hp.obo'));linker=SimpleNamespace(alt=alt)
    canon=lambda x:alt.get(x,x)
    inputs={str(x['pmc_id']):x for x in rows(input_path)};gold={str(x['pmc_id']):x for x in rows(gold_path)};entities={str(x['pmc_id']):x['entities'] for x in rows(candidates)}
    prompts=rows(b/'meta.prompts.jsonl');raw={x['row_id']:x for x in rows(b/'meta.responses.jsonl')};train_manifest=json.loads((b/'preparation-manifest.json').read_text());trained={str(x['pmc_id']) for x in rows(b/'train.jsonl')}
    assert set(inputs)==set(gold)==set(entities)=={str(x['pmc_id']) for x in prompts} and len(inputs)==16 and not trained&set(inputs)
    assert all('entities' not in x and 'association' not in x for x in inputs.values())
    node=next(x for x in ast.parse((b/'prepare_pair_rag.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='evidence')
    all_source=ast.get_source_segment((b/'prepare_pair_rag.py').read_text(),node).replace('if len(unique)>=4:break','if False:break')
    env=dict(pp.__dict__);exec(all_source,env);all_evidence=env['evidence']
    G=set();G_raw=set();C=set();P=set();details={};cid_decisions={};missing_details=[]
    for did,doc in gold.items():
        for pid,concepts in base.gold_associations(doc,linker).items():G.update((did,str(pid),c) for c in concepts)
        for pid,concepts in base.gold_associations(doc).items():G_raw.update((did,str(pid),c) for c in concepts)
    for prompt in prompts:
        did=str(prompt['pmc_id']);doc=inputs[did];response=raw[prompt['row_id']];parsed=json.loads(response['response'])
        assert not response['hit_token_limit'] and response['removed_demonstrations']==0 and set(parsed)==set(prompt['code_map'])
        text=prompt['messages'][-1]['content'];anchors,serialized=text.split('\n\nCandidate phenotypes:\n',1);items=json.loads(serialized);bycode={x['code']:x for x in items};grouped=pp.grouped_concepts(entities[did]);assert anchors=='Patient identities and global anchors:\n'+pp.patient_anchors(doc)
        for cid,concept in prompt['code_map'].items():
            canonical=canon(concept);kept=bycode[cid]['evidence'];expected=pp.evidence(doc,grouped[concept],500);allrows=all_evidence(doc,grouped[concept],500);assert kept==expected
            selected=parsed[cid];assert isinstance(selected,list) and len(selected)==len(set(selected)) and all(pid in prompt['patient_ids'] for pid in selected)
            cid_decisions[(did,canonical)]=set(selected)
            local=marker_ids('\n'.join(e['context'] for e in kept));all_local=marker_ids('\n'.join(e['context'] for e in allrows));global_ids=marker_ids(anchors)
            for pid in prompt['patient_ids']:
                k=(did,pid,canonical);assert k not in C;C.add(k)
                if pid in selected:P.add(k)
                details[k]=dict(target_marker_selected_local=pid in local,target_marker_any_local=pid in all_local,
                    target_marker_global=pid in global_ids,other_marker_selected_local=bool(local-{pid}),
                    lost_all_target_at_top4=pid in all_local and pid not in local,all_occurrence_windows=len(allrows),selected_windows=len(kept),selected_patients=selected,patient_count=len(prompt['patient_ids']))
    assert len(C)==1557 and len(P)<=len(C)
    missing=G-C;covered=G&C;false_negative=G-P;within=covered-P;true_positive=P&G;false_positive=P-G
    for did,pid,concept in sorted(missing):
        ges=[e for e in gold[did]['entities'] if concept in {canon(k) for k in base.association_keys(e)}];pred_es=entities[did]
        def span(e):return int(e['offset']),int(e['length'])
        matched=[(g,e) for g in ges for e in pred_es if span(g)==span(e)]
        overlap=[(g,e) for g in ges for e in pred_es if max(int(g['offset']),int(e['offset']))<min(int(g['offset'])+int(g['length']),int(e['offset'])+int(e['length']))]
        missing_details.append(dict(pmc_id=did,patient_id=pid,concept=concept,gold_entity_occurrences=len(ges),exact_span_any_candidate=bool(matched),overlap_any_candidate=bool(overlap)))
    def visibility(keys):
        c=Counter()
        for k in keys:
            x=details[k]
            for n in ['target_marker_selected_local','target_marker_any_local','target_marker_global','other_marker_selected_local','lost_all_target_at_top4']:c[n]+=int(x[n])
            c['no_target_marker_any_500char_window']+=not x['target_marker_any_local'];c['empty_CID_answer']+=not x['selected_patients'];c['nonempty_CID_without_target']+=bool(x['selected_patients']) and k not in P
            c['multiple_supplied_patients']+=x['patient_count']>1
        return dict(pairs=len(keys),**c)
    empty={k for k in within if not details[k]['selected_patients']};other=within-empty
    counts=dict(documents=16,patients=sum(len(x['patient']) for x in inputs.values()),prompts=80,candidate_CIDs=len(cid_decisions),candidate_patient_concept_keys=len(C),
        raw_gold_relations=len(G_raw),canonical_gold_relations=len(G),raw_covered_positive=len(G_raw&C),canonical_covered_positive=len(covered),
        candidate_missing_positive=len(missing),covered_recall_ceiling=len(covered)/len(G),prediction_relations=len(P),tp=len(true_positive),fp=len(false_positive),fn=len(false_negative),
        missing_candidates_part_of_FN=len(missing),within_candidates_FN=len(within),within_FN_empty_CID=len(empty),within_FN_nonempty_other_patients=len(other),
        within_FN_nonempty_all_selected_patients_are_gold=sum(all((k[0],pid,k[2]) in G for pid in details[k]['selected_patients']) for k in other),
        FP_concept_belongs_to_another_supplied_gold_patient=sum(any(gg[0]==k[0] and gg[2]==k[2] and gg[1]!=k[1] for gg in G) for k in false_positive),
        FP_concept_belongs_to_no_supplied_gold_patient=sum(not any(gg[0]==k[0] and gg[2]==k[2] for gg in G) for k in false_positive),
        missing_with_gold_entity=sum(bool(x['gold_entity_occurrences']) for x in missing_details),missing_with_exact_candidate_span=sum(x['exact_span_any_candidate'] for x in missing_details),missing_with_overlapping_candidate_span=sum(x['overlap_any_candidate'] for x in missing_details))
    prior=r/'research/next-direction-review/training-anchor-truncation-audit.json';old=json.loads(prior.read_text())['results']['same_CE_policy_on_training_OOF_candidates']['counts']
    report=dict(status='training_F1_OOF_structural_error_diagnostic_only',counts=counts,visibility={name:visibility(keys) for name,keys in [('covered_positive',covered),('within_FN',within),('empty_within_FN',empty),('other_patient_within_FN',other),('TP',true_positive),('FP',false_positive)]},
        prior_CE48_visibility={k:old[k] for k in ['pairs','positive_pairs','positive_no_target_any_original_local_context','positive_lost_all_target_at_top3','positive_lost_all_target_at_token_truncation']},
        inference_input_truncated_rows=sum(x['input_tokens']>14000 or x['removed_demonstrations']>0 for x in raw.values()),inference_output_hit_limit_rows=sum(x['hit_token_limit'] for x in raw.values()),
        labels='Canonical HPO only in this postrun training diagnosis. Both frozen meta trees trained original raw365; predictions/models/metrics unchanged.',
        limitations=['F1 designated meta training16, strictly excluded by SFT32; not a new outer validation or Score.','Empty-CID false negatives describe output behavior, not proof of calibrated uncertainty or a correct reason to reject.','Target marker visibility is not annotated relation evidence/coreference correctness. Categories overlap; do not add visibility counts to disjoint FN counts.','Top4 audit preserves ±500 original passages; it cannot establish distant missing evidence or whether the retained clue suffices.','No logprobs saved; no confidence threshold inferred from discrete responses.','Candidate missing combines span detection/linking; exact/overlap is a structural locator, not a supervised repair result.'],
        references={n:ref(p) for n,p in dict(source=Path(__file__),F1_gold=gold_path,F1_input=input_path,F1_candidates=candidates,prompts=b/'meta.prompts.jsonl',raw=b/'meta.responses.jsonl',raw_freeze=b/'meta-responses-freeze.json',SFT32_data=b/'train.jsonl',prompt_source=b/'prepare_pair_rag.py',base_source=b/'run.py',hpo=d/'hp.obo',oldCE_visibility=prior).items()},
        outer_or_originaldev_gold_read=False,no_new_GPU_or_inference=True)
    (out/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(out/'missing-candidate-keys.json').write_text(json.dumps(missing_details,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':ref(out/'diagnostic.json'),'counts':counts,'visibility':report['visibility'],'prior_CE48_visibility':report['prior_CE48_visibility']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
