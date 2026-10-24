"""Reuse the frozen original80 runner/merger and bounded launcher; old SFT32 only."""
import pathlib,json,hashlib,difflib
P=pathlib.Path(__file__).parent;B=pathlib.Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/sft-meta1-fusion');S=B/'json-value-weighted-cpu-preparation/F1-old80/source'
a=(S/'infer80.py').read_text();new=a.replace('original80-weighted124','original80-coreference-B-oldSFT32').replace('complete80_weighted_original_concepts_frozen','complete80_coreference_B_oldSFT32_frozen')
(P/'infer80.py').write_text(new);(P/'infer80-source.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),new.splitlines(True),fromfile=str(S/'infer80.py'),tofile='infer80.py')))
a=(S/'evaluate_f1.py').read_text();new=a.replace('weighted124 original80','coreference-B oldSFT32 original80').replace('weighted_tree','coref_tree').replace('weighted_fusion','coref_fusion').replace('complete_original124_JSON_value_weight2_fixed_original80_F1','complete_B_coreference_oldSFT32_fixed_original80_F1').replace('old A800 vs new A100 runtime difference disclosed','old SFT32 cached A800 baseline vs new A800 B-only markers; no retraining').replace("keep_by_score_increase=new['metrics']['Score']>old['metrics']['Score'],original124_JSON_and80_prompts_unchanged=True,entities1189_exact=True,no_public62_no_search=True", "keep_by_precommitted_policy=(new['metrics']['Score']>old['metrics']['Score'] and new['metrics']['macF1']>=old['metrics']['macF1']),policy='Score strictly increases and macro does not decrease; no alternate arm',old_SFT32_adapter_exact=True,only_B_markers_and_fixed_system_sentence_changed=True,entities1189_exact=True,no_union_no_intersection_no_training_no_search=True")
needle=" save(b/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))"
insert=""" # Only after complete raw/prediction freeze and the unique score, derive fixed aggregate error accounting.
 import sys,collections
 sys.path.insert(0,str(Path(r['evaluator']['path']).parent));ev=module('coref_metric_helpers',r['evaluator']['path'])
 def triples(data):return {(str(d['pmc_id']),str(pid),concept) for d in data for pid,concepts in ev.associations(d).items() for concept in concepts}
 truth=triples(rows(plan['gold_path']));before=triples(base);after=triples(newpred);added=after-before;removed=before-after;goldowners=collections.defaultdict(set)
 for doc,pid,concept in truth:goldowners[(doc,concept)].add(pid)
 def wrong_patient(v):return sum(bool(goldowners[(doc,concept)]-{pid}) for doc,pid,concept in v)
 errors=after-truth;old_errors=before-truth
 changes=dict(added_relations=len(added),added_correct=len(added&truth),added_incorrect=len(added-truth),added_wrong_supplied_patient=wrong_patient(added-truth),added_concept_in_no_gold_supplied_patient=sum(not goldowners[(doc,concept)] for doc,pid,concept in added-truth),removed_relations=len(removed),removed_correct=len(removed&truth),removed_incorrect=len(removed-truth),new_wrong_supplied_patient_total=wrong_patient(errors),old_wrong_supplied_patient_total=wrong_patient(old_errors),source='same evaluator associations canonicalization; aggregate only after unique F1 score')
 save(b/'relation-change-diagnostic.json',changes);result['relation_changes']=changes
 save(b/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))"""
assert needle in new;new=new.replace(needle,insert);(P/'evaluate_f1.py').write_text(new);(P/'evaluate-source.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),new.splitlines(True),fromfile=str(S/'evaluate_f1.py'),tofile='evaluate_f1.py')))
for n in ['infer80.py','evaluate_f1.py']:compile((P/n).read_text(),str(P/n),'exec')
(P/'control-source-manifest.json').write_text(json.dumps({'upstream':[{'path':str(S/n),'sha256':hashlib.sha256((S/n).read_bytes()).hexdigest()} for n in ['infer80.py','evaluate_f1.py']],'derived':[{'path':n,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in ['infer80.py','evaluate_f1.py']],'runner_and_parser_unmodified':True},indent=2)+'\n')
