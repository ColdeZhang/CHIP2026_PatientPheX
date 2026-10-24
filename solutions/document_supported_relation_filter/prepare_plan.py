"""Bind prior frozen refs; do not compute real intersections or read gold."""
from pathlib import Path
import json
from run import ref,checked,put,now,FORMULA
S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
W=Path('/workspace/patientphex-worktrees')
def main():
 v=json.loads((W/'explore-unanimous-relation-veto/solutions/unanimous_relation_veto/plan-F1.json').read_text())
 ds=json.loads((W/'explore-span-document-support/solutions/span_document_support/plan-F1.json').read_text())
 raw=json.loads((W/'explore-relation-calibration/solutions/relation_calibration/fit-plan.json').read_text())
 four=json.loads((W/'explore-four-source-fusion/solutions/four_source_fusion/plan-F1.json').read_text())
 old_sources={
 'span_support_add_only':W/'explore-span-document-support/solutions/span_document_support/run.py',
 'unanimous_explicit_negative':W/'explore-unanimous-relation-veto/solutions/unanimous_relation_veto/run.py',
 'old_SFT_rule_intersection':W/'round2-case-retrieval/solutions/case_retrieval/operations/sft_rule_intersection/F1_run.py',
 'old_public_new_support_only':W/'round2-case-retrieval/solutions/case_retrieval/operations/public_relation_outer/f1_support.py',
 'learned_document_bag_filter':W/'round2-ontology-rules/solutions/ontology_rules/filter_relation_bags.py'}
 history=dict(status='NO_IDENTICAL_COMPLETED_FORMULA_FOUND_IN_BOUNDED_SOURCE_CHECK',at_utc=now(),sources={k:ref(p) for k,p in old_sources.items()},distinctions={'span_support_add_only':'B union supported geometry; never deletes B','unanimous_explicit_negative':'Requires four explicit negative streams; F1positive/F0negative closed','old_SFT_rule_intersection':'Filters old SFT by patient-level rules while preserves all public65 additions','old_public_new_support_only':'Only newly generated relations filtered; all old B protected','learned_document_bag_filter':'Learned doc-HPO probability .5, NIL protected; no four pure positive override'},scope='Main method map/stage6/stage7 plus five actual adjacent sources; not exhaustive proof of all filesystem history',new_formula=FORMULA,real_delete_counts_not_read=True)
 put(S/'evidence/history.json',history)
 decision=dict(experiment='E-20261024-DOC-SUPPORT-FILTER-001',at_utc=now(),base_main='504659ec5abf4a045bb2fcd6333bcd954cbe0756',formula=FORMULA,positive_source_count=2,source_policy='G pure grounded, S actual S1 intersect S2 pure positives, Q native Qwen3 pure, C original strict clinical-ID response-code pure; never Bunion as source vote',doc_policy='Public v2 document_concepts(all,text) of fixed strong Task1; valid HP compounds atomized; NO excluded; NIL exact text/case/whitespace; no aliases or hierarchy',relation_policy='Only filter existing B4 v2 literal atoms; keep original order/remaining compound atoms and other objects; zero deletion byteexact',scope='Fixed candidate reliability filter, not clinical absence/negative evidence. Streams are correlated, not statistically independent.',prelock='Source/decision/actual refs/contracts pushed before actual deletion counts or gold',fold_gate='F1 strictly above .701325377212172 then unchanged F0 strictly above .7281956289113009; both positive required for any later full64 proposal',no_scans=True,no_support_only_arm=True,no_additions=True,new_v2_per_nonzero_fold=1,old_v2=0,model=0,fit=0,remote=0,originaldev=False,official=False,diagnostic_selection_disclosure='Root observed aggregate originaldev05cfa errors before this proposal; F1/F0 are reused development folds, not blind validation.')
 put(S/'decision.json',decision)
 refs={k:v['refs'][k] for k in ['input','gold','baseline','baseline_metrics','hpo','evaluator','validator']}
 refs.update(task1=ds['refs']['task1'],decision=ref(S/'decision.json'))
 pure=dict(G=raw['raw_sources']['grounded'],Q=raw['raw_sources']['qwen3'],C=four['refs']['coded'])
 for k,q in refs.items():
  if k!='gold':checked(q)
 for q in pure.values():checked(q)
 checked(raw['raw_sources']['sampled_diagnostics'])
 p=dict(experiment=decision['experiment'],at_utc=now(),fold=1,formula=FORMULA,positive_source_count=2,refs=refs,pure_sources=pure,sampled_diagnostics=raw['raw_sources']['sampled_diagnostics'],sample_rows_per_seed=80,baseline_relations=524,baseline_score=.701325377212172,entities=1189,patients=39,source_files={p.name:ref(p) for p in sorted(S.glob('*.py'))},output=str(R/'research/document-supported-relation-filter-20261024/F1-v1'),history=ref(S/'evidence/history.json'),ancestry_refs=[raw['ancestry'],*ds['ancestry_refs'][-2:],ref(W/'explore-four-source-fusion/solutions/four_source_fusion/plan-F1.json')],ancestry='Inherited actual all sources excludeF1; Task1 LGB uses train48 while three generative backbones train32; no new learning. S canonical relations inherited from full80 row diagnostics, C strict whole-row abstention unchanged.')
 for q in p['ancestry_refs']:checked(q)
 put(S/'plan-F1.json',p);print(json.dumps(dict(plan=ref(S/'plan-F1.json'),decision=ref(S/'decision.json'))))
if __name__=='__main__':main()
