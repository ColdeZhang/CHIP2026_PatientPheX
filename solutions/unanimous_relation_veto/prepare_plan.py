"""Bind pre-existing source files, not pair intersections or held labels."""
from pathlib import Path
import json
from run import ref,put,now,STREAMS
import contracts
R=Path('/workspace/patientphex-runs/round2-20261023');S=Path(__file__).parent.resolve()
def main():
 root=R/'research/unanimous-relation-veto-20261024';root.mkdir(exist_ok=True)
 policy=dict(experiment='E-20261024-UNANIMOUS-VETO-001',at_utc=now(),base_main='0878cbdf72972e459c755b16f1fb9e94b148a5a8',
  formula='B4 minus (S1neg intersect S2neg intersect Qneg intersect Cneg minus Gpositive)',
  definitions={'negative':'At least one exact atomic candidate for the actual patient; every relevant row complete strict/noncap/naturalEOS, all reject this patient. Any positive, missing candidate/patient, missing row, parsefail, cap or abstention keeps.',
   'compound':'Positive compound splits using frozen v2; rejected compound never authorizes atom negatives. A related compound-only rejection remains unknown even when another row offers an atomic negative.',
   'literal_NIL':'Existing literal candidate text unchanged (including semicolons), never split unless entire string is pure compound HPO. Empty/-1/mixed HP strings do not authorize negative.',
   'G':'Use pure grounded prediction as positive protection only; absence is not negative.',
   'render':'Preserve original JSON bytes outside association, patient/association object fields and order; remove only proven atoms, retain order inside partially retained pure-HPO compounds.'},
  endpoints={'F1':'current B4 .701325377212172/524 pairs; exactly one full prediction; no deletion=>byteexact0v2 close; otherwise freeze1v2/old0; totalScore strictly higher to proceed',
   'F0':'Only if F1 strictly wins; identical method on current B4 .728195628911/343; source SHA plan freeze before actual intersection; no alternate cut or subset',
   'full64':'Only both internal folds strictly positive; no automatic training or originaldev scoring in this task'},
  new_model_fit_remote=0,development_not_blind=True,no_threshold_or_subset_scan=True)
 put(S/'decision.json',policy)
 four=json.loads(Path('/workspace/patientphex-worktrees/explore-four-source-fusion/solutions/four_source_fusion/plan-F1.json').read_text())
 base=R/'research/four-source-fusion-20261024/F1-v1'
 qplan=json.loads(Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations/quality-plan.json').read_text())
 cp=json.loads(Path('/workspace/patientphex-worktrees/explore-patient-response-code/solutions/patient_response_code/quality-plan.json').read_text())
 sp=R/'research/nonthinking-sampled-consensus-native-20261024';spl=json.loads((sp/'gpu-plan.json').read_text());st=sp/'collected/run-v1/native';sc=json.loads((st/'complete.json').read_text())
 qt=R/'stage8-qwen3-relations/collected/run/infer';qc=json.loads((qt/'complete.json').read_text())
 ct=R/'research/patient-response-code-v1/F1-full80-v1/terminal';cc=json.loads((ct/'summary.json').read_text())
 streams={}
 for name,seed in [('S1',20261024),('S2',20261025)]:
  summary=next(x for x in sc['seeds'] if x['seed']==seed)
  streams[name]={'prompts':ref(spl['input_refs']['prompts']['path']),'canonical':ref(st/str(seed)/'canonical.jsonl'),'raw':ref(st/str(seed)/'responses.jsonl'),'prior_complete':ref(st/'complete.json'),'seed':seed}
  assert streams[name]['canonical']['sha256']==summary['canonical_sha256'] and streams[name]['raw']['sha256']==summary['responses_sha256']
 streams['Q']={'prompts':ref(qplan['refs']['prompts']['path']),'canonical':ref(qt/'canonical.jsonl'),'raw':ref(qt/'responses.jsonl'),'prior_complete':ref(qt/'complete.json')}
 for x in ['canonical','responses']:
  k='raw' if x=='responses' else x
  assert streams['Q'][k]['sha256']==qc[x]['sha256']
 streams['C']={'prompts':ref(cp['prompts']['path']),'canonical':ref(ct/'canonical.jsonl'),'raw':ref(ct/'responses.jsonl'),'prior_complete':ref(ct/'summary.json')}
 assert streams['C']['canonical']['sha256']==cc['canonical_sha256'] and streams['C']['raw']['sha256']==cc['responses_sha256']
 rp=json.loads(Path('/workspace/patientphex-worktrees/explore-relation-calibration/solutions/relation_calibration/fit-plan.json').read_text())
 refs={k:four['refs'][k] for k in ['input','gold','evaluator','validator','hpo']}
 # Gold identity is inherited metadata only: never read/hash it during planning.
 refs.update(baseline=ref(base/'prediction.jsonl'),baseline_metrics=ref(base/'metrics.json'),baseline_freeze=ref(base/'freeze.json'),grounded_pure=ref(rp['raw_sources']['grounded']['path']),decision=ref(S/'decision.json'),prior_ancestry=rp['ancestry'])
 prior=json.loads((base/'metrics.json').read_text())['metrics'];assert prior['Score']==.701325377212172
 contract=contracts.main();put(S/'evidence/contracts.json',contract)
 plan=dict(id=policy['experiment'],fold=1,at_utc=now(),formula=policy['formula'],streams=list(STREAMS),stream_refs=streams,refs=refs,
  source_files={p.name:ref(p) for p in S.glob('*.py')},contracts=ref(S/'evidence/contracts.json'),output=str(root/'F1-v1'),baseline_score=prior['Score'],baseline_relations=524,entities=1189,patients=39,rows_per_stream=80,
  ancestry='All four neural streams use strictF2F3 train32 and F1 candidate sources; Gsame. B4 inherited source/review; older supportLGB maytrain48 but excludesF1. No fitting in this control.',
  source_manifest_binding='Canonical/raw SHA verified against original native completion metadata; raw text never reparsed; row IDs joined explicitly.',
  forbidden_before_freeze=['held gold read/hash','v2','actual alternative subset/cut search'])
 put(S/'plan-F1.json',plan)
 print(json.dumps({'plan':ref(S/'plan-F1.json'),'decision':ref(S/'decision.json'),'contracts':contract['checks'],'actual_pair_intersections_not_read':True}))
if __name__=='__main__':main()
