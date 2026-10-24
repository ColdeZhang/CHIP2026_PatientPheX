"""After positive F1 only, bind unchanged mechanism to complete existing F0 streams."""
from pathlib import Path
import json
from run import ref,put,now,STREAMS
R=Path('/workspace/patientphex-runs/round2-20261023');S=Path(__file__).parent.resolve();ROOT=R/'research/unanimous-relation-veto-20261024'
def main():
 prior=json.loads((ROOT/'F1-v1/result.json').read_text());assert prior['retain'] and prior['delta']['Score']>0 and prior['new_v2']==1
 p1=json.loads((S/'plan-F1.json').read_text());assert p1['source_files']['run.py']==ref(S/'run.py')
 four=json.loads(Path('/workspace/patientphex-worktrees/explore-four-source-fusion/solutions/four_source_fusion/plan-F0.json').read_text())
 qplan=json.loads(Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations/f0-quality-plan.json').read_text())
 cp=json.loads(Path('/workspace/patientphex-worktrees/explore-patient-response-code/solutions/patient_response_code/f0/quality-plan.json').read_text())
 sp=R/'research/nonthinking-sampled-consensus-F0-20261024';spl=json.loads((sp/'gpu-plan.json').read_text());st=sp/'collected/run-v1/native';sc=json.loads((st/'complete.json').read_text())
 qt=R/'stage8-qwen3-relations/f0-v1/collected/run/native';qc=json.loads((qt/'complete.json').read_text())
 ct=R/'research/patient-response-code-v1/F0-full58-v1/terminal';cc=json.loads((ct/'summary.json').read_text())
 streams={}
 for name,seed in [('S1',20261024),('S2',20261025)]:
  z=next(x for x in sc['seeds'] if x['seed']==seed)
  streams[name]={'prompts':ref(spl['input_refs']['prompts']['path']),'canonical':ref(st/str(seed)/'canonical.jsonl'),'raw':ref(st/str(seed)/'responses.jsonl'),'prior_complete':ref(st/'complete.json'),'seed':seed}
  assert streams[name]['canonical']['sha256']==z['canonical_sha256'] and streams[name]['raw']['sha256']==z['responses_sha256']
 streams['Q']={'prompts':ref(qplan['refs']['prompts']['path']),'canonical':ref(qt/'canonical.jsonl'),'raw':ref(qt/'responses.jsonl'),'prior_complete':ref(qt/'complete.json')}
 for x in ['canonical','responses']:assert streams['Q']['raw' if x=='responses' else x]['sha256']==qc[x]['sha256']
 streams['C']={'prompts':ref(cp['prompts']['path']),'canonical':ref(ct/'canonical.jsonl'),'raw':ref(ct/'responses.jsonl'),'prior_complete':ref(ct/'summary.json')}
 assert streams['C']['canonical']['sha256']==cc['canonical_sha256'] and streams['C']['raw']['sha256']==cc['responses_sha256']
 rp=json.loads(Path('/workspace/patientphex-worktrees/explore-relation-calibration/solutions/relation_calibration/F0-plan.json').read_text());base=R/'research/four-source-fusion-20261024/F0-v1'
 refs={k:four['refs'][k] for k in ['input','gold','evaluator','validator','hpo']}
 refs.update(baseline=ref(base/'prediction.jsonl'),baseline_metrics=ref(base/'metrics.json'),baseline_freeze=ref(base/'freeze.json'),grounded_pure=ref(rp['raw_sources']['grounded']['path']),decision=ref(S/'decision.json'),prior_ancestry=rp['F0_ancestry'],coded_ancestry=ref(Path('/workspace/patientphex-worktrees/explore-four-source-fusion/solutions/four_source_fusion/plan-F0.json')))
 b=json.loads((base/'metrics.json').read_text())['metrics'];assert b['Score']==.7281956289113009
 p={**p1,'fold':0,'at_utc':now(),'stream_refs':streams,'refs':refs,'source_files':{x.name:ref(x) for x in S.glob('*.py')},'output':str(ROOT/'F0-v1'),'baseline_score':b['Score'],'baseline_relations':343,'entities':947,'patients':47,'rows_per_stream':58,'gate_result':ref(ROOT/'F1-v1/result.json'),'ancestry':'S1/S2 normal48 and F0candidate48 excludeF0; Q/C/G useF2F3 stricttrain32 and F0candidates48. Inherit205ac0a4+Q44493fcb+originalC strictF0 ancestry; no newfit. F0known development, not blind.'}
 put(S/'plan-F0.json',p)
 put(S/'evidence/F1-result.json',prior)
 w=S/'WORKLOG.md';w.write_text(w.read_text()+'\nF1唯一终点已正：13:34:36.223完整freeze后1v2/旧0，.701325377212→.702208481659（+.000883104447）；删9=2TP/7FP，micro+.002648676/mac+.000883742，其他对象/实体精确。S1/S2/Q各1557明确状态；C原40unknown保留。依原条件现固定F0来源/相同run.py，不改方法；F0实际相交/删数尚未读取。\n')
 print(json.dumps({'plan':ref(S/'plan-F0.json'),'source_unchanged':p['source_files']['run.py']==p1['source_files']['run.py'],'gate':prior['delta']['Score'],'F0_gold_not_read':True,'F0_intersections_not_computed':True}))
if __name__=='__main__':main()
