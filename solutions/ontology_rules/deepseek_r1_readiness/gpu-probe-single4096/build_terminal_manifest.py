"""Freeze existing single4096 receipts; no NN, parser, quality or remote execution."""
from pathlib import Path
import json,hashlib,datetime
G=Path(__file__).parent;D=G.parent

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 summary=json.loads((G/'terminal-summary.json').read_text());parent=ref(D/'terminal-manifest.json');assert parent['sha256']=='d172a788d6b074c2758661e0cad64a0c0f8c108b59908b0344ae70bbaaac946c'
 paths=['plan.json','run_probe.py','launch_probe.py','strict_final.py','preparation.json','source-deploy-check.json','start-intent.json','job.json','actual-prestart-check.json','summarize_terminal.py','summary-source-check.json','terminal-summary.json','recovered/collection-manifest.json'];refs=[ref(G/n) for n in paths];collection=json.loads((G/'recovered/collection-manifest.json').read_text())
 for n,v in collection['files'].items():
  r=ref(G/'recovered'/n);assert r['bytes']==v['bytes'] and r['sha256']==v['sha256'];refs.append(r)
 prefix=None
 if (G/'recovered/terminal/row0.json').exists():
  new=json.loads((G/'recovered/terminal/row0.json').read_text());old=json.loads((D/'gpu-probe/recovered/terminal/row0.json').read_text());prefix={'comparison_scope':'existing generatedIDs only, no model or quality','old1024row':ref(D/'gpu-probe/recovered/terminal/row0.json'),'newrow':ref(G/'recovered/terminal/row0.json'),'prefix_length':min(len(old['generated_ids']),len(new['generated_ids'])),'old_prefix_equal_new_prefix':old['generated_ids'][:len(new['generated_ids'])]==new['generated_ids'][:len(old['generated_ids'])],'both_fresh_from_prompt_not_KV_continuation':True}
 result={'status':'SINGLE4096_NATIVE_FORMAT_COST_TERMINAL_NOT_QUALITY','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent1024':parent,'parent_review':ref(Path('/workspace/patientphex-runs/round2-20261023/validation/deepseek-r1-format-cost-review/review.json')),'model_revision':'5ee5fa4832e658ab84298806de06ed2a167834bd','scope':'samefirstrow/index0/2171inputs/freshpromptseed;max4096,total240+TERM30,225softstop;nootherrows/full80/gold/quality/forcedreasoningclose/retry','terminal_status':summary['terminal_status'],'summary':ref(G/'terminal-summary.json'),'prefix_existing_artifact_comparison':prefix,'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_namespace':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/gpu-probe-single4096-v1','refs':refs,'quality_unknown':True}
 with (G/'terminal-manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(ref(G/'terminal-manifest.json')))
if __name__=='__main__':main()
