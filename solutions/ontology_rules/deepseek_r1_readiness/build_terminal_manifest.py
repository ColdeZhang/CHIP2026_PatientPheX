"""Freeze already existing R1 assets/CPU/probe receipts; no model or remote calls."""
from pathlib import Path
import json,hashlib,datetime
D=Path(__file__).parent
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 A=D/'assets';G=D/'gpu-probe';summary=json.loads((G/'terminal-summary.json').read_text());assets=json.loads((A/'complete.json').read_text());assert assets['status']=='ASSET_FILES_AND_ACTUAL_HEADERS_VERIFIED'
 paths=['cpu-terminal-manifest.json','assets/intent.json','assets/download-job.json','assets/started.json','assets/expected-files.json','assets/derived-tensor-shapes.json','assets/download_deepseek.py','assets/complete.json','assets/actual-header-inventory.json','assets/monitor-state.json','gpu-probe/plan.json','gpu-probe/run_probe.py','gpu-probe/launch_probe.py','gpu-probe/strict_final.py','gpu-probe/start-intent.json','gpu-probe/job.json','gpu-probe/actual-prestart-check.json','gpu-probe/rotary-prestart-change.json','gpu-probe/native-buffer-source-detail.json','gpu-probe/rotary-redeploy-check.json','gpu-probe/terminal-summary.json','gpu-probe/recovered/collection-manifest.json']
 refs=[ref(D/n) for n in paths];collection=json.loads((G/'recovered/collection-manifest.json').read_text())
 for n,x in collection['files'].items():
  rr=ref(G/'recovered'/n);assert rr['bytes']==x['bytes'] and rr['sha256']==x['sha256'];refs.append(rr)
 result={'status':'R1_NATIVE_ASSETS_CPU_GPU_FORMAT_COST_TERMINAL_NOT_QUALITY','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_id':assets['model_id'],'MS_revision':assets['MS_revision'],'assets_status':assets['status'],'asset_bytes':sum(x['bytes'] for x in assets['files']),'download_reported_seconds':assets['elapsed_seconds'],'download_timing_scope':'Downloader elapsed ends immediately before small headerverification; original absolutedeadline and managedterminal receipts retained. No fabricated wrapperexitcode.','actual_header_parameters':8190735360,'actual_header_tensors':399,'CPU_terminal_sha256':ref(D/'cpu-terminal-manifest.json')['sha256'],'probe_terminal_status':summary['terminal_status'],'GPU_summary_ref':ref(G/'terminal-summary.json'),'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_namespace':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness','remote_GPU_terminal_relative':'gpu-probe-v1/terminal','no_quality_score':True,'no_full80_inference':True,'no_extra_probe_retry_or_forced_reasoning_close':True,'meaning':'Format/cost within this1024/180 budget only; truncation does not close model quality mechanism. Root determines any new budget.','original_engineering_failures_retained':'CPU BatchEncoding/list comparison and sourceplaceholder syntax; prestart rotary CPU/GPU byte check replaced beforefirstload with explicitFP32/samedevice8eps comparison. Allhistory retained.','refs':refs}
 with (D/'terminal-manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(ref(D/'terminal-manifest.json')))
if __name__=='__main__':main()
