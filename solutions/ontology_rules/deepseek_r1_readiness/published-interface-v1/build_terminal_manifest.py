"""Freeze published input/derived config/actual pilot receipts; no inference or quality."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).parent;D=P.parent;G=P/'gpu';C=P/'cpu'
def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 summary=json.loads((G/'terminal-summary.json').read_text());cpu=json.loads((C/'actual-result.json').read_text());assert cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT'
 refs=[ref(P/n) for n in ['cpu/prepare_cpu.py','cpu/plan.json','cpu/actual-result.json','cpu/actual-tool.json','gpu/run_probe.py','gpu/launch_probe.py','gpu/strict_final.py','gpu/sparse_final.py','gpu/plan.json','gpu/source-check.json','gpu/source-deploy-check.json','gpu/actual-prestart-check.json','gpu/start-intent.json','gpu/job.json','gpu/summarize_terminal.py','gpu/terminal-summary.json','gpu/recovered/collection-manifest.json']]
 for n in ['derived-config.json','effective-config.json','selected-prompts.jsonl','selected-encoded.json','all80-input-audit.json']:
  x=ref(C/'recovered'/n);e=cpu['outputs'][n];assert x['sha256']==e['sha256'] and x['bytes']==e['bytes'];refs.append(x)
 collection=json.loads((G/'recovered/collection-manifest.json').read_text())
 for n,e in collection['files'].items():
  x=ref(G/'recovered'/n);assert x['sha256']==e['sha256'] and x['bytes']==e['bytes'];refs.append(x)
 ancestry=[ref(D/n) for n in ['terminal-manifest.json','gpu-probe-single4096/terminal-manifest.json','decoder-cpu-diagnostic/actual-result.json','vllm-cpu-compat/actual-result.json']]
 result={'status':'PUBLISHED_INTERFACE_PILOT_FROZEN_FIRST_FORMAT_CONTRACT_FAILURE_NO_QUALITY','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':ref(G/'terminal-summary.json'),'CPU_full80_rows':80,'actual_input_contract':'raw==generic==nativeIDs and both roundtrip all80; complete raw/generic backend components equal; selected originalfirst/index0 and earliest tiedlongest/index73','actual_selected_input_tokens':[2014,5689],'CPU_lengths_upper_middle':3131,'CPU_lengths_conventional_median':3129,'two_interface_fixes':'Published JSON tokenizer plus fixed publicvLLM attention_factor1.0 semantics, originalconfig bytes retained; no single-variable causalclaim','policy':'Strict EveryCID reported separately from legal sparse; missingCID abstention, never zero-filled. NaturalEOS/complete think/strict JSON/allowed uniqueCID/PID mandatory; no fence stripping/type conversion/repair.','actual_completed_rows':len(summary['rows']),'longest_was_not_run':len(summary['rows'])==1,'terminal_status':summary['terminal_status'],'quality_unknown':True,'no_full80_quality_or_gold':True,'runtime_assets_and_old_engineering_records_preserved':ancestry,'remote_all80_encoded':cpu['outputs']['all80-published-encoded.jsonl.gz'],'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_namespace':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1','refs':refs}
 with (P/'terminal-manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(ref(P/'terminal-manifest.json')))
if __name__=='__main__':main()
