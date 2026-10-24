"""Freeze new-codec existing-first CPU plus single fresh longest GPU, no quality."""
from pathlib import Path
import json,hashlib,datetime
C=Path(__file__).parent;P=C.parent;G=C/'gpu'
def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 summary=json.loads((G/'terminal-summary.json').read_text());cpu=json.loads((C/'actual-cpu-result.json').read_text());assert cpu['status']=='PASS_FIXED_CODEC_CPU_ONLY_ON_EXISTING_FIRST746'
 paths=['normalized_final.py','strict_final.py','check_existing_first_cpu.py','codec-plan.json','actual-cpu-result.json','cpu-tool.json','gpu/run_probe.py','gpu/launch_probe.py','gpu/strict_final.py','gpu/sparse_final.py','gpu/normalized_final.py','gpu/plan.json','gpu/source-check.json','gpu/source-deploy-check.json','gpu/actual-prestart-check.json','gpu/start-intent.json','gpu/job.json','gpu/summarize_terminal.py','gpu/terminal-summary.json','gpu/recovered/collection-manifest.json'];refs=[ref(C/n) for n in paths]
 collection=json.loads((G/'recovered/collection-manifest.json').read_text())
 for n,e in collection['files'].items():
  r=ref(G/'recovered'/n);assert r['sha256']==e['sha256'] and r['bytes']==e['bytes'];refs.append(r)
 result={'status':'FIXED_NORMALIZATION_CODEC_FIRST_CPU_LONGEST_GPU_TERMINAL_NO_QUALITY','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_published_oldpolicy':ref(P/'terminal-manifest.json'),'parent_published_input_CPU':ref(P/'cpu/actual-result.json'),'first746_CPU':ref(C/'actual-cpu-result.json'),'first_was_not_regenerated':True,'old_first_raw_and_strict_rejection_unchanged':True,'new_longest_GPU_summary':ref(G/'terminal-summary.json'),'new_longest_input_tokens':5689,'new_longest_original_row_index':73,'fresh_fixed_seed_not_prior_RNG_continuation':True,'budget':'4096newtokens,240s+TERM30,225softstop','codec_scope':'Singlewholefinaljson/emptyfence,exactnonnegativeinttoexistingP<decimal>,allowedstringsunchanged,duplicatesincludingpostmapreject;missingCIDabstention neverfill;EOS/completeoptionalreasoningrequired;invalidnorepair','quality_unknown':True,'no_full80_or_gold':True,'terminal_status':summary['terminal_status'],'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_namespace':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1/codec-longest-v1','refs':refs}
 with (C/'terminal-manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(ref(C/'terminal-manifest.json')))
if __name__=='__main__':main()
