"""Bind the full47 inference after actual CPU success and prior GPU release."""
import argparse,json,sys
from pathlib import Path
S=Path(__file__).resolve().parent;C=S.parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/copy-grounded-decode-v1/F0-v1');P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/copy-grounded-decode-v1/F0-v1/infer';CPU=P+'/copy-grounded-decode-v1/F0-v1/cpu'
sys.path.insert(0,str(C));from local_common import ref,save
SG=Path('/workspace/patientphex-worktrees/explore-grounded-sampled-consensus/solutions/grounded_sampled_consensus/f0')
def read(p):return json.loads(Path(p).read_text())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release-evidence',required=True);a=ap.parse_args()
 col=read(D/'cpu/collection.json');assert col['CPU_reported_success'] and col['all_actual_SHA_verified']
 cp=read(S/'cpu-plan.json');result=read(D/'cpu/terminal/actual/result.json');assert result['status']=='PASS_ACTUAL_HF_CPU_ORDERED_TRIE47' and result['rows']==47
 assert result['all47_native_HF_original_IDs_template_decode_exact'] and result['no_weights_no_forward_no_generate_no_GPU_no_quality']
 assert result['source']['sha256']==ref(S/'cpu_runtime.py')['sha256'] and result['plan']['sha256']==ref(S/'cpu-plan.json')['sha256']
 wrapper=read(a.release_evidence);assert wrapper['target_has_no_process'] and 'GPU-126293b9-6677-8464-cee7-01e693efbf29' not in wrapper['gpu_release']['processes']
 p=read(C/'infer-plan.json');old=read(SG/'infer-plan.json');p.update(schema='grounded-copy47-inference-v1',id='E-20261024-COPY-GROUNDED-001-F0',authorization='Root fixed completeF0 after positiveF1 post-hoc abstaining pipeline; all47greedy2400+30, strict failed rows whole-row abstain, no probes/regeneration',budget_seconds=2400,authorized_budget_seconds=2400,expected_rows=47,expected_status='COMPLETE_FRESH47_CONSTRAINED_RAW_NO_QUALITY',output_dir=ROOT+'/output',argv=[p['python'],'-u',ROOT+'/infer.py','--plan',ROOT+'/infer-plan.json'],operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),binding_source=ref(__file__),input_sources=dict(F1_result=ref(C/'evidence/F1-fallback-v2-result.json'),F0_decision=ref(S/'decision.json'),actual_CPU_collection=ref(D/'cpu/collection.json'),actual_CPU_plan=ref(S/'cpu-plan.json')),constraint_cpu_source_sha256=ref(S/'cpu_runtime.py')['sha256'],constraint_cpu_plan_sha256=ref(S/'cpu-plan.json')['sha256'],predecessor_release=dict(confirmed_by_owner=True,actual=ref(a.release_evidence),gpu_release=wrapper['gpu_release']),primary=read(S/'decision.json')['formula'])
 for key in ['prompts','held_ids','held_audit','native_cpu_result']:p['assets'][key]=old['assets'][key]
 p['assets']['candidates']=cp['assets']['candidates'];p['assets']['constraint_cpu_result']=dict(path=CPU+'/actual/result.json',**ident(ref(D/'cpu/terminal/actual/result.json')))
 payload={name:ref((S if name in ['infer.py','job_driver.py'] else C)/name) for name in ['infer.py','job_driver.py','raw_store.py','processors.py','triple_trie.py']};p['payload_sources']=payload;p['files']={ROOT+'/'+n:ident(v) for n,v in payload.items()}
 for key,name in [('runner','infer.py'),('driver','job_driver.py'),('raw_store','raw_store.py'),('processors','processors.py'),('trie','triple_trie.py')]:p['assets'][key]=dict(path=ROOT+'/'+name,**ident(payload[name]))
 save(S/'infer-plan.json',p);print(json.dumps(ref(S/'infer-plan.json')))
if __name__=='__main__':main()
