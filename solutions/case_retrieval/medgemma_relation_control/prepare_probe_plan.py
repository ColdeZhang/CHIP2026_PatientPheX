"""Freeze one exact probe plan after completed verified transfer; does not launch."""
from pathlib import Path
import json,hashlib
E=Path(__file__).resolve().parent;F=E/'probe-a800';Q=E/'transfer-a800-recovery/formal';R=E.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
ack=Q/'asset-copy-ack.json';assert (Q/'complete.json').exists() and ack.exists();a=json.loads(ack.read_text());assert len(a['files'])==19 and a['total_bytes']==8639802098 and not a['target_is_persistent']
assert not F.exists();F.mkdir()
prep='research/medgemma-relation-control/a800-preparation';model=a['target_directory'];out='/dev/shm/patientphex-medgemma15-1045-20261024/probe-longest-once';py='/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python'
manifest=json.loads((E/'transfer-a800/formal/run-plan.json').read_text())['files']
env={'CUDA_VISIBLE_DEVICES':'GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc','PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a800-migration/runtime','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
argv=[py,'-u',prep+'/probe_longest.py','--train',prep+'/train.jsonl','--excluded-input',prep+'/excluded.input.jsonl','--data-manifest',prep+'/preparation-manifest.json','--model',model,'--model-source-manifest',prep+'/expected-files.json','--cpu-reference',prep+'/reference/token-mask-audit.json','--output-dir',out,'--gpu-uuid',env['CUDA_VISIBLE_DEVICES']]
deltaout=prep+'/a800-runtime-delta.json';delta=[py,'-u',prep+'/check_runtime_delta.py','--model',model,'--expected-files',prep+'/expected-files.json','--reference',prep+'/reference','--train',prep+'/train.jsonl','--output',deltaout]
sources={str(x.relative_to(E)):ref(x) for x in [E/'run_probe_once.py',E/'check_runtime_delta.py',E/'probe_longest.py',E/'train_medgemma32.py',E/'medgemma_interface.py',E/'source-manifest.json']}
sourcefile=F/'source-manifest.json';sourcefile.write_text(json.dumps({'status':'exact_single_probe_sources_after_recovery','files':sources},indent=2)+'\n')
p={'mount_id':'69b991ee-183f-4add-af1a-579af92479c9','gpu_uuid':env['CUDA_VISIBLE_DEVICES'],'model':model,'shm_root':'/dev/shm/patientphex-medgemma15-1045-20261024','env':env,'remote_source_files':{prep+'/'+n:r for n,r in manifest.items()},'asset_copy_ack':ref(ack),'sequence_final_release':ref(R/'research/patient-first-closed-set/formal-preparation/inference-terminal/wrapper-complete.json'),'source_manifest':ref(sourcefile),'launcher_sha256':sha(E/'run_probe_once.py'),'common_controller_sha256':sha(R/'research/candidate-order-warmstart-entry/run_paired_ack_paths.py'),'probe_argv':argv,'probe_output':out,'runtime_delta_argv':delta,'runtime_delta_output':deltaout,'GPU_budget_seconds':180,'TERM_grace_seconds':30,'CPU_delta_seconds':45,'CPU_delta_grace_seconds':10,'full19_assets_SHA_before_GPU':True,'no_formal_training_no_quality':True,'fresh_empty_GPU_required':True,'max_native_greedy_tokens':2,'native_cache_claim_requires_actual_populated_cache_forward':True,'first_EOS_only_prefill_no_retry':True}
(F/'run-plan.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(ref(F/'run-plan.json')))
