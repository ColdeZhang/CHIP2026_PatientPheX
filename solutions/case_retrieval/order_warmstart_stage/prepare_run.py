"""Freeze the selected paired fresh-stage plan; no MCP, GPU, or gold access."""
from pathlib import Path
import datetime,hashlib,json

E=Path(__file__).resolve().parent
R=E.parent.parent
B=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
BASE='case-retrieval/qwen35-preparation'
REMOTE=BASE+'/order-warmstart-stage'
PY='/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python'

def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')

def main():
    old=json.loads((B/'run-plan.json').read_text())
    quality=json.loads((B/'json-value-weighted-cpu-preparation/F1-old80/run-plan.json').read_text())
    # The historical endpoint contains a gold path/hash; copy only strings, never open it here.
    common=['tree_helper','fusion_helper','parser','parser_base','input','old_entities','model_source','prompts',
            'old_baseline_pred','old_baseline_metrics','hpo','evaluator','validator','runner']
    evaluation={'schema':'patientphex-order-warmstart-paired-F1-v1','control_sha256':sha(E/'evaluate_paired.py'),
        'python':quality['python'],'trainer_sha256':sha(E/'train_order_warmstart.py'),
        'gold_path':quality['gold_path'],'historical_F1_gold_sha256':quality['historical_F1_gold_sha256'],
        'references':{k:quality['references'][k] for k in common},'arms':{'original':{},'reverse-first':{}}}
    for record in evaluation['references'].values():assert ref(record['path'])==record
    save(E/'evaluation-template.json',evaluation)
    refs={name:ref(E/(name+'.py')) for name in ['run_paired','evaluate_paired','train_order_warmstart',
          'export_checkpoint_server','fetch_checkpoint','freeze_evaluation_plan','prepare_run']}
    refs.update({name:ref(E/name) for name in ['cpu-runtime-complete.json','export-cpu-contract.json',
        'cpu-source-contract.json','matched-32-call-schedule.json','trainer-source-manifest.json']})
    refs.update({k:old['local_references'][k] for k in ['train','excluded','data_manifest','base_source','meta_prompts','runner']})
    refs['reverse_train']=ref(E.parent/'candidate-order-training-feasibility/reverse124.jsonl')
    actual=json.loads((E/'deployment-plan.json').read_text())['remote_sha256']
    expected={BASE+'/'+n:h for n,h in old['remote_sources_sha256'].items() if not n.endswith('train_qwen35_meta32.py')}
    expected.update(actual)
    uuid=old['gpu_uuid'];env=dict(old['env']);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    plan={'schema':'patientphex-order-warmstart-paired-native-v1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'controller_sha256':sha(E/'run_paired.py'),'references':refs,'evaluation_template':ref(E/'evaluation-template.json'),
      'cap_seconds':3900,'grace_seconds':30,'mount_id':old['mount_id'],'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c',
      'gpu_uuid':uuid,'shm_root':'/dev/shm/patientphex-order-warm32-1045-20261024',
      'remote_root':REMOTE,'persistent_root':REMOTE+'/checkpoints','persistent_tools':REMOTE+'/tools',
      'fetcher_sha256':sha(E/'fetch_checkpoint.py'),'remote_source_sha256':expected,
      'training_common_argv':[PY,'-u',REMOTE+'/train_order_warmstart.py','--mode','train',
        '--train',BASE+'/sft-meta1-fusion/train.jsonl','--excluded-input',BASE+'/sft-meta1-fusion/excluded.input.jsonl',
        '--data-manifest',BASE+'/sft-meta1-fusion/preparation-manifest.json','--model','/dev/shm/patientphex-qwen35-1045-c20223623576',
        '--model-source-manifest',BASE+'/model-source-manifest.json','--warm-start-checkpoint',BASE+'/sft-meta1-fusion/formal2ep/epoch-002',
        '--reverse-train',REMOTE+'/reverse124.jsonl','--gpu-uuid',uuid],
      'inference_common_argv':[PY,'-u',BASE+'/sft-meta1-fusion/qwen35_runner.py','--model','/dev/shm/patientphex-qwen35-1045-c20223623576',
        '--source-manifest',BASE+'/model-source-manifest.json','--prompts',BASE+'/sft-meta1-fusion/meta.prompts.jsonl',
        '--max-input-tokens','14000','--max-new-tokens','1600'],
      'env':env,'prompts_sha256':old['meta_prompts_sha256'],'runner_sha256':old['runner_sha256'],
      'model_source_sha256':quality['references']['model_source']['sha256'],'row_documents':old['meta_row_documents'],
      'policy':{'arms':['original','reverse-first'],'sequential_on_one_gpu':True,'old_stage_epochs':2,'new_stage_epochs':2,
        'fresh_optimizer_rng_lr':True,'not_exact_old_optimizer_resume':True,'exposures_per_arm':248,'optimizer_calls_per_arm':32,
        'complete_epoch_exports_before_resume':True,'base_and_adapter_retained_on_disk':True,
        'shared_budget_includes_train_export_infer':True,'minimum_shm_and_persistent_free_bytes':4294967296,
        'quality_only_after_both_fullpred_freeze':True,'quality_evaluations_per_arm':1,'no_official_upload':True,
        'interruption':'Inspect recorded process IDs and complete persistent ack. Never rerun controller or duplicate ambiguous start; no budget extension.',
        'time_estimate_seconds':{'training':2299,'inference':565,'four_exports_proxy':224,'extra_loads_proxy':50,'margin':762},
        'estimates_not_guarantees':True}}
    target=E/'formal';target.mkdir(exist_ok=False);save(target/'run-plan.json',plan)
    print(json.dumps({'plan':ref(target/'run-plan.json'),'evaluation_template':ref(E/'evaluation-template.json'),'not_started':True}))

if __name__=='__main__':main()
