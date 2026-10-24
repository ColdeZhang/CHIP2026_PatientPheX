"""Root-authorized sole254 execution; actual admission is frozen exactly once."""
import copy
from datetime import datetime,timezone
from pathlib import Path
import time
from common import read, ref, save

S=Path(__file__).resolve().parent


def main():
    draft=read(S/'query-plan-draft.json');p=copy.deepcopy(draft['execution_template']);root=draft['remote_root']
    payload={name:ref(v['path']) for name,v in draft['payload_sources'].items()}
    payload['held_driver.py']=ref(S/'driver.py')
    for key in ['common.py','adapted_semantic.py','query.py']:
        p['source_files'][key]={**payload[key],'path':root+'/'+key}
    now=time.time();p.update(execution_authorized=True,authorized=True,authorized_budget_seconds=180,
        authorization='Root approved sole254 queries180+30 on .3 GPU0, 300s admission and fixed soleF1 quality after complete freeze; no newprobe or training',
        launch_bound_at_unix=now,launch_valid_until_unix=now+300,
        launch_bound_at_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        launch_valid_until_utc=datetime.fromtimestamp(now+300,timezone.utc).isoformat(),
        remote_root=root,python=draft['python'],storage_parent='/home/zhangyuheng/projects/CHIP2026_PatientPheX',
        environment={**draft['environment'],'CUDA_VISIBLE_DEVICES':p['gpu_uuid'],'PATIENTPHEX_DURABILITY_ROOT':'/home/zhangyuheng/projects/CHIP2026_PatientPheX'},
        argv=[draft['python'],'-u',root+'/query.py','--plan',root+'/held-query-plan.json'],
        payload_sources=payload,reuse_packages=[],reuse_actual_artifacts={},operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),
        quality_execution_authorized=True,new_v2_limit=1,old_v2_limit=0,
        input_sources=dict(preparation=draft['preparation'],draft=ref(S/'query-plan-draft.json'),decision=ref(S/'decision.json'),
            owner_certificate=read(S/'evidence/model-binding.json')['owner_certificate'],controller_contract=ref(S/'evidence/controller-contract.json')))
    files={root+'/'+name:{k:v[k] for k in ['bytes','sha256']} for name,v in payload.items()}
    for key in ['checkpoint','bank','labels','hpo']:
        v=p[key];files[v['path']]={k:v[k] for k in ['bytes','sha256']}
    manifest=read(S/'bank-manifest.json')
    for name,sha in manifest['base_model_files'].items():
        # Native inherited inventory has actual sizes of the same public base files.
        native=read(read(read(S/'evidence/preparation.json')['preparation']['path'])['inherited_native_CPU']['path'])
        item=native['assets'][name];assert item['sha256']==sha
        files[p['model_dir']+'/'+name]={k:item[k] for k in ['bytes','sha256']}
    p['files']=files
    save(S/'execution-plan.json',p);print(__import__('json').dumps(ref(S/'execution-plan.json')))


if __name__=='__main__':main()
