"""Prepare the exact query payload/cost plan; not an executable GPU authorization."""
from pathlib import Path
import json
from common import checked, deny_gold, read, ref, save

S=Path(__file__).resolve().parent


def main():
    deny_gold();e=read(S/'evidence/preparation.json');p=read(checked(e['preparation']))
    b=read(S/'bank-manifest.json');sourceplan=read(checked(p['adapted_training_plan']))
    P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';root=P+'/research/adapted-quote-linking-v1/F1-query-v1'
    oldplan=read(Path(p['original_policy']['path']).parent/'held-query-plan.json')
    template=dict(schema='adapted-Grounded-necessary-query-execution-v1',execution_authorized=False,
        launch_valid_until_unix=None,budget_seconds=180,grace_seconds=30,complete_necessary_queries=True,query_count=254,
        gpu_uuid=sourceplan['gpu_uuid'],model_dir=sourceplan['model'],checkpoint=b['checkpoint'],bank=b['bank'],
        labels=dict(path=oldplan['labels'],bytes=p['public_labels']['bytes'],sha256=p['public_labels']['sha256']),
        expected_runtime=dict(torch='2.7.0+cu128',transformers='4.50.0'),
        output_dir=root+'/output',source_files={},legacy_sources={})
    payload={name:ref(S/name) for name in ['common.py','adapted_semantic.py','query.py','policy_v1.json','bank-manifest.json']}
    payload['queries.json']=p['queries'];payload['query-native-ids.json']=p['query_native_ids'];payload['original-policy.json']=p['original_policy']
    for name in ['strict.py','semantic.py','public_bank.py']:payload['legacy/'+name]=p['sources'][name]
    for name in ['common.py','adapted_semantic.py','query.py']:
        template['source_files'][name]={**payload[name], 'path':root+'/'+name}
    for name in ['strict.py','semantic.py','public_bank.py']:
        template['legacy_sources'][name]={**payload['legacy/'+name], 'path':root+'/legacy/'+name}
    for key,name in [('queries','queries.json'),('native_ids','query-native-ids.json'),('policy','policy_v1.json'),('original_policy','original-policy.json'),('bank_manifest','bank-manifest.json')]:
        template[key]={**payload[name], 'path':root+'/'+name}
    template['hpo']={**p['hpo'],'path':oldplan['hpo']}
    training=read(checked(p['adapted_training_complete']))
    save(S/'query-plan-draft.json',dict(schema='adapted-quote-cost-and-query-preparation-v1',source=ref(__file__),
        decision=ref(S/'decision.json'),preparation=e['preparation'],contracts=ref(S/'evidence/contracts.json'),
        model_and_bank_final_before_this_plan=True,query_text_and_native_ids_frozen=True,
        mount='1593f159-19bc-41ca-8b4c-2e7455f5db99',remote_root=root,python=sourceplan['python'],
        environment={**sourceplan['environment'],'PYTHONPATH':root+':'+P+'/semantic-neural/venv/lib/python3.10/site-packages'},
        payload_sources=payload,execution_template=template,execution_forbidden_until_root_allocation=True,
        launch_protocol=dict(validity_seconds=300,entry_and_prePopen_and_preModel_checks=True,actual_runtime_separate_180plus30=True,
            controller_not_yet_bound='Bind root GPU release identity/fresh gate, absolute admission, source and unique claim before start; UNKNOWN recover original intent only'),
        cost=dict(query_count=254,input_tokens=1538,max_tokens=15,new_bank_encodes=0,new_fits=0,new_9B_generations=0,
            proposed_seconds=180,grace_seconds=30,conservative_estimate_seconds=60,estimate_not_measurement=True,
            basis='Samehost entire219updates+4x41099bank+303held took50.879s; this does full1.309GB state hash/load plus254 shortqueries and CPU distinct-concept ranking. 60s is conservative planning proxy, actual180s deadline authoritative.',
            source_complete=p['adapted_training_complete'],measured_parent_complete_seconds=training['elapsed_seconds'],
            existing_checkpoint_bytes=b['checkpoint']['bytes'],existing_bank_bytes=b['bank']['bytes'],
            estimated_host_RAM_GiB=4,estimated_GPU_GiB=4,estimated_output_MiB=2,
            no_checkpoint_or_bank_transfer_needed_on_3=True),
        remaining=['root GPU assignment and actual sourceplan admission bind/push','target HF all254 IDs check before model','complete254 raw query evidence and actual release','versioned fixed normalization and complete sole primary freeze before quality']))
    print(json.dumps(ref(S/'query-plan-draft.json')))


if __name__=='__main__':main()
