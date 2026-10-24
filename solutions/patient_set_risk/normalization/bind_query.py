"""CPU-only dynamic package binding; transport/start remain owner operations."""
import argparse,time,zipfile
from pathlib import Path
from common import *

def bind(preparation,remote_root,shared_term,shared_kill,output):
    prep=load(preparation);verify(prep['queries']);assert prep['query_count']>0
    assert shared_kill==shared_term+30
    original();assets=load(HERE/'target-assets100.json');out=Path(output);out.mkdir(parents=True,exist_ok=False);remote=Path(remote_root)
    payload={name:HERE/name for name in ('common.py','query.py','query_driver.py','reuse-manifest.json')}
    payload.update({name:HERE/name for name in load(HERE/'reuse-manifest.json')})
    payload['queries.json']=Path(prep['queries']['path'])
    files={str(remote/n):{k:v for k,v in ref(p).items() if k!='path'} for n,p in payload.items()};files.update(assets['files'])
    issued=time.time();estimate=20+.02*prep['query_count'];assert issued+estimate+5<shared_term
    env=dict(PYTHONPATH=str(remote)+':'+assets['pythonpath'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=assets['gpu_uuid'])
    p=dict(schema='scst-train-only-public-queries-v1',authorized=True,authorization='Root fixed4x2 SCST pilot: all necessary public queries on .100 GPU0, within original1800 deadline',
           budget_seconds=180,grace_seconds=30,issued_unix=issued,admission_expires_unix=issued+300,
           shared_TERM_unix=shared_term,shared_KILL_unix=shared_kill,query_count=prep['query_count'],complete_necessary_queries=True,
           cost_estimate_seconds=estimate,cost_basis='Prior624 actual15.347s; conservative20+.02*n, no promise of runtime',
           preparation=ref(preparation),remote_root=str(remote),output_dir=str(remote/'output'),files=files,
           query_source=str(remote/'query.py'),queries=str(remote/'queries.json'),policy=str(remote/'reuse/solutions/grounded_generation/normalization/policy_v1.json'),
           environment=env,mount=assets['mount'],assets_inherited=ref(HERE/'target-assets100.json'),
           **{k:assets[k] for k in ('python','gpu_uuid','model_dir','hpo','bank','labels','bank_manifest','expected_encoder_runtime')})
    save(out/'query-plan.json',p);payload['query-plan.json']=out/'query-plan.json'
    with zipfile.ZipFile(out/'deployment.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in payload.items():z.write(path,name)
    result=dict(schema='scst-query-deployment-package-v1',archive=ref(out/'deployment.zip'),plan=ref(out/'query-plan.json'),
                members={n:ref(path) for n,path in payload.items()},remote_root=str(remote),mount=assets['mount'],environment=env,
                argv=[p['python'],'-u',str(remote/'query_driver.py'),'--plan',str(remote/'query-plan.json'),'--plan-sha',ref(out/'query-plan.json')['sha256']],
                no_remote_action=True,requires_fsync_reopen_all_payload_before_single_start=True,
                collect_all=['driver-started.json','child-started.json','fresh-resource-gate.json','stdout.log','stderr.log','wrapper-complete.json','output/started.json','output/encoder-runtime.json','output/query-vectors.npy','output/query-token-counts.json','output/query-scores.json','output/complete.json'])
    save(out/'package.json',result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--preparation',required=True);a.add_argument('--remote-root',required=True);a.add_argument('--shared-term',required=True,type=float);a.add_argument('--shared-kill',required=True,type=float);a.add_argument('--output',required=True);a=a.parse_args()
    r=bind(a.preparation,a.remote_root,a.shared_term,a.shared_kill,a.output);print(json.dumps(dict(package=str(Path(a.output)/'package.json'),argv=r['argv'])))
