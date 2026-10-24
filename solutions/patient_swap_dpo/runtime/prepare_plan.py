"""Build a disabled execution template and immutable local deployment objects."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def ref(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main(output):
    S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023')
    D=R/'research/patient-swap-dpo-readiness-20261024';O=Path(output);O.mkdir(parents=True,exist_ok=False)
    Q=Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations')
    old=json.loads((Q/'f0-plan.json').read_text());parent='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
    remote=parent+'/research/patient-swap-dpo-v1';application=remote+'/application'
    def remote_ref(p,name):r=ref(p);r['path']=remote+'/inputs/'+name;return r
    rows=[json.loads(x) for x in gzip.decompress((D/'cpu-v1/encoded-pairs.jsonl.gz').read_bytes()).splitlines()]
    sizes=[(max(len(x['chosen_input_ids']),len(x['rejected_input_ids'])),max(len(x['chosen_input_ids']),len(x['rejected_input_ids']))-x['prompt_length']) for x in rows]
    pareto=[i for i,a in enumerate(sizes) if not any(b[0]>=a[0] and b[1]>=a[1] and b!=a for b in sizes)]
    order=sorted(range(74),key=lambda i:(sum(map(len,[rows[i]['chosen_input_ids'],rows[i]['rejected_input_ids']])),rows[i]['row_id']))
    inherited=R/'research/mention-hard-negatives-v1/gpu-v1-collected/formal/verified-assets.json'
    assert ref(inherited)['sha256']=='cd37dad5f2bc2a0a9d39364f0e3e0be377f8f917a24c59cfb79114f3e87fcff3'
    sources={p.name:ref(p)['sha256'] for p in sorted(S.glob('*.py')) if p.name!='prepare_plan.py'}
    sources['../dpo_ops.py']=ref(S.parent/'dpo_ops.py')['sha256']
    p={k:old[k] for k in ['model_id','model_dir','model_files','native_forward_sha256','mount_id','gpu_uuid','python','environment','runtime']}
    p.update(id='E-20261024-PATIENT-SWAP-DPO-001',execution_authorized=False,gpu_index=0,
             seed=20261024,run_root=remote+'/run',durability_root=remote,application=application,source_files=sources,
             parent_adapter_dir=old['adapter_dir'],parent_adapter_files={n:dict(path=old['adapter_dir']+'/'+n,**r) for n,r in old['adapter_files'].items()},
             inherited_assets=remote_ref(inherited,'inherited-assets.json'),
             pairs=remote_ref(D/'cpu-v1/pairs.jsonl.gz','pairs.jsonl.gz'),encoded=remote_ref(D/'cpu-v1/encoded-pairs.jsonl.gz','encoded-pairs.jsonl.gz'),
             CPU_ready=None,
             recipe=dict(epochs=1,rows=74,pairs=74,documents=17,batch_size=1,gradient_accumulation=8,updates=10,updates_per_epoch=10,
                         learning_rate=1e-5,weight_decay=.01,warmup_steps=1,max_grad_norm=1.,beta=.1,dropout=0.,
                         quantization=False,packing=False,truncation=False,loss='original reference-relative sigmoid DPO summed fullsuffix plus EOS and newline'),
             lora=json.loads((Q/'plan.json').read_text())['lora'],
             pareto_indices=pareto,probe_pair_indices=order[:11]+pareto,
             budgets=dict(reference=900,probe=600,formal=1800,infer=1200,TERM_grace=30),
             budget_history='Readiness proposal240/600/1200 superseded before GPU by root authorized600probe/900reference/1800formal/1200infer, allowing load and complete state ACK.',
             objective='Freeze original543c reference once; discard probe; fresh original543c LoRA warmstart+newAdam one74pair epoch; no hyperparameter scans.',
             primary='B471 UNION at_least_two(G,S,Q_DPO,C)',quality_calls_before_complete_freeze=0)
    assert len(pareto)>=1 and max(x[0] for x in sizes)==5534
    (O/'disabled-plan.json').write_text(json.dumps(p,indent=2)+'\n')
    deployment=[]
    for f in [*sorted(S.glob('*.py')),S.parent/'dpo_ops.py']:
        if f.name=='prepare_plan.py':continue
        destination=application+'/runtime/'+f.name if f.parent==S else application+'/'+f.name
        deployment.append(dict(local=ref(f),remote=destination))
    for src,name in [(inherited,'inherited-assets.json'),(D/'cpu-v1/pairs.jsonl.gz','pairs.jsonl.gz'),(D/'cpu-v1/encoded-pairs.jsonl.gz','encoded-pairs.jsonl.gz')]:
        deployment.append(dict(local=ref(src),remote=remote+'/inputs/'+name))
    deployment.append(dict(local=ref(O/'disabled-plan.json'),remote=application+'/disabled-plan.json'))
    (O/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    print(json.dumps(dict(files=len(deployment),pareto=pareto,pressure_pairs=len(p['probe_pair_indices']),plan=ref(O/'disabled-plan.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
