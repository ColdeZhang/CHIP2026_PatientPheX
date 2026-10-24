#!/usr/bin/env python3
import argparse,datetime,json
from pathlib import Path
from census import load,ref,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory
    prepared=load(b/'preparation.json');cpu=load(b/'cpu-result.json');resource=load(b/'resource.json')
    assert cpu['status']=='ACTUAL_CPU_AUTOGRAD_PASS' and cpu['CUDA_initialized'] is False
    actual=load(b.parent/'actual-resource-preflight.json');assert actual['exit_code']==0;runtime=json.loads(actual['stdout'])
    assert runtime['checkpoint_supported'] and runtime['CUDA_initialized'] is False
    sourcefiles={name:r['sha256'] for name,r in prepared['files'].items()}
    for name,r in prepared['files'].items():assert ref(b/name)==r
    for name,h in cpu['sources'].items():assert name in sourcefiles and sourcefiles[name]==h
    for name in ['torch_head.py','torch_contracts.py','global_head.py','probe.py']:assert ref(Path(__file__).with_name(name))['sha256']==sourcefiles[name]
    plan=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Only two fresh single-step longest-training-document probes; no formal fit or F1 quality',
              authorization='Root granted shared180seconds+TERM15 GPU probe after actual CPU contract and checkpoint/push; no 8ep formal authorization',
              files=sourcefiles,cpu_result_sha256=ref(b/'cpu-result.json')['sha256'],operations_sha256=ref(Path(__file__).with_name('probe_operations.py'))['sha256'],
              preparation=ref(b/'preparation.json'),resource=resource,resource_evidence=ref(b.parent/'actual-resource-preflight.json'),
              selected_document=prepared['selected_document']['pmc_id'],input_selection=prepared['selection'],
              class_weights=prepared['class_weights'],class_loss=prepared['class_loss'],old_class_loss_source=prepared['old_CE_source'],
              model=runtime['model_path'],model_files=runtime['files'],gpu_uuid='GPU-07726469-bf7a-b832-9dee-019dc276dc1d',seed=20261023,
              arms=['uniform','pair'],precision='FP32; disable TF32; eager attention',encoder_trainable=True,gradient_checkpointing_required=True,checkpoint_use_reentrant=False,
              head_parameters=344577,encoder_window_microbatch=4,pairchunk=32,pairchunk_note='Each op has <=32pairs; aggregate all chunk loss graphs then one backward through shared full-document encoder graph. This is not a claim that all saved head activations are limited to one chunk.',
              LR=dict(encoder=2e-5,head=1e-4),weight_decay=.01,clip_norm=1.,optimizer='AdamW',steps_per_arm=1,
              require=dict(strict_public_encoder_weight_load=True,all_encoder_grads_finite=True,major_module_grad_nonzero_and_parameter_update=True,identical_initial_head_hash=True),
              budget_seconds=180,grace_seconds=15,save_model=False,save_optimizer=False,
              class_weight_limit='Longest input has 5 negative train pairs. Do not change selection; mixed positive/negative CE-vs-BCE/autograd checked in CPU contracts. Global counts2506/594/1912. Per-doc mean differs from old random16-example batching; same exact loss/reduction in both new arms.',
              future_proposal_only=dict(epochs=8,docbatch=1,updates_per_arm=256,fixed_probability_cut=.5,formal_authorization=False),F1_labels=False,originaldev=False)
    save(b/'probe-plan.json',plan);print(json.dumps(ref(b/'probe-plan.json')))

if __name__=='__main__':main()
