"""Read existing metadata and pure-input sources; no weights/gold/model/evaluation."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from delta_mean import validate_configs


def ref(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def read(path): return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--grounded-source',type=Path,required=True)
    parser.add_argument('--main',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); R=args.runs; G=R/'research/grounded-generation-v1'; S=args.grounded_source
    terminal=G/'formal-v1/terminal'; collection=read(G/'formal-v1/collection.json')
    assert collection['mount']=='476f81cf-09e7-445b-a7f0-d0e12b12e50f' and collection['all_actual_SHA_verified']
    configs=[];parents=[];fingerprints=[];refs={}
    for epoch in (1,2):
        tag=f'epoch-{epoch:03d}'; config=terminal/f'output/{tag}/adapter/adapter_config.json'
        checkpoint=terminal/f'output/{tag}/checkpoint.json';ack=terminal/f'output/durability-{tag}.json'
        for path in (config,checkpoint,ack):
            name=path.relative_to(terminal).as_posix()
            assert ref(path)==collection['files'][name]
            refs[name]=ref(path)
        c=read(config);p=read(checkpoint);a=read(ack);configs.append(c);fingerprints.append(p['fingerprint'])
        key=f'output/{tag}/adapter/adapter_model.safetensors'
        weight=collection['actual_external_checkpoints'][key]
        assert a['files']['adapter/adapter_model.safetensors']['sha256']==weight['sha256']==p['files']['adapter/adapter_model.safetensors']
        assert a['checkpoint_metadata_sha256']==ref(checkpoint)['sha256']
        assert a['files']['adapter/adapter_config.json']['sha256']==ref(config)['sha256']
        assert p['completed_epochs']==epoch and p['global_step']==epoch*11 and not p['smoke_only']
        assert a['all_members_reopened_and_SHA_verified'] and weight['actual_fsynced_and_reopened_SHA']
        parents.append(dict(epoch=epoch,global_step=11*epoch,mount=collection['mount'],
            checkpoint=ref(checkpoint),adapter_config=ref(config),durability=ref(ack),
            adapter=weight,source_status='Previously actualfsync/reopen/SHA verified in original collection; not re-polled remotely in this readiness'))
    assert fingerprints[0]==fingerprints[1]
    proposed=validate_configs(*configs)
    run=read(terminal/'output/run_manifest.json');coverage=run['loaded']['coverage']
    assert coverage['base_parameters']==8953803264 and coverage['adapter_parameters']==43278336
    assert len(coverage['target_module_names'])==248 and len(coverage['all_trainable_names'])==496
    assert set(proposed['target_modules'])==set(coverage['target_counts'])
    target=read(G/'targets-v2/manifest.json'); fp=fingerprints[0]
    assert fp['train_sha256']==target['artifacts']['train']['sha256']
    assert set(fp['train_document_ids'])==set(target['train_ids'])
    assert len(target['train_ids'])==32 and len(target['held_ids'])==16 and not set(target['train_ids'])&set(target['held_ids'])
    assert target['no_held_or_original_dev_gold_access'] and target['input_contains_no_gold_entities_HPO_candidates_or_relations']
    held=Path(target['artifacts']['held_input']['path']); assert ref(held)==target['artifacts']['held_input']
    held_rows=[json.loads(line) for line in held.read_text().splitlines()]
    assert len(held_rows)==39 and len({(r['pmc_id'],r['target_patient_id']) for r in held_rows})==39
    assert all('answer' not in r for r in held_rows) and {r['pmc_id'] for r in held_rows}==set(target['held_ids'])
    audit_path=G/'token-census-v1/F1-input-audit.json';audit=read(audit_path)
    assert {r['row_id'] for r in audit}=={r['row_id'] for r in held_rows}
    assert sum(r['input_tokens'] for r in audit)==203588 and max(r['input_tokens'] for r in audit)==10080
    infer=read(S/'inference/infer-plan.json'); inp=infer['payload_sources']
    for name in ['held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json','normalization-policy.json']:
        assert ref(inp[name]['path'])==inp[name];refs[name]=inp[name]
    policy=read(S/'inference/policy.json'); assert policy['generation']==dict(do_sample=False,num_beams=1,enable_thinking=False,max_new_tokens=1600,context_truncation=False,all_rows_fresh=True)
    oldraw=read(S/'evidence/inference-terminal.json'); assert oldraw['rows']==oldraw['EOS_rows']==39 and oldraw['cap_rows']==0
    metrics=R/'research/four-source-fusion-20261024/F1-v1/metrics.json'
    baseline=metrics.with_name('prediction.jsonl'); strong=read(metrics)['metrics'];assert strong['Score']==.701325377212172
    originalG=G/'quality-F1-metadata-recovery-v1/assembled/new-only.jsonl'
    b=[json.loads(line) for line in baseline.read_text().splitlines()]
    g=[json.loads(line) for line in originalG.read_text().splitlines()]
    # Reuse the frozen v2 association atomizer only. Never call evaluate_report or read gold.
    ep=args.main/'scripts/evaluate.py';sys.path.insert(0,str(ep.parent));spec=importlib.util.spec_from_file_location('v2_atomizer_only',ep);ev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ev;spec.loader.exec_module(ev)
    pairs=lambda rows:{(d['pmc_id'],pid,h) for d in rows for pid,hs in ev.associations(d).items() for h in hs}
    old,gpure=pairs(b),pairs(g);assert len(old)==524 and len(gpure)==282
    assert len(b)==len(g)==16 and sum(len(d['association']) for d in b)==39
    nref=read(S/'evidence/normalization-complete.json');nr=read(nref['manifest']['path'])
    assert ref(nref['manifest']['path'])==nref['manifest'] and nr['policy_sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
    review=read(S/'evidence/final-F1-review.json')['review'];assert ref(review['path'])==review
    refs.update({
        'formal_collection':ref(G/'formal-v1/collection.json'),'actual_model_inventory':ref(terminal/'output/run_manifest.json'),
        'target_manifest':ref(G/'targets-v2/manifest.json'),'native39_audit':ref(audit_path),
        'inference_plan':ref(S/'inference/infer-plan.json'),'inference_policy':ref(S/'inference/policy.json'),
        'original_raw_terminal':ref(S/'evidence/inference-terminal.json'),'original_raw':oldraw['responses'],
        'normalization_manifest':nref['manifest'],'normalization_policy':ref(S/'normalization/policy_v1.json'),
        'normalizer_ancestry':ref(S/'normalization/evidence/asset-inventory.json'),
        'prior_complete_review':review,'original_G_pure':ref(originalG),'current_B4':ref(baseline),'current_B4_metrics':ref(metrics),
        'v2_atomizer_source':ref(ep),'method_map':ref(args.main/'docs/research/method-coverage-map-20261024/MAP.md'),
        'cpu_contract':ref(args.output/'cpu-v1/result.json'),
        'PEFT_cat_source':ref(R/'research/qwen35-feasibility/independent-review/peft-lora-model.py'),
        'PEFT_layer_source':ref(R/'research/qwen35-feasibility/independent-review/peft-lora-layer.py'),
        'PEFT_exact_tag':ref(R/'research/qwen35-feasibility/independent-review/peft-tag-object.json')})
    actual=dict(status='READINESS_ONLY_NO_ACTUAL_WEIGHT_COMPOSITION_OR_MODEL_CALL',parents=parents,
        configs_exact=True,same_checkpoint_fingerprint=True,rank_before=16,alpha_before=32,scaling_before=2,
        proposed_rank=32,proposed_alpha=32,proposed_scaling=1,
        target_counts=coverage['target_counts'],modules=248,saved_A_B_keys_expected=496,
        no_DoRA_RsLoRA_bias_modules_to_save_or_nonuniform_rank_alpha=True,
        parameter_accounting=dict(actual_parent_text_base=8953803264,actual_parent_adapter=43278336,
            proposed_concat_adapter=86556672,proposed_loaded_text_model=9040359936,
            public_bundle_including_unused_vision_MTP=9653104368,
            conservative_full_bundle_plus_concat_adapter=9739661040,each_model_less_than10B=True,
            new_model_not_yet_instantiated=True),
        actual_parent_runtime=fp['runtime'],held=dict(documents=16,patients=39,rows=39,total_tokens=203588,max_tokens=10080,no_truncation=True),
        baseline=dict(score=strong['Score'],relations=524,entities=sum(len(d['entities']) for d in b),patients=39),
        potential_supply=dict(original_G_relations=len(gpure),G_outside_B4=len(gpure-old),
            interpretation='Only a ceiling on proposed consensus additions; no gold labels, quality estimate, or average-model prediction'),
        cost=dict(two_existing_adapter_bytes=2*173181072,proposed_FP32_factor_payload_bytes=86556672*4,
            serialized_output_estimate_bytes=350_000_000,extra_FP32_resident_adapter_bytes=43278336*4,
            CPU_synthesis_scratch_estimate_bytes=1_200_000_000,CPU_synthesis_observed=False,
            parent39_generation_seconds=oldraw['native_seconds'],parent39_peak_allocated_bytes=oldraw['max_gpu_allocated_bytes'],
            proposed_new_generation_seconds='Not measured; prior1103.2s is proxy, changed output length/rank can change cost',
            suggested_owner_budget_seconds=1500,suggested_grace_seconds=30,
            new_norm_query_count_unknown=True,old_needed_queries=254,model_soup_zero_overhead_claim_not_applied=True),
        readiness_limits=['No fresh remote existence/SHA poll; parent durable inventory proves prior existence, later consumption must revalidate both.',
            'No real PEFT load or actual adapter tensor-key/header read in this phase; complete tensor inventory and tiny CPU PEFT save/load remain engineering gates.',
            'Exact real-number delta identity does not imply BF16/GPU bitwise same results or averaged output logits.',
            'Original inference binder requires epoch2/r16 checkpoint and8997081600 parameters; future wrapper must explicitly use new synthesis provenance and9040359936, never pretend concat is an old trained epoch.'],
        refs=refs,no_gold_no_fit_no_GPU_no_quality=True,no_new_weights_downloaded=True)
    plan=dict(experiment='GWA-20261024-001',status='UNEXECUTED_CANDIDATE_PROTOCOL_GPU_NOT_AUTHORIZED',
        formula='B4 UNION (G_meanweights INTERSECT G_originalgreedy)',coefficient=.5,
        parent_epochs=[1,2],parent_optimizer_steps=[11,22],base_unchanged=True,
        effective_delta='0.5*(2*B1@A1+2*B2@A2)',adapter_representation='concat_A_axis0_concat_B_axis1_rank32_alpha32',
        checkpoint_selection=False,alpha_scan=False,SVD_or_factor_mean=False,fresh_training=False,
        complete_input_rows=39,generation=policy['generation'],raw_first=True,
        exact_quote_and_fixed_normalization=refs['normalization_policy'],
        relation_identity='public-v2 literal atom (document, patient, HP ID or exact NIL text); NO excluded',
        union_rule='Keep every B4 relation/list prefix and all nonassociation/patient objects. Add only literal intersections of both pure G views, without B-union contamination.',
        post_Generation_quality_if_later_authorized='Freeze exactly one complete primary prediction, then one newv2/old0; if byteexact no change inherit baseline with0v2.',
        no_alone_or_epoch_or_full_union_endpoints=True,
        gate='Strict Score>B4 .701325377212172; disclose micro and macro. Further F0 or full64 only by root decision.',
        sources={k:refs[k] for k in ['current_B4','current_B4_metrics','original_G_pure','inference_plan','normalization_policy']},
        GPU_authorized=False,quality_authorized=False,actual_adapter_synthesis=False,
        next_if_authorized=['Revalidate both actual parent adapterSHA/keys/config; no retraining.',
            'CPU compose in a unique namespace and preserve source tensor values; new config r32/alpha32; no optimizer state fabricated.',
            'Tiny ordinary CPU PEFT rank32 config/tensor save-load/selected linear delta contract using pinned0.21.2; full-backbone GPU separately bounded by owner.',
            'Reuse full39 native inputIDs, greedy1600/raw-first/no repair, exactquote and fixedtrain32/public normalizer.',
            'Assemble only the single primary fullF1 relation intersection endpoint; no threshold or patient scans.'])
    save(args.output/'readiness.json',actual);save(args.output/'candidate-plan.json',plan)
    save(args.output/'proposed-adapter-config.json',proposed)
    save(args.output/'expected-saved-key-inventory.json',dict(
        provenance=refs['actual_model_inventory'],
        derived_from_actual_loaded_trainable_names=True,actual_saved_tensor_header_read=False,
        keys=[name.replace('.default.weight','.weight') for name in coverage['all_trainable_names']]))
    print(json.dumps(dict(readiness=ref(args.output/'readiness.json'),plan=ref(args.output/'candidate-plan.json'),potential_additions=len(gpure-old),proposed_parameters=9040359936)))


if __name__=='__main__':main()
