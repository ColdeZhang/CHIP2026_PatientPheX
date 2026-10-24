"""One complete incremental review entry; never open gold or call an evaluator."""
import hashlib
import json
import os
from pathlib import Path
from actual import H

S = Path(__file__).resolve().parent
R = Path('/workspace/patientphex-runs/round2-20261023')
B = R / 'research/grounded-generation-full64-v1'


def ref(p):
    p = Path(p).resolve(); data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    guard = H.gold_guard({'gold': {'path': '/workspace/patientphex-data/day1/dev16.gold.jsonl'}})
    assert not guard['enabled']
    formal = B / 'formal-v1'
    campaign = H.read(formal / 'campaign-complete.json')
    assert campaign['status'] == 'TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
    collections = {}
    phases = ['native-cpu-v1', 'probe-v1', 'formal-v1', 'infer-dev36-v1']
    preparation = H.read(B / 'dev36-normalization-preparation/manifest.json')
    if preparation['necessary_public_query_count']:
        phases.append('normalization-dev36-v1')
    for phase in phases:
        path = B / phase / 'collection.json'; value = H.read(path)
        assert value['all_actual_SHA_verified']
        for item in value['files'].values(): assert ref(item['path']) == item
        assert ref(value['archive']['path']) == value['archive']
        collections[phase] = dict(collection=ref(path), original_archive=value['archive'],
                                 member_count=len(value['files']))
    assert H.read(formal / 'collection.json')['formal_reported_success']
    assert H.read(B / 'infer-dev36-v1/collection.json')['inference_reported_success']
    acks = {}
    for epoch in (1, 2):
        path = formal / ('epoch-%03d-export-ack.json' % epoch); value = H.read(path)
        assert value['status'] == 'persistent_checkpoint_verified'
        assert value['global_step'] == 22 * epoch and value['completed_epochs'] == epoch
        assert value['fsync_all_files'] and value['reopened_final_filename_SHA_all_files']
        assert value['destination_directories_fsynced']
        assert ref(path)['sha256'] == campaign['epoch%d_ack_sha256' % epoch]
        acks[str(epoch)] = ref(path)
    freeze_path = B / 'dev36-pure-relations-v1/freeze.json'; freeze = H.read(freeze_path)
    assert freeze['all36_complete'] and freeze['nonrelation_bytes_exact'] and freeze['no_gold_no_score']
    assert (freeze['documents'], freeze['patients'], freeze['entities']) == (16, 36, 1061)
    for key in ['pure_keys', 'consumption', 'validation']:
        assert ref(freeze[key]['path']) == freeze[key]
    assert ref(freeze['predictions']['new-only']['path']) == freeze['predictions']['new-only']
    generated = H.read(B / 'infer-dev36-v1/terminal/output/complete.json')
    normalized = H.read(B / 'dev36-normalized-v1/manifest.json')
    entry = dict(schema='grounded-full64-dev36-complete-review-entry-v1',
        source=ref(__file__), scope='Only new full64 engineering/formal/dev36 supply; originaldev quality belongs to root',
        source_checkpoint_contract=ref(S / 'source-contract.json'),
        formal_plan=ref(formal / 'formal-plan.json'), controller_plan=ref(formal / 'controller-plan.json'),
        original_campaign=ref(formal / 'campaign.json'), completed_campaign=ref(formal / 'campaign-complete.json'),
        complete_epoch_ACKs=acks, collections=collections,
        actual_full_checkpoint_locators=H.read(formal / 'collection.json')['actual_external_checkpoints'],
        inference_plan=ref(S / 'infer-plan.json'), full36_terminal=ref(B / 'infer-dev36-v1/terminal/output/complete.json'),
        extraction_plan=ref(S / 'extraction-plan.json'), extraction=ref(B / 'dev36-quotes-v1/manifest.json'),
        normalization_preparation=ref(B / 'dev36-normalization-preparation/manifest.json'),
        same_asset_host_migration=ref(S / 'held-target-assets.json'),
        normalization_plan=ref(S / 'held-finalize-plan.json'), normalization=ref(B / 'dev36-normalized-v1/manifest.json'),
        pure_plan=ref(S / 'pure-plan.json'), pure_freeze=ref(freeze_path),
        all_new_relation_prediction=freeze['predictions']['new-only'],
        all_normalized_quotes=normalized['normalizations'],
        actual_summary=dict(train_rows=173, epochs=2, optimizer_calls=44, warmup4=True,
                            held_rows=generated['rows'], EOS=generated['EOS_rows'], cap=generated['cap_rows'],
                            native_generation_seconds=generated['seconds'],
                            normalization_status_counts=normalized['status_counts'],
                            relation_count=freeze['relation_count']),
        preserved=dict(original901_nonrelation_bytes=True, Task1_entities=1061, all_original36_patients=True),
        prior_proofs=dict(
            original_grounded_engineering=ref(R / 'validation/grounded-engineering-formal-source-review/review.json'),
            original_normalization=ref(R / 'validation/schema-extraction-final-review/review.json'),
            train32_F1=ref(R / 'validation/grounded-F1-final-review/review.json'),
            train32_F0=ref(R / 'validation/grounded-qwen-F0-incremental-review/review.json')),
        current_originaldev_gold_opened=False, current_quality_calls=0, official_calls=0)
    target = B / 'dev36-final-review-entry.json'
    H.save(target, entry)
    print(json.dumps(ref(target)))


if __name__ == '__main__':
    main()
