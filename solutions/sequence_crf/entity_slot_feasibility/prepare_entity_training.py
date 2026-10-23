#!/usr/bin/env python3
"""Freeze fresh entity32 training, paired inference and fixed assembly contracts."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--feasibility',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.feasibility;b=a.output;assert not b.exists();b.mkdir()
    old=json.loads((root/'probe-v1/plan.json').read_text());origin=json.loads((root/'input-only-freeze.json').read_text())
    held=next(x for x in origin['origins'] if x['kind']=='heldout')['source_paths'];source=Path(__file__).parent
    for name in ['train_entities.py','infer_entities.py','microbatch_slots.py']:
        shutil.copy2(source/name,b/name)
    for name in ['slot_model.py','qwen35_runner.py','qwen35_relation_cost.py','qwen35_smoke.py','model-source-manifest.json','train.json.gz','heldout.json.gz']:
        shutil.copy2(root/'probe-v1'/name,b/name)
    train=json.loads(gzip.decompress((b/'train.json.gz').read_bytes()));test=json.loads(gzip.decompress((b/'heldout.json.gz').read_bytes()))
    assert len(train)==637 and len(test)==398
    trainids={x['pmc_id'] for x in train};testids={x['pmc_id'] for x in test}
    assert len(trainids)==32 and len(testids)==16 and not trainids&testids
    assert all('targets' not in x for x in test)
    plan={k:old[k] for k in ['mount','gpu_uuid','python','model','env','gate','parameter_budget']}
    recipe=old['planned_recipe_not_authorized_for_formal'].copy()
    recipe['reference_batch_size']=recipe.pop('batch_size')
    recipe.update(effective_group_rows=8,microbatch_padded_token_cap=8192,epochs=2,updates=160,warmup_steps=8,
        original_order='seed20261023+epoch shuffle, consecutive8 groups; no intra-group sorting',
        checkpoint_policy='epoch1 recovery only; only epoch2 is final quality; partial never inference',
        actual_head='frozen existing [15,16] via F.linear, no new parameters',
        engineering_difference='same objective/order; microbatch changes dropout draws/floating point reduction vs batch1')
    plan.update(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        remote_directory='sequence-round2/entity-slot-formal32-v1',
        trainer_sha256=sha(b/'train_entities.py'),inference_sha256=sha(b/'infer_entities.py'),
        local_files={p.name:sha(p) for p in b.iterdir()},recipe=recipe,
        budgets=dict(training_seconds=2200,training_TERM_KILL_grace=30,checkpoint_margin=180,next_group_margin=40,
            two_arm_inference_seconds=600,inference_TERM_KILL_grace=30,minimum_free_bytes=4*1024**3),
        source_checkpoint='to be bound by actual launch after commit/push',
        references=dict(gate=ref(root/'gate-p01/census.json'),isolation=ref(root/'isolation-check.json'),
            input_only=ref(root/'input-only-freeze.json'),microbatch_probe=ref(root/'microbatch-probe-v1/terminal/complete.json'),
            actual_cost_proxy=ref(root/'microbatch-probe-v1/post-probe-summary.json'),
            data_review=ref(root.parent.parent/'validation/entity-slot-probe-review/data-review.json'),
            microbatch_review=ref(root.parent.parent/'validation/entity-slot-microbatch-review/review.json')),
        assembly=dict(source_sha256=sha(source/'assemble_entities.py'),cut=.5,
            references=dict(input=held['input'],baseline=held['base'],heldout_prompts=ref(root/'gate-p01/heldout.prompts.jsonl.gz')),
            policy='old calibrate (-prob,-length,offset) greedy, stable frozen gated prompt candidate order tie; any overlap rejected',
            entities='existing input candidate HPO/compound/-1 unchanged, type Phenotype/note None; no new keys or special filters',
            nonentities='exact baseline7ef8 objects unchanged for both arms; no patient reassignment',
            comparison='public9B adapter-disabled vs sole fresh final; same1589 gate77 candidates. Not current public97 HGB pipeline.',
            quality='both complete probabilities and both fullpred frozen first, then one F1 Task1 evaluation; no extra cut/epoch scan/originaldev'),
        training_ids=sorted(trainids),heldout_ids=sorted(testids),
        fresh_public_base=True,no_probe_or_old_adapter_reuse=True,heldout_targets=False,
        model_source_and_target_task='entity exactspan+ID validity, not patient association')
    (b/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(plan=ref(b/'plan.json'),files=len(plan['local_files']),train_rows=len(train),heldout_rows=len(test))))

if __name__=='__main__':main()
