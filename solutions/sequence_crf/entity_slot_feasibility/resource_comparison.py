#!/usr/bin/env python3
"""Read-only same-F1 resource and score coverage; no labels, fitting, or scan."""
from pathlib import Path
import argparse, gzip, hashlib, json, time


def load(p):return json.loads(Path(p).read_text())
def rows(p):
    p=Path(p);b=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return [json.loads(s) for s in b.splitlines()]
def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def key(p,c):return str(p),c['offset'],c['length'],c['identifier']
def entities(ds):return {key(d['pmc_id'],e) for d in ds for e in d['entities']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();tick=time.monotonic()
    route=a.root/'sequence_crf';b=route/'entity-slot-formal32-v1';g=route/'entity-slot-feasibility';old=route/'public-bioformer-selector-control/fold1/fitted'
    refs=dict(freeze=g/'input-only-freeze.json',gate_census=g/'gate-p01/census.json',gate_scores=g/'gate-p01/heldout.gate-scores.jsonl.gz',
      original_F1_manifest=old/'manifest.json',original_F1_scores=old/'selector-scores.jsonl',
      final_probabilities=b/'terminal-infer/inference/final.probabilities.jsonl',final_pred=b/'assembled/final.pred.jsonl',
      legacy_HGB_pred=b/'legacy-hgb-baseline/pred.jsonl',legacy_HGB_metrics=b/'legacy-hgb-baseline/evaluation/metrics.json',
      final_metrics=b/'evaluation/final.metrics.json',training_complete=b/'terminal-train/training_complete.json')
    f=load(refs['freeze']);gate=load(refs['gate_census']);oldmeta=load(refs['original_F1_manifest'])
    train=set(f['train_ids']);held=set(f['heldout_ids']);assert len(train)==32 and len(held)==16 and not train&held
    oldref=load(oldmeta['arguments']['reference']);oldtrain={str(d['pmc_id']) for d in rows(oldref['config']['train'])}
    assert len(oldtrain)==48 and not oldtrain&held and train<=oldtrain
    scores=rows(refs['gate_scores']);pool={key(x['pmc_id'],x):x['gate_probability'] for x in scores}
    selected={k for k,p in pool.items() if p>=.1}
    final={}
    for r in rows(refs['final_probabilities']):
        for k,v in zip(r['keys'],r['probabilities']):
            kk=(str(r['pmc_id']),*k);assert kk not in final;final[kk]=v
    assert set(final)==selected
    public={key(d['pmc_id'],c) for d in rows(refs['original_F1_scores']) for c in d['candidates']}
    hgb_entities=entities(rows(refs['legacy_HGB_pred']));new_entities=entities(rows(refs['final_pred']))
    assert new_entities<=selected and hgb_entities<=selected
    metrics={n:load(refs[n]) for n in ['legacy_HGB_metrics','final_metrics']}
    diag=gate['training_diagnostic']
    output=dict(source=ref(__file__),references={k:ref(v) for k,v in refs.items()},
      strict_train32=dict(train_documents=len(train),heldout_documents=len(held),intersection=0,
        lower_generators=[dict(kind=x['kind'],train=len(x['generator_ids']),target=len(x['target_ids']),F1_overlap=len(set(x['generator_ids'])&held)) for x in f['origins']]),
      current_public_system=dict(training_documents=oldmeta['training_documents'],feature_count=len(oldmeta['features']),
        same_F1_excluded=True,extra_training_documents=len(oldtrain-train),same_train32_control=False,
        no_same32_public_model_found_in_known_manifests=True,owner_ontology_confirmed_none=True,
        warning='Its F1 OOF is valid for its own train48 task, but is not a same-resource train32 control.'),
      coverage_without_gold=dict(entire_77_pool=len(pool),gated_77_pool=len(selected),entity_score_keys=len(final),
        entity_key_coverage_of_gated_pool=len(set(final)&selected)/len(selected),
        original_public48_pool=len(public),shared_full_pool_keys=len(set(pool)&public),shared_gated_keys=len(selected&public),
        HGB_point4_before_greedy=sum(p>=.4 for p in pool.values()),entity_point5_before_greedy=sum(p>=.5 for p in final.values()),
        HGB_selected=len(hgb_entities),entity_selected=len(new_entities),both_selected=len(hgb_entities&new_entities),
        HGB_only_selected=len(hgb_entities-new_entities),entity_only_selected=len(new_entities-hgb_entities)),
      training_side=dict(gate_diagnostic=diag,raw_positive_fraction=diag['all_positive']/diag['all_candidates'],
        gated_positive_fraction=diag['retained_positive']/diag['retained_candidates'],
        epoch_mean_group_loss=[load(b/'terminal-train'/f'epoch-{i:03d}-summary.json')['mean_group_loss'] for i in [1,2]],
        interpretation='Resubstitution enriches easy training positives; shift is observed, causal bias/underfitting is not established.'),
      cached_quality_counts={k:v['counts'] for k,v in metrics.items()},
      conclusions=['Matching77 HGB .4 is newly evaluated; direct final gain is small and mention precision driven, document TP unchanged.',
        'All1589 gated candidates have context scores; missing current-public48 keys are a resource mismatch, not silently unscored in this control.',
        'No candidate recall ceiling is inferred from unlabelled key coverage. Current aggregate FN cannot identify gating versus decoder versus ID causes.',
        'Decreasing training loss does not establish either underfitting or generalization; no probability calibration scan was run.',
        'Next smallest distinct control, if authorized: fixed equal-mean HGB/context probability at .5 on these same1589 keys; old relations/greedy fixed. Heuristic probability scales, one development arm only, no new neural work.'],
      no_gold_read=True,no_new_quality=True,no_fit=True,no_GPU=True,no_cut_scan=True,seconds=time.monotonic()-tick)
    with a.output.open('x') as s:json.dump(output,s,ensure_ascii=False,indent=2);s.write('\n')
    print(json.dumps({k:output[k] for k in ['strict_train32','current_public_system','coverage_without_gold','training_side','seconds']},ensure_ascii=False))

if __name__=='__main__':main()
