"""Recover only metadata from complete immutable probe observations; no model import."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from probe import write, sha
from metadata import native_summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--collection',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    collection=json.loads(Path(args.collection).read_text())
    for n,ref in collection['members'].items():
        p=root/n;assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256']
    failure=json.loads((root/'actual/failure.json').read_text());ended=json.loads((root/'ended.json').read_text());progress=json.loads((root/'actual/progress.json').read_text())
    assert failure['type']=='ValueError' and 'JSON compliant: inf' in failure['message'] and failure['completed_steps']==6
    assert ended['child_exit_code']==1
    assert 'GPU-126293b9-6677-8464-cee7-01e693efbf29, 0' in ended['gpu'] and 'GPU-126293b9-6677-8464-cee7-01e693efbf29' not in ended['apps']
    partial=json.loads((root/'actual/complete.json.pending').read_text()+'null}}')
    assert partial['summary']['total_steps']==6 and partial['summary']['total_epochs']==2
    native_seconds=partial['summary']['total_time_seconds']
    # The original traceback and partial prefix identify this exact no-eval sentinel.
    recovered_native=native_summary(dict(partial['summary'],best_metric=math.inf,eval_metrics_history=[]))
    assert recovered_native['best_metric'] is None and recovered_native['best_metric_status']=='not_computed_no_eval'
    selected=json.loads((root/'actual/selected.json').read_text())['indices'];steps=[];batches=[]
    for i in range(6):
        s=json.loads((root/f'actual/step-{i:03d}-complete.json').read_text());raw=json.loads((root/f'actual/batch-{i:03d}-actual.json').read_text());assert s==progress['steps'][i]
        assert s['applied'] and s['finite_loss'] and s['finite_gradients'] and s['gradient_norm']>0 and all(x['finite'] for x in s['raw_loss_terms'])
        assert all(math.isfinite(s[x]) for x in ['loss','gradient_norm','forward_backward_seconds','optimizer_seconds'])
        assert len(s['parameter_changes'])==2 and all(x['changed'] for x in s['parameter_changes'])
        for x in raw['rows']:
            assert x['actual_input_ids']==x['independent_tokenizer_json_ids'];assert sorted(x['actual_positive_pairs'])==sorted(x['expected_positive_pairs']);assert x['query_mask']==[True]
        if s['epoch']==2:assert s['optimizer_states_before']==300
        steps.append(s);batches.append(raw)
    for epoch in (1,2):assert Counter(i for s in steps if s['epoch']==epoch for i in s['row_ids'])==Counter(selected)
    final=json.loads((root/'actual/final-discarded.json').read_text());assert final=={'probe_weights_discarded':True,'global_step':6,'epoch':1}
    core=[s['pair_loss_calls'][0] for s in steps];durations=[s['forward_backward_seconds']+s['optimizer_seconds'] for s in steps]
    result={'status':'NATIVE_PROBE_NUMERICAL_AND_UPDATE_CONTRACT_PASS_METADATA_RECOVERED_NO_RERUN','original_child_exit_code':1,'original_failure':'No-eval best_metric=inf sentinel rejected while writing final JSON after6 completed updates. Original failure/partial JSON unchanged.','best_metric':None,'best_metric_status':'not_computed_no_eval','epochs':2,'optimizer_steps':6,'selected_rows':selected,'all_rows_once_each_epoch':True,'second_epoch_Adam_states_before':300,'finite_raw_loss_terms':sum(len(s['raw_loss_terms']) for s in steps),'all_losses_gradients_finite':True,'encoder_and_boundary_parameters_changed_each_step':True,'raw_schema_ID_target_contract':True,'native_query_augmentation_probability':.2,'actual_max_wordpieces':max(n for s in steps for n in s['actual_input_lengths']),'max_gold_per_query':49,'peak_allocated_bytes':max(s['peak_allocated_bytes'] for s in steps),'peak_reserved_bytes':max(s['peak_reserved_bytes'] for s in steps),'step_compute_seconds':durations,'post_first_max_compute_seconds':max(durations[1:]),'step_mean_compute_seconds':statistics.mean(durations),'native_training_total_seconds_from_partial_summary':native_seconds,'wrapper_total_seconds':ended['elapsed_seconds'],'released_utc':ended['ended_utc'],'core_pair_effective_positive':[x['effective_positive'] for x in core],'core_pair_effective_negative':[x['effective_negative'] for x in core],'NO_annotated_window_instances':sum(y['annotated'] for x in core for y in x['NO_coverage']),'NO_effective_pair_negative_instances':sum(y['effective_negative'] for x in core for y in x['NO_coverage']),'NO_limit':'Most annotated NO spans were not proposed; native sparse pair loss does not guarantee every NO target. Marginal/inside losses still use original native negative context.','weights_discarded':True,'fresh_formal_required':True,'quality_evaluations':0,'suggested_formal_budget_seconds':1200,'formal_budget_basis':'1600 updates × observed post-first maximum .421468s =674.35s; native whole-loop average .434075s gives694.52s. Add loading/save/metadata and margin. Six shapes do not guarantee full-run cost; hard bound proposed1200+30.','collection_sha256':sha(args.collection),'recovery_source_sha256':sha(__file__)}
    write(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
