"""Fixed N-full union R-full and R-full alone; jointly freeze before two v2 calls."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import os
import sys
import time
from common import ENDPOINTS, GOLD_SHA, QUALITY_COMPLETE, checked, fsync_directory, module, read, ref, reference_gate, rows, save, sha
from inference_gate import verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();plan_ref=ref(args.plan);plan=read(args.plan)
    records=reference_gate(plan,__file__,True)
    contract=read(plan['quality_contract']['path'])
    expected=contract['baseline_metrics']
    assert expected['Score']==.676546354662594 and contract['quality']['new_v2_calls']==2
    assert contract['quality']['old_v2_calls']==0 and contract['quality']['no_new_originaldev_F0_or_official']
    common=plan['references'];baseline=read(common['normal_metrics']['path'])
    assert common['normal_pred']==contract['normal_pred'] and common['normal_metrics']==contract['normal_metrics']
    assert baseline['metrics']==expected
    w=module('frozen_normal_fulltext_canonical_helpers',checked(plan['legacy_helpers']))
    prompts,raw,canonical,coverage,actual_files=verify(plan)
    # Reuse only pure canonical validation and literal assembly, never old main,
    # verify_inference, reference_gate, retain, or any model-response parser.
    evidence=[w.canonical_gate(p,raw[p['row_id']],canonical[p['row_id']]) for p in prompts]
    inputs=rows(common['input']['path'])
    normal_lines=Path(common['normal_pred']['path']).read_bytes().decode().splitlines(keepends=True)
    reverse_documents,parse_stats=w.make_primary(normal_lines,inputs,prompts,evidence)
    assert len(reverse_documents)==16 and sum(len(d['entities']) for d in reverse_documents)==1189
    assert sum(len(d['association']) for d in reverse_documents)==39
    assert parse_stats.get('provided_CIDs',0)+parse_stats.get('abstained_CIDs',0)==587
    reverse_lines,union_lines,counts=w.assemble_two(normal_lines,reverse_documents)
    counts['normal_fulltext']=counts.pop('old');counts['reverse_fulltext']=counts.pop('med')
    assert w.nonrelation_bytes(normal_lines)==w.nonrelation_bytes(reverse_lines)==w.nonrelation_bytes(union_lines)
    directory=Path(plan['output_dir']);directory.mkdir(parents=True,exist_ok=False);fsync_directory(directory.parent)
    frozen={}
    for name,lines in zip(ENDPOINTS,(union_lines,reverse_lines)):
        target=directory/name;target.mkdir()
        with (target/'pred.jsonl').open('xb') as handle:
            handle.write(''.join(lines).encode());handle.flush();os.fsync(handle.fileno())
        fsync_directory(target);frozen[name]=ref(target/'pred.jsonl')
    for record in records+list(frozen.values()):checked(record)
    assert ref(args.plan)==plan_ref
    freeze_path=directory/'paired-predictions-freeze.json'
    save(freeze_path,dict(status='BOTH_COMPLETE_FIXED_PREDICTIONS_BEFORE_FIRST_GOLD_BYTE',
        at_utc=datetime.now(timezone.utc).isoformat(),plan=plan_ref,endpoints=frozen,coverage=coverage,
        parse=parse_stats,literal_relation_counts=counts,documents=16,patients=39,entities=1189,
        nonassociation_bytes_exact=True,actual_source_files=actual_files,gold_read=False,quality_calls=0))
    fsync_directory(directory);freeze=ref(freeze_path)
    # Independent exclusive marker also prevents rerun under another quality output.
    terminal_root=Path(plan['endpoint_references']['terminal']['path']).parent
    save(terminal_root/'evaluation-started-fulltext-reverse-v1.json',dict(freeze=freeze,new_v2_calls=2,old_v2_calls=0))
    save(directory/'evaluation-started.json',dict(freeze=freeze,new_v2_calls=2,old_v2_calls=0))
    # H50: canonical validator registration and first gold bytes follow joint freeze.
    if 'validate_predictions' in sys.modules:
        assert sha(sys.modules['validate_predictions'].__file__)==common['validator']['sha256']
    validator=module('validate_predictions',checked(common['validator']))
    evaluator=module('fixed_fulltext_reverse_public_v2',checked(common['evaluator']))
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    hpo=validator.read_hpo(common['hpo']['path']);predictions={}
    for name in ENDPOINTS:
        predictions[name]=rows(checked(frozen[name]))
        validation=validator.validate_predictions(inputs,predictions[name],hpo)
        save(directory/name/'validation.json',validation);assert validation['valid']
    assert sha(plan['gold_path'])==GOLD_SHA
    gold=rows(plan['gold_path'])
    assert len(gold)==16 and [str(d['pmc_id']) for d in gold]==[str(d['pmc_id']) for d in inputs]
    endpoints={}
    for name in ENDPOINTS:
        checked(frozen[name]);metric=evaluator.evaluate_report(gold,predictions[name],'all','text')
        assert metric['documents']==16 and metric['patients']==39
        assert all(metric['metrics'][k]==expected[k] for k in ('menF1','docF1'))
        assert all(math.isfinite(v) for v in metric['metrics'].values())
        save(directory/name/'metrics.json',metric)
        endpoints[name]=dict(metrics=metric['metrics'],counts=metric['counts'],prediction=frozen[name],
            metrics_file=ref(directory/name/'metrics.json'),validation_file=ref(directory/name/'validation.json'),
            delta_vs_normal_fulltext=w.delta(metric['metrics'],expected),
            keep=metric['metrics']['Score']>expected['Score'])
    passing=[name for name in ENDPOINTS if endpoints[name]['keep']]
    # Stable ENDPOINTS order selects primary on an exact Score tie.
    selected=max(passing,key=lambda name:endpoints[name]['metrics']['Score']) if passing else None
    for record in records+list(frozen.values()):checked(record)
    checked(freeze);assert ref(args.plan)==plan_ref
    summary=dict(status=QUALITY_COMPLETE,endpoints=endpoints,selected=selected,keep=bool(passing),
        comparator=expected,normal_cached_metrics=common['normal_metrics'],paired_freeze=freeze,
        retention='Either endpoint Score strictly above N; all components reported, no macro hard gate; exact tie primary. Identities fixed.',
        new_v2_calls=2,old_v2_calls=0,no_F0_originaldev_official=True,
        known_F1_development_not_blind=True,hardware_disclosure=contract['hardware_disclosure'],
        elapsed_wall_seconds=time.monotonic()-started)
    save(directory/'summary.json',summary);fsync_directory(directory)
    print(json.dumps(dict(summary=ref(directory/'summary.json'),keep=summary['keep'],selected=selected)))


if __name__=='__main__':main()
