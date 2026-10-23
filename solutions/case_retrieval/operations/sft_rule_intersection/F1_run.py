"""One predeclared F1 transfer: intersect old SFT, preserve all 65 public additions."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--contracts-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['source_sha256'] == sha(__file__)
    refs = plan['references']
    assert sha(refs['intersection_core']['path']) == refs['intersection_core']['sha256']
    core = load_module('frozen_intersection_core', refs['intersection_core']['path'])
    core.contracts()
    # Preserve unsupported-by-rules additions while filtering only old SFT atoms.
    assert core.allowed_intersection(['HP:0000001', 'HP:0000002'], ({'HP:0000001'} & set()) | {'HP:0000002'}) == ['HP:0000002']
    assert core.allowed_intersection(['HP:0000001;HP:0000002'], ({'HP:0000001'} & {'HP:0000001'}) | {'HP:0000002'}) == ['HP:0000001;HP:0000002']
    if args.contracts_only:
        print(json.dumps({'contracts': 10, 'status': 'PASS'})); return
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    # Every source is frozen first; gold may only be opened after full prediction freeze.
    for name, ref in refs.items():
        if name != 'gold':
            assert sha(ref['path']) == ref['sha256'], name
    inputs = {str(d['pmc_id']): d for d in rows(refs['input']['path'])}
    sft = {str(d['pmc_id']): d for d in rows(refs['sft']['path'])}
    rules = {str(d['pmc_id']): d for d in rows(refs['rules']['path'])}
    supported = rows(refs['supported']['path'])
    frozen_additions = {tuple(map(str, item)) for item in json.loads(Path(refs['supported65']['path']).read_text())}
    assert len(frozen_additions) == 65
    assert len(inputs) == len(sft) == len(rules) == len(supported) == 16
    assert set(inputs) == set(sft) == set(rules) == {str(d['pmc_id']) for d in supported}
    counts = dict(documents=0, patients=0, entities=0, sft_atoms=0, rules_atoms=0,
                  old_intersection=0, old_removed=0, supported_additions=0, supported_baseline=0, final_atoms=0)
    predictions, seen_additions = [], set()
    for original in supported:
        did = str(original['pmc_id']); inp = inputs[did]
        assert not {'entities', 'association'} & set(inp)
        assert {k:v for k,v in original.items() if k != 'association'} == {k:v for k,v in sft[did].items() if k != 'association'}
        assert original['entities'] == rules[did]['entities']
        patient_ids = {a['patient_id'] for a in inp['patient']}
        ss = {a['patient_id']: {p for value in a['phenotype'] for p in core.atoms(value)} for a in sft[did]['association']}
        rr = {a['patient_id']: {p for value in a['phenotype'] for p in core.atoms(value)} for a in rules[did]['association']}
        assert set(ss) == set(rr) == patient_ids
        assert len(ss) == len(sft[did]['association']) and len(rr) == len(rules[did]['association'])
        assert {a['patient_id'] for a in original['association']} == patient_ids
        assert len(original['association']) == len(patient_ids)
        result = copy.deepcopy(original)
        for before, after in zip(original['association'], result['association']):
            pid = before['patient_id']; values = before['phenotype']
            pp = {p for value in values for p in core.atoms(value)}
            assert ss[pid] <= pp
            additions = pp - ss[pid]
            seen_additions.update((did, str(pid), p) for p in additions)
            desired = (ss[pid] & rr[pid]) | additions
            after['phenotype'] = core.allowed_intersection(values, desired)
            kept = {p for value in after['phenotype'] for p in core.atoms(value)}
            assert kept == desired and kept <= pp and additions <= kept
            assert kept - ss[pid] == additions
            assert {k:v for k,v in before.items() if k != 'phenotype'} == {k:v for k,v in after.items() if k != 'phenotype'}
            counts['sft_atoms'] += len(ss[pid]); counts['rules_atoms'] += len(rr[pid])
            counts['old_intersection'] += len(ss[pid] & rr[pid]); counts['old_removed'] += len(ss[pid] - rr[pid])
            counts['supported_additions'] += len(additions); counts['supported_baseline'] += len(pp)
            counts['final_atoms'] += len(kept)
        assert {k:v for k,v in original.items() if k != 'association'} == {k:v for k,v in result.items() if k != 'association'}
        counts['documents'] += 1; counts['patients'] += len(patient_ids); counts['entities'] += len(original['entities'])
        predictions.append(result)
    assert seen_additions == frozen_additions
    assert counts == plan['expected_counts_without_gold']
    pred_path = output/'intersection-supported.pred.jsonl'
    pred_path.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in predictions))
    freeze = {'status':'complete_prediction_frozen_before_quality', 'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'prediction_sha256':sha(pred_path),
              'counts_without_gold':counts,'supported65_membership_exact':True,'gold_opened':False,
              'no_new_training_or_gpu':True,'nonrelation_and_association_metadata_unchanged':True}
    dump(output/'prediction-freeze.json',freeze)
    # Exactly one quality arm. Cached comparators are read without recomputing.
    assert sha(refs['gold']['path']) == refs['gold']['sha256']
    sys.path.insert(0, str(Path(refs['evaluator']['path']).parent))
    ev = load_module('f1_intersection_fixed_eval', refs['evaluator']['path'])
    from validate_predictions import validate_predictions, read_hpo
    validation = validate_predictions(list(inputs.values()), predictions, read_hpo(refs['hpo']['path']))
    assert validation['valid'] and not validation['errors'] and not validation['warnings']
    report = ev.evaluate_report(rows(refs['gold']['path']), predictions, 'all','text')
    report.update(gold_sha256=refs['gold']['sha256'], prediction_sha256=sha(pred_path))
    validation.update(input_sha256=refs['input']['sha256'], prediction_sha256=sha(pred_path), hpo_sha256=refs['hpo']['sha256'])
    old = json.loads(Path(refs['supported_metrics']['path']).read_text())
    sft_metrics = json.loads(Path(refs['sft_metrics']['path']).read_text())
    assert report['metrics']['menF1'] == old['metrics']['menF1'] and report['metrics']['docF1'] == old['metrics']['docF1']
    delta = {k:report['metrics'][k]-old['metrics'][k] for k in report['metrics']}
    elapsed = time.monotonic()-started
    assert elapsed <= plan['cpu_limit_seconds']
    dump(output/'metrics.json',report); dump(output/'validation.json',validation)
    result = {'status':'complete_single_fixed_F1_development_transfer','elapsed_seconds':elapsed,
              'plan_sha256':sha(args.plan),'source_sha256':sha(__file__),'freeze_sha256':sha(output/'prediction-freeze.json'),
              'prediction_sha256':sha(pred_path),'metrics':report['metrics'],'counts':report['counts'],
              'delta_vs_same_supported65_baseline':delta,'supported65_baseline_metrics':old['metrics'],
              'context_only_old_sft_metrics':sft_metrics['metrics'],
              'keep_by_predeclared_criterion':delta['micF1']+delta['macF1']>0 and delta['macF1']>=0,
              'invariants':counts,'supported65_membership_exact':True,
              'format_errors':len(validation['errors']),'format_warnings':len(validation['warnings']),
              'limits':plan['limitations'],
              'outputs':{n:{'path':str(output/n),'sha256':sha(output/n),'bytes':(output/n).stat().st_size} for n in ['prediction-freeze.json','intersection-supported.pred.jsonl','metrics.json','validation.json']}}
    dump(output/'summary.json',result)
    print(json.dumps({'status':result['status'],'metrics':result['metrics'],'delta':delta,'keep':result['keep_by_predeclared_criterion'],'seconds':elapsed,'summary':str(output/'summary.json')}))


if __name__ == '__main__':
    main()
