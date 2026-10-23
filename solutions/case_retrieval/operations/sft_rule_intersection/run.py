"""One predeclared literal-v2 SFT/rules intersection; freeze before F0 gold."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atoms(value):
    assert isinstance(value, str)
    return value.split(';') if COMPOUND.fullmatch(value) else [value]


def allowed_intersection(values, allowed):
    result, seen = [], set()
    for value in values:
        parts = atoms(value)
        keep = [p for p in parts if p in allowed and p not in seen]
        if keep:
            # Preserve unchanged raw string when all its literal atoms survive.
            result.extend([value] if keep == parts else keep)
            seen.update(keep)
    return result


def contracts():
    assert allowed_intersection([], {'HP:0000001'}) == []
    assert allowed_intersection(['HP:0000001'], set()) == []
    assert allowed_intersection(['HP:0000001'], {'HP:0000001'}) == ['HP:0000001']
    assert allowed_intersection(['HP:0000001;HP:0000002'], {'HP:0000002'}) == ['HP:0000002']
    assert allowed_intersection(['HP:0000001;HP:0000002'], {'HP:0000001','HP:0000002'}) == ['HP:0000001;HP:0000002']
    assert allowed_intersection(['text;with;semicolons'], {'text'}) == []
    assert allowed_intersection(['metabolic diseases'], {'metabolic diseases'}) == ['metabolic diseases']
    assert allowed_intersection(['HP:0000001','HP:0000001'], {'HP:0000001'}) == ['HP:0000001']


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--contracts-only', action='store_true')
    args = parser.parse_args()
    contracts()
    if args.contracts_only:
        print(json.dumps({'contracts': 8, 'status': 'PASS'})); return
    plan = json.loads(Path(args.plan).read_text())
    assert plan['source_sha256'] == sha(__file__)
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    refs = plan['references']
    # Do not open the gold file until the complete output has been frozen.
    for name, ref in refs.items():
        if name != 'gold':
            assert sha(ref['path']) == ref['sha256'], name
    inputs = {str(d['pmc_id']): d for d in rows(refs['input']['path'])}
    sft = rows(refs['sft']['path'])
    rule = {str(d['pmc_id']): d for d in rows(refs['rules']['path'])}
    assert len(inputs) == len(sft) == len(rule) == 16
    assert set(inputs) == set(rule) == {str(d['pmc_id']) for d in sft}
    counts = dict(documents=0, patients=0, entities=0, sft_atoms=0, rules_atoms=0,
                  kept_atoms=0, removed_atoms=0, compound_sft_strings=0, raw_text_sft_strings=0)
    predictions = []
    for original in sft:
        did = str(original['pmc_id']); inp = inputs[did]; other = rule[did]
        assert not {'entities', 'association'} & set(inp)
        assert original['entities'] == other['entities']
        patient_ids = {a['patient_id'] for a in inp['patient']}
        rr = {a['patient_id']: {p for value in a['phenotype'] for p in atoms(value)} for a in other['association']}
        assert set(rr) == patient_ids and len(rr) == len(other['association'])
        assert {a['patient_id'] for a in original['association']} == patient_ids
        assert len(original['association']) == len(patient_ids)
        result = copy.deepcopy(original)
        for before, after in zip(original['association'], result['association']):
            pid = before['patient_id']; values = before['phenotype']
            ss = {p for value in values for p in atoms(value)}
            after['phenotype'] = allowed_intersection(values, rr[pid])
            kept = {p for value in after['phenotype'] for p in atoms(value)}
            assert kept == ss & rr[pid] and kept <= ss
            assert {k:v for k,v in before.items() if k != 'phenotype'} == {k:v for k,v in after.items() if k != 'phenotype'}
            counts['sft_atoms'] += len(ss); counts['rules_atoms'] += len(rr[pid])
            counts['kept_atoms'] += len(kept); counts['removed_atoms'] += len(ss - kept)
            counts['compound_sft_strings'] += sum(bool(COMPOUND.fullmatch(v)) for v in values)
            counts['raw_text_sft_strings'] += sum(not re.fullmatch(r'HP:[0-9]{7}',v) and not COMPOUND.fullmatch(v) for v in values)
        assert {k:v for k,v in original.items() if k != 'association'} == {k:v for k,v in result.items() if k != 'association'}
        counts['documents'] += 1; counts['patients'] += len(patient_ids); counts['entities'] += len(original['entities'])
        predictions.append(result)
    assert (counts['patients'], counts['entities'], counts['sft_atoms'], counts['rules_atoms'], counts['kept_atoms'], counts['removed_atoms']) == (47,947,276,345,261,15)
    pred_path = output/'intersection.pred.jsonl'
    pred_path.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in predictions))
    freeze = {'status':'complete_prediction_frozen_before_quality', 'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'prediction_sha256':sha(pred_path),
              'counts_without_gold':counts,'gold_opened':False,'no_new_training_or_gpu':True,'nonrelation_unchanged':True}
    dump(output/'prediction-freeze.json',freeze)
    # The only quality arm begins here. Cached comparators remain unchanged.
    assert sha(refs['gold']['path']) == refs['gold']['sha256']
    sys.path.insert(0,str(Path(refs['evaluator']['path']).parent))
    spec = importlib.util.spec_from_file_location('intersection_fixed_eval', refs['evaluator']['path'])
    ev = importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    from validate_predictions import validate_predictions, read_hpo
    validation = validate_predictions(list(inputs.values()), predictions, read_hpo(refs['hpo']['path']))
    assert validation['valid'] and not validation['errors'] and not validation['warnings']
    report = ev.evaluate_report(rows(refs['gold']['path']), predictions, 'all','text')
    report.update(gold_sha256=refs['gold']['sha256'], prediction_sha256=sha(pred_path))
    validation.update(input_sha256=refs['input']['sha256'], prediction_sha256=sha(pred_path), hpo_sha256=refs['hpo']['sha256'])
    old = json.loads(Path(refs['sft_metrics']['path']).read_text())
    strong = json.loads(Path(refs['rules_metrics']['path']).read_text())
    assert report['metrics']['menF1'] == old['metrics']['menF1'] and report['metrics']['docF1'] == old['metrics']['docF1']
    delta = {k:report['metrics'][k]-old['metrics'][k] for k in report['metrics']}
    elapsed = time.monotonic()-start
    assert elapsed <= plan['cpu_limit_seconds']
    dump(output/'metrics.json',report);dump(output/'validation.json',validation)
    result = {'status':'complete_single_fixed_F0_development_control','elapsed_seconds':elapsed,
              'plan_sha256':sha(args.plan),'source_sha256':sha(__file__),'freeze_sha256':sha(output/'prediction-freeze.json'),
              'prediction_sha256':sha(pred_path),'metrics':report['metrics'],'counts':report['counts'],
              'delta_vs_same_source_sft9b':delta,'strong_rules_metrics':strong['metrics'],
              'keep_by_predeclared_criterion':delta['micF1']+delta['macF1']>0 and delta['macF1']>=0,
              'invariants':counts,'format_errors':len(validation['errors']),'format_warnings':len(validation['warnings']),
              'limits':['F0 previously used for development; not a blind estimate.',
                        'Old Qwen3-8B intersection was already tested and below rules; this uses the distinct fixed9B SFT.',
                        'Only one new intersection; no threshold or subset selection, no original-dev evaluation or transfer.',
                        'Non-HPO text remains literal; no alt-ID normalization.'],
              'outputs':{n:{'path':str(output/n),'sha256':sha(output/n),'bytes':(output/n).stat().st_size} for n in ['prediction-freeze.json','intersection.pred.jsonl','metrics.json','validation.json']}}
    dump(output/'summary.json',result)
    print(json.dumps({'status':result['status'],'metrics':result['metrics'],'delta':delta,'keep':result['keep_by_predeclared_criterion'],'seconds':elapsed,'summary':str(output/'summary.json')}))


if __name__ == '__main__':
    main()
