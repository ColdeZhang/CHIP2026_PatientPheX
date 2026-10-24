"""The two predeclared complete predictions freeze before two new v2 reports."""
import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
from common import checked, ref, save, sha, stamp
from literal_helpers import assemble_two, nonrelation_bytes


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    output = Path(plan['output'])
    output.mkdir(exist_ok=False)
    save(output / 'started.json', {'at_utc': stamp(), 'plan': ref(args.plan), 'source': ref(__file__)})
    for name, entry in plan['refs'].items():
        if name != 'gold':
            checked(entry)
    inputs = jsonl(plan['refs']['input']['path'])
    old_lines = Path(plan['refs']['baseline']['path']).read_text().splitlines(keepends=True)
    base = [json.loads(line) for line in old_lines]
    prompts = jsonl(plan['refs']['prompts']['path'])
    root = Path(plan['collected'])
    driver = json.loads((root / 'run/driver-result.json').read_text())
    assert driver['status'] == 'COMPLETE_ALL_THREE_PHASES'
    formal = json.loads((root / 'run/formal/complete.json').read_text())
    infer = json.loads((root / 'run/infer/complete.json').read_text())
    assert formal['global_step'] == 32 and formal['counts'] == {'rows': 248, 'tokens': 806302, 'supervised': 14340}
    assert len(formal['acknowledgements']) == 2
    assert infer['fresh_rows'] == 80 and infer['cached_rows'] == 0
    canonical = jsonl(root / 'run/infer/canonical.jsonl')
    raw = jsonl(root / 'run/infer/responses.jsonl')
    assert len(prompts) == len(canonical) == len(raw) == 80
    assert len({row['row_id'] for row in prompts}) == 80
    assert len(inputs) == len(base) == 16 and sum(len(row['entities']) for row in base) == 1189
    assert [str(row['pmc_id']) for row in inputs] == [str(row['pmc_id']) for row in base]
    patient_ids = {str(row['pmc_id']): {p['patient_id'] for p in row['patient']} for row in inputs}
    assert sum(map(len, patient_ids.values())) == 39
    selected = defaultdict(lambda: defaultdict(set))
    seen = set()
    complete = 0
    for index, (prompt, record, response) in enumerate(zip(prompts, canonical, raw)):
        assert prompt['row_id'] == record['row_id'] == response['row_id']
        assert str(prompt['pmc_id']) == str(record['pmc_id']) == str(response['pmc_id'])
        pmc = str(prompt['pmc_id'])
        assert set(prompt['patient_ids']) == patient_ids[pmc]
        assert record['original_index'] == response['original_index'] == index
        assert record['provenance'] == response['provenance'] == 'fresh'
        tokens = json.loads((root / f'run/infer/generated-ids/{index:03d}.json').read_text())
        assert tokens['row_id'] == prompt['row_id']
        assert response['output_tokens'] == len(tokens['generated_ids'])
        assert record['natural_eos'] == response['natural_eos'] == tokens['natural_eos']
        assert record['hit_token_limit'] == response['hit_token_limit'] == tokens['hit_token_limit']
        assert list(prompt['code_map']) == [f'C{i:03d}' for i in range(len(prompt['code_map']))]
        for concept in prompt['code_map'].values():
            assert (pmc, concept) not in seen
            seen.add((pmc, concept))
        if not record['strict_complete']:
            assert record['candidate_patients'] == {} and record['abstention']
            continue
        assert record['natural_eos'] and not record['hit_token_limit'] and record['abstention'] is None
        mapping = record['candidate_patients']
        assert set(mapping) == set(prompt['code_map'])
        complete += 1
        for cid, patients in mapping.items():
            assert isinstance(patients, list) and len(patients) == len(set(patients))
            assert all(type(pid) is str and pid in patient_ids[pmc] for pid in patients)
            for patient in patients:
                selected[pmc][patient].add(prompt['code_map'][cid])
    assert len(seen) == 587 and complete == infer['strict_complete_rows']
    new_docs = []
    for original in base:
        doc = deepcopy(original)
        pmc = str(doc['pmc_id'])
        assert {row['patient_id'] for row in doc['association']} == patient_ids[pmc]
        for patient in doc['association']:
            patient['phenotype'] = sorted(selected[pmc][patient['patient_id']])
        new_docs.append(doc)
    single, union, relation_counts = assemble_two(old_lines, new_docs)
    predictions = {}
    for arm, lines in [('single', single), ('fullsupport_union', union)]:
        assert nonrelation_bytes(lines) == nonrelation_bytes(old_lines)
        path = output / (arm + '.prediction.jsonl')
        with path.open('x') as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        predictions[arm] = ref(path)
    freeze = {'at_utc': stamp(), 'predictions': predictions, 'relation_counts': relation_counts,
              'complete_documents': 16, 'patients': 39, 'same_entities': 1189,
              'strict_complete_rows': complete, 'whole_row_abstentions': 80 - complete,
              'nonassociation_bytes_exact': True, 'gold_reads_before_freeze': 0, 'v2_calls_before_freeze': 0,
              'raw_response_policy': 'Original strict fulltext complete-CID/PID; invalid entire rows abstain, no repair or fallback'}
    save(output / 'complete-freeze.json', freeze)
    validator = module(plan['refs']['validator']['path'], 'validate_predictions')
    evaluator = module(plan['refs']['evaluator']['path'], 'qwen3_fixed_evaluator')
    gold = evaluator.load_jsonl(checked(plan['refs']['gold']))
    hpo = validator.read_hpo(plan['refs']['hpo']['path'])
    baseline = json.loads(Path(plan['refs']['baseline_metrics']['path']).read_text())['metrics']
    assert baseline['Score'] == .6863389812716092
    save(output / 'evaluation-started.json', {'at_utc': stamp(), 'freeze': ref(output / 'complete-freeze.json'), 'new_calls': 2, 'old_calls': 0})
    results = {}
    for arm, entry in predictions.items():
        rows = jsonl(entry['path'])
        valid = validator.validate_predictions(inputs, rows, hpo)
        save(output / (arm + '.validation.json'), valid)
        assert valid['valid'] and valid['warning_count'] == 0
        report = evaluator.evaluate_report(gold, rows, 'all', 'text')
        save(output / (arm + '.metrics.json'), report)
        results[arm] = {'prediction': entry, 'metrics': report['metrics'], 'counts': report['counts'],
                        'delta': {key: report['metrics'][key] - baseline[key] for key in baseline}}
    improving = [key for key in results if results[key]['metrics']['Score'] > baseline['Score']]
    save(output / 'result.json', {'status': 'COMPLETE_FIXED_NATIVE_QWEN3_TWO_ENDPOINT_CONTROL',
         'baseline': baseline, 'results': results, 'new_v2_calls': 2, 'old_v2_calls': 0,
         'keep_for_F0': bool(improving),
         'selected': max(improving, key=lambda key: results[key]['metrics']['Score']) if improving else None,
         'scope': 'Known training-internal development F1, not blind test or originaldev. Both upstream training and candidate sources exclude allF1. No new training from F1 labels.',
         'freeze': ref(output / 'complete-freeze.json')})


if __name__ == '__main__':
    main()
