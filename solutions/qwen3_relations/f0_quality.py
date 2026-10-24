"""One frozen B0-protected union quality call; raw new arm is never scored."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
from common import checked, ref, save, sha, stamp
from literal_helpers import assemble_two, nonrelation_bytes
from quality import jsonl, module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    save(out / 'started.json', {'at_utc': stamp(), 'source': ref(__file__), 'plan': ref(args.plan)})
    for name, entry in plan['refs'].items():
        if name != 'gold':
            checked(entry)
    inputs, prompts = jsonl(plan['refs']['input']['path']), jsonl(plan['refs']['prompts']['path'])
    lines = Path(plan['refs']['baseline']['path']).read_text().splitlines(keepends=True)
    base = [json.loads(x) for x in lines]
    root = Path(plan['collected'])
    assert json.loads((root / 'run/driver-result.json').read_text())['status'] == 'COMPLETE_FIXED_F0'
    terminal = json.loads((root / 'run/native/complete.json').read_text())
    assert terminal['rows'] == 58 and terminal['cached'] == terminal['training_steps'] == terminal['quality_calls'] == 0
    raw, canonical = jsonl(root / 'run/native/responses.jsonl'), jsonl(root / 'run/native/canonical.jsonl')
    assert len(raw) == len(canonical) == len(prompts) == 58
    assert len(inputs) == len(base) == 16 and sum(len(x['entities']) for x in base) == 947
    assert [str(x['pmc_id']) for x in inputs] == [str(x['pmc_id']) for x in base]
    patient_ids = {str(x['pmc_id']): {p['patient_id'] for p in x['patient']} for x in inputs}
    assert sum(map(len, patient_ids.values())) == 47
    concepts, complete, selected = set(), 0, defaultdict(lambda: defaultdict(set))
    for index, (prompt, response, record) in enumerate(zip(prompts, raw, canonical)):
        assert prompt['row_id'] == response['row_id'] == record['row_id']
        pmc = str(prompt['pmc_id'])
        assert str(response['pmc_id']) == str(record['pmc_id']) == pmc
        assert set(prompt['patient_ids']) == patient_ids[pmc]
        assert response['original_index'] == record['original_index'] == index
        assert response['provenance'] == record['provenance'] == 'fresh'
        tokens = json.loads((root / f'run/native/generated-ids/{index:03d}.json').read_text())
        assert tokens['row_id'] == prompt['row_id'] and len(tokens['generated_ids']) == response['output_tokens']
        assert tokens['natural_eos'] == response['natural_eos'] == record['natural_eos']
        assert tokens['hit_token_limit'] == response['hit_token_limit'] == record['hit_token_limit']
        assert list(prompt['code_map']) == [f'C{i:03d}' for i in range(len(prompt['code_map']))]
        for hp in prompt['code_map'].values():
            assert (pmc, hp) not in concepts
            concepts.add((pmc, hp))
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
            for pid in patients:
                selected[pmc][pid].add(prompt['code_map'][cid])
    assert len(concepts) == 415 and complete == terminal['strict_complete']
    documents = deepcopy(base)
    for doc in documents:
        pmc = str(doc['pmc_id'])
        assert {x['patient_id'] for x in doc['association']} == patient_ids[pmc]
        for row in doc['association']:
            row['phenotype'] = sorted(selected[pmc][row['patient_id']])
    single, union, counts = assemble_two(lines, documents)
    frozen = {}
    for name, output in [('new-alone-unscored', single), ('protected-union', union)]:
        assert nonrelation_bytes(output) == nonrelation_bytes(lines)
        path = out / (name + '.prediction.jsonl')
        with path.open('x') as stream:
            stream.writelines(output)
            stream.flush()
            os.fsync(stream.fileno())
        frozen[name] = ref(path)
    save(out / 'complete-freeze.json', {'at_utc': stamp(), 'predictions': frozen, 'relations': counts,
         'documents': 16, 'patients': 47, 'entities': 947, 'nonassociation_bytes_exact': True,
         'fresh_rows': 58, 'strict_complete': complete, 'whole_row_abstentions': 58 - complete,
         'gold_reads_before_freeze': 0, 'v2_calls_before_freeze': 0,
         'quality_endpoint': 'protected-union only; raw new-alone reserved for root fixed ensemble, never independently scored'})
    validator = module(plan['refs']['validator']['path'], 'validate_predictions')
    evaluator = module(plan['refs']['evaluator']['path'], 'qwen3_f0_fixed_evaluator')
    gold = evaluator.load_jsonl(checked(plan['refs']['gold']))
    hpo = validator.read_hpo(plan['refs']['hpo']['path'])
    baseline = json.loads(Path(plan['refs']['baseline_metrics']['path']).read_text())['metrics']
    assert baseline['Score'] == .7115075800266846
    prediction = jsonl(frozen['protected-union']['path'])
    valid = validator.validate_predictions(inputs, prediction, hpo)
    save(out / 'validation.json', valid)
    assert valid['valid'] and valid['warning_count'] == 0
    save(out / 'evaluation-started.json', {'at_utc': stamp(), 'freeze': ref(out / 'complete-freeze.json'), 'new_v2': 1, 'old_v2': 0})
    report = evaluator.evaluate_report(gold, prediction, 'all', 'text')
    save(out / 'metrics.json', report)
    save(out / 'result.json', {'status': 'COMPLETE_FIXED_QWEN3_F0_PROTECTED_UNION_ONLY',
         'baseline': baseline, 'metrics': report['metrics'], 'counts': report['counts'],
         'delta': {k: report['metrics'][k] - baseline[k] for k in baseline},
         'keep_independent_union': report['metrics']['Score'] > baseline['Score'],
         'new_v2': 1, 'old_v2': 0, 'raw_new_alone_quality_calls': 0, 'freeze': ref(out / 'complete-freeze.json'),
         'disclosure': 'Same F2F3 train32 epoch2 Qwen3 adapter; F0 candidate48 vs F1 candidate32, all exclude their held fold. F1-selected policy fixed on existing F0 development, not blindfinal. No refit, threshold search or patient selection.'})


if __name__ == '__main__':
    main()
