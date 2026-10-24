"""Assemble the complete, unscored originaldev16 relation component.

Consumes already frozen strict canonical responses; no raw parsing, model,
gold, scorer, threshold, or fusion. Original nonrelation bytes remain exact.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import time

TEMPLATE_SHA = 'c1c2e508ec49272aa19ccdd37a7d0f39bb5dd33bc28e081ed9385f725489eca2'
HELD_SHA = 'b76e4b3de00432c692c0cefd84baa44275cf62404e9887f1e051ec0982db1f9b'


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=Path('/workspace/patientphex-runs/round2-20261023/sequence_crf/gliner-dev-transfer/assembled/full-lr.jsonl'))
    args = parser.parse_args()
    root, terminal = args.inference, args.inference / 'terminal'
    assert ref(args.template)['sha256'] == TEMPLATE_SHA
    assert ref(root / 'held.response-code.jsonl')['sha256'] == HELD_SHA
    plan, manifest = read(root / 'plan.json'), read(root / 'terminal-manifest.json')
    assert manifest['plan_sha256'] == ref(root / 'plan.json')['sha256']
    assert plan['training_completed_epochs'] == 2 and plan['training_global_step'] == 68
    assert plan['cached_original_indices'] == [] and plan['fresh_rows'] == 64
    for item in manifest['files']:
        actual = ref(terminal / item['name'])
        assert all(actual[key] == item[key] for key in ('bytes', 'sha256'))
    wrapper, summary = read(terminal / 'wrapper-complete.json'), read(terminal / 'summary.json')
    assert wrapper['returncode'] == 0 and not wrapper['timed_out']
    assert not read(terminal / 'gpu-release.json')['target_compute_rows']
    assert summary['fresh_rows'] == summary['expected_rows'] == 64 and summary['cached_rows'] == 0
    assert summary['source_sha256'] == plan['files']['infer_full64.py']['sha256']
    assert summary['plan_sha256'] == ref(root / 'plan.json')['sha256']
    assert summary['canonical_sha256'] == ref(terminal / 'canonical.jsonl')['sha256']
    assert summary['responses_sha256'] == ref(terminal / 'responses.jsonl')['sha256']
    prompts, canonical, responses = (rows(root / 'held.response-code.jsonl'),
                                      rows(terminal / 'canonical.jsonl'), rows(terminal / 'responses.jsonl'))
    assert len(prompts) == len(canonical) == len(responses) == 64
    assert len({row['row_id'] for row in prompts}) == 64
    templates = rows(args.template)
    assert (len(templates), sum(len(doc['entities']) for doc in templates),
            sum(len(doc['association']) for doc in templates)) == (16, 955, 36)
    decisions = {str(doc['pmc_id']): {str(a['patient_id']): set() for a in doc['association']} for doc in templates}
    assert len(decisions) == 16 and set(decisions) == {str(row['pmc_id']) for row in prompts}
    candidates = pairs = abstentions = 0
    for index, (row, parsed, raw) in enumerate(zip(prompts, canonical, responses)):
        assert row['row_id'] == parsed['row_id'] == raw['row_id']
        pmc = str(row['pmc_id'])
        assert pmc == str(parsed['pmc_id']) == str(raw['pmc_id'])
        assert parsed['original_index'] == index and parsed['provenance'] == 'fresh'
        assert set(row['patient_ids']) == set(decisions[pmc])
        assert read(terminal / f'canonical-records/{index:03d}.json') == parsed
        assert read(terminal / f'raw-responses/{index:03d}.json') == raw
        ids = read(terminal / f'generated-ids/{index:03d}.json')
        assert ids['row_id'] == row['row_id'] and len(ids['generated_ids']) == raw['output_tokens']
        assert ids['natural_eos'] == raw['natural_eos'] and ids['hit_token_limit'] == raw['hit_token_limit']
        mapping = parsed['candidate_patients']
        if parsed['strict_complete']:
            assert parsed['abstention'] is None and raw['natural_eos'] and not raw['hit_token_limit']
            assert set(mapping) == set(row['code_map'])
            for cid, patients in mapping.items():
                assert len(patients) == len(set(patients)) and set(patients) <= set(row['patient_ids'])
                for patient in patients:
                    decisions[pmc][patient].add(row['code_map'][cid])
        else:
            assert mapping == {} and parsed['abstention']
            abstentions += 1
        candidates += len(row['code_map'])
        pairs += len(row['code_map']) * len(row['patient_ids'])
    assert (candidates, pairs) == (466, 1006)
    assert abstentions == summary['whole_row_abstentions']
    helper = Path(__file__).resolve().parents[2] / 'quality/literal_union_helpers.py'
    spec = importlib.util.spec_from_file_location('pure_literal_association_helper', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_lines = args.template.read_text().splitlines(keepends=True)
    output = []
    for line in original_lines:
        doc = json.loads(line)
        for association in doc['association']:
            association['phenotype'] = sorted(decisions[str(doc['pmc_id'])][str(association['patient_id'])])
        begin, end = module.association_span(line)
        output.append(line[:begin] + json.dumps(doc['association'], ensure_ascii=False) + line[end:])
    assert module.nonrelation_bytes(original_lines) == module.nonrelation_bytes(output)
    destination = root / 'pure-component'
    destination.mkdir(exist_ok=False)
    with (destination / 'pred.jsonl').open('x') as handle:
        handle.write(''.join(output))
        handle.flush()
        os.fsync(handle.fileno())
    save(destination / 'manifest.json', dict(status='COMPLETE_UNSCORED_RESPONSE_CODE_FULL64_COMPONENT',
        at_unix=time.time(), source=ref(__file__), literal_helper=ref(helper),
        template=ref(args.template), prompts=ref(root / 'held.response-code.jsonl'),
        plan=ref(root / 'plan.json'), inference_manifest=ref(root / 'terminal-manifest.json'),
        canonical=ref(terminal / 'canonical.jsonl'), prediction=ref(destination / 'pred.jsonl'),
        documents=16, entities=955, patients=36, rows=64, candidates=466, candidate_pairs=1006,
        relations=sum(len(values) for doc in decisions.values() for values in doc.values()),
        whole_row_abstentions=abstentions, all_nonrelation_bytes_preserved=True,
        new_model_calls=0, raw_reparsed=False, gold_read=False, quality_calls=0, fusion_performed=False))
    descriptor = os.open(str(destination), os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps(ref(destination / 'manifest.json')))


if __name__ == '__main__':
    main()
