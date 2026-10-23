#!/usr/bin/env python3
"""Freeze the one label-free outerfold0 package for the identity-context control.

Only supplied input text/patients and the fixed prediction entities enter the
builder. Existing metrics are bound by SHA and reused; no evaluator, gold file,
model, optimizer, remote service or GPU interface is imported or invoked.
"""
from collections import Counter
from pathlib import Path
from types import ModuleType
import argparse
import datetime
import gzip
import hashlib
import json
import time

from tokenizers import Tokenizer


EXPECTED_SHA = {
    'prepare_source': '6fa1cf3c3d21363cefb2d1a12d428280034a0b021597929f687433ecef5e728e',
    'input': 'ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37',
    'entity_lr_prediction': '3e400ed82c5f2913e159116ec9e1ff1623ad559a14e69bc65889f9d0c2f6d25a',
    'rules_prediction': 'b44c10f1a39a2eae5960aa2c0a50c8d879f50b71f93f25f103d5fbefa064b145',
    'lr_metrics': 'd0aba2c7abe71295f8a1db937330c4d1a467950c654b69a8de66dd5dc2b499f9',
    'rules_metrics': 'a22261c246552c6c55f4395433fe7cb3855a345ceaa76ad3c7adf92a29780272',
    'old_prompts': 'd387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0',
    'train_inputs': 'd75a88b07aa8adc0ed3bbfd674857402c1d45833f7f2beaf332616aa964ffd27',
    'training_cpu_result': '5dacbb662c79079aba96bf6c1131ac6485b763b2713b7a494ec0d07893133377',
    'base_source': '41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0',
    'ce_source': '09ae97d291add225b295fe5d053f84bb35a788efaca66d9c1ef5240d97b2ae5d',
    'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
    'tokenizer': 'b5bedb48718888bcfeb46c8949793038c8aeb4013575fb54bf4d50798329659b',
}
OUTPUT_NAMES = ('inputs.jsonl', 'encoded-holdout16.json.gz', 'holdout-manifest.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def entities_only(path):
    """Prediction associations are deliberately never extracted or consumed."""
    result = {}
    for prediction in rows(path):
        pmc = str(prediction['pmc_id'])
        require(pmc not in result, 'Duplicate prediction document')
        result[pmc] = prediction['entities']
    return result


def verify_arrays(row, tokenizer):
    encoded = tokenizer.encode(row['header'], row['evidence'])
    old = row['old_active_tokens']
    extra = row['bridge_active_tokens']
    require(old == len(encoded.ids) and 0 < old <= 384 and 0 <= extra <= 128,
            'Old encoding or bridge length changed')
    for arm in ('baseline', 'bridge'):
        item = row['arms'][arm]
        require(set(item) == {'input_ids', 'token_type_ids', 'attention_mask'}, 'Wrong tensor fields')
        require(all(len(values) == 512 and all(type(v) is int for v in values)
                    for values in item.values()), 'Wrong tensor type or length')
        active = old + (extra if arm == 'bridge' else 0)
        require(active <= 512, 'Tensor overflow')
        require(item['input_ids'][:old] == encoded.ids and
                item['token_type_ids'][:old] == encoded.type_ids, 'Original active prefix changed')
        require(item['attention_mask'] == [1] * active + [0] * (512 - active), 'Wrong attention mask')
        require(item['input_ids'][old - 1] == item['input_ids'][active - 1] == 3, 'SEP moved')
        require(item['input_ids'][active:] == [0] * (512 - active) and
                item['token_type_ids'][active:] == [0] * (512 - active), 'Wrong padding')
        require(all(0 <= v < 30522 for v in item['input_ids']) and
                all(v in (0, 1) for v in item['token_type_ids']), 'Token or segment outside vocabulary')
    require(row['arms']['bridge']['token_type_ids'][old:old + extra] == [1] * extra,
            'Bridge segment changed')
    if extra == 0:
        require(row['arms']['baseline'] == row['arms']['bridge'], 'Empty bridge is not exact')


def prepare(plan_path, output):
    started = time.monotonic()
    require(all(not (output / name).exists() for name in OUTPUT_NAMES),
            'Existing holdout artifacts: recover them, never overwrite or rebuild implicitly')
    plan = json.loads(plan_path.read_text())
    require(plan['schema'] == 'patient-identity-holdout-preparation-plan-v1', 'Wrong scope')
    require(plan['source_sha256'] == sha(__file__), 'Holdout builder source changed')
    require(set(plan['sources']) == set(EXPECTED_SHA), 'Unexpected source role, including possible gold')
    sources = {}
    for role, expected in EXPECTED_SHA.items():
        item = plan['sources'][role]
        require(item['sha256'] == expected, 'Not the fixed source: ' + role)
        path = Path(item['path'])
        require(ref(path) == item, 'Source bytes changed: ' + role)
        sources[role] = path

    helper = ModuleType('frozen_identity_bridge_preparation')
    helper.__file__ = str(sources['prepare_source'])
    exec(compile(sources['prepare_source'].read_text(), helper.__file__, 'exec'), helper.__dict__)
    base, linker, old_make, traced_make = helper.helpers(
        sources['base_source'], sources['ce_source'], sources['hpo'])

    def forbid_gold(*args, **kwargs):
        raise AssertionError('Gold association access forbidden in holdout preparation')

    # Both extracted make_examples functions reference this same object.
    base.gold_associations = forbid_gold
    inputs = list(rows(sources['input']))
    require(len(inputs) == len({str(d['pmc_id']) for d in inputs}) == 16, 'Wrong input coverage')
    require(all(set(d) <= {'pmc_id', 'pmid', 'patient', 'full_text'} for d in inputs),
            'Unexpected input fields or annotations')
    require(sum(len(d['patient']) for d in inputs) == 47, 'Wrong supplied patient count')
    input_ids = {str(d['pmc_id']) for d in inputs}
    train_ids = set()
    train_rows = 0
    for row in rows(sources['train_inputs']):
        require('label' not in row, 'Training input artifact is not label-free')
        train_ids.add(str(row['pmc_id']))
        train_rows += 1
    require(train_rows == 4127 and len(train_ids) == 48 and not train_ids & input_ids,
            'Train48/holdout16 identity overlap or coverage mismatch')
    training = json.loads(sources['training_cpu_result'].read_text())
    require(training['source_sha256'] == EXPECTED_SHA['prepare_source'] and
            training['outputs']['inputs.jsonl']['sha256'] == EXPECTED_SHA['train_inputs'],
            'Training policy/source is not the same frozen preparation')

    pool = entities_only(sources['entity_lr_prediction'])
    rules_pool = entities_only(sources['rules_prediction'])
    require(set(pool) == set(rules_pool) == input_ids and pool == rules_pool,
            'Rules/LR entity arrays differ or input documents missing')
    require(sum(len(values) for values in pool.values()) == 947, 'Wrong fixed entity count')
    patient_ids = {str(d['pmc_id']): {p['patient_id'] for p in d['patient']} for d in inputs}
    require(all(len(patient_ids[str(d['pmc_id'])]) == len(d['patient']) for d in inputs),
            'Duplicate supplied patient IDs')
    expected_pairs = set()
    expected_concepts = set()
    prompt_count = 0
    for prompt in rows(sources['old_prompts']):
        prompt_count += 1
        pmc = str(prompt['pmc_id'])
        require('answer' not in prompt and pmc in input_ids and
                set(prompt['patient_ids']) == patient_ids[pmc], 'Changed old prompt identity/labels')
        for concept in prompt['code_map'].values():
            require((pmc, concept) not in expected_concepts, 'Repeated old document concept')
            expected_concepts.add((pmc, concept))
            expected_pairs.update((pmc, pid, concept) for pid in prompt['patient_ids'])
    require(prompt_count == 58 and len(expected_concepts) == 415 and len(expected_pairs) == 1166,
            'Original 58/415/1166 candidate contract changed')

    tokenizer = Tokenizer.from_file(str(sources['tokenizer']))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer.enable_truncation(max_length=384, strategy='only_second', direction='right')
    require(tokenizer.get_vocab_size() == 30522, 'Do not add marker embeddings')
    raw_tokenizer = Tokenizer.from_str(tokenizer.to_str())
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    records = []
    pair_keys = set()
    counts = Counter()
    decisions = Counter()
    for doc in inputs:
        pmc = str(doc['pmc_id'])
        for left, right in zip(doc['full_text'], doc['full_text'][1:]):
            require(int(left['offset']) + len(left['text']) <= int(right['offset']),
                    'Passages overlap or are out of order')
        for entity in pool[pmc]:
            pi, passage = base.passage_for(doc, int(entity['offset']))
            local = int(entity['offset']) - int(passage.get('offset', 0))
            require(pi >= 0 and local >= 0 and local + int(entity['length']) <= len(passage['text']) and
                    passage['text'][local:local + int(entity['length'])] == entity['text'],
                    'Fixed entity is not an exact original slice')
        old = old_make(doc, pool[pmc], linker, training=False)
        traced = traced_make(doc, pool[pmc], linker, training=False)
        require(old == [{k: v for k, v in row.items() if k != 'trace'} for row in traced],
                'Instrumented old header/evidence/row order changed')
        anchors, invalid = helper.anchors_from_input(doc, base)
        require(not invalid, 'Supplied patient anchor does not match the original input')
        counts['patient_anchors'] += len(anchors)
        for row in traced:
            key = (str(row['pmc_id']), row['patient_id'], row['concept'])
            require(key not in pair_keys and 'label' not in row, 'Duplicate or labeled holdout row')
            pair_keys.add(key)
            choices, selected = helper.bridge_candidates(doc, row, anchors, base)
            packed = helper.pack(tokenizer, row, choices, raw_tokenizer=raw_tokenizer)
            record = dict(pmc_id=key[0], patient_id=key[1], concept=key[2],
                          header=row['header'], evidence=row['evidence'], old_occurrences=row['trace'],
                          decisions=selected, **packed)
            verify_arrays(record, tokenizer)
            for bridge in record['kept_bridges']:
                passage = doc['full_text'][bridge['source_passage_index']]
                start = bridge['source_start'] - int(passage['offset'])
                end = bridge['source_end'] - int(passage['offset'])
                require(passage['text'][start:end] == bridge['quote'], 'Bridge quote provenance changed')
            records.append(record)
            counts['rows'] += 1
            counts['old_active_tokens'] += record['old_active_tokens']
            counts['extra_bridge_tokens'] += record['bridge_active_tokens']
            counts['bridge_rows'] += bool(record['bridge_active_tokens'])
            counts['empty_bridge_rows'] += not record['bridge_active_tokens']
            decisions.update(d['reason'] for d in selected)
            decisions.update(d['reject_reason'] for d in record['rejected_bridges'])
    require(pair_keys == expected_pairs and len(records) == 1166, 'Heldout pair keys differ from old prompts')

    controls = {}
    for name, prediction_role in [('rules', 'rules_prediction'), ('lr', 'entity_lr_prediction')]:
        metric = json.loads(sources[name + '_metrics'].read_text())
        require(metric['prediction_sha256'] == EXPECTED_SHA[prediction_role] and
                metric['documents'] == 16 and metric['patients'] == 47 and
                metric['evaluator_version'] == 'patientphex-public-approx-v2' and
                metric['configuration']['body_scope'] == 'all' and
                metric['configuration']['unlinked_document_mode'] == 'text', 'Old metric binding changed')
        controls[name] = dict(prediction=plan['sources'][prediction_role],
                              metrics_file=plan['sources'][name + '_metrics'],
                              metrics=metric['metrics'], reused_without_rescoring=True)
    require(all(controls['rules']['metrics'][k] == controls['lr']['metrics'][k]
                for k in ('menF1', 'docF1')), 'Same entities have incompatible old metrics')

    package = dict(schema='patient-identity-encoded-holdout-v1',
                   input_source_sha256=EXPECTED_SHA['input'],
                   entity_source_sha256=EXPECTED_SHA['entity_lr_prediction'],
                   tokenizer_source_sha256=EXPECTED_SHA['tokenizer'],
                   preparation_source_sha256=EXPECTED_SHA['prepare_source'],
                   label_free=True,
                   rows=[dict(key=[r['pmc_id'], r['patient_id'], r['concept']],
                              old_active_tokens=r['old_active_tokens'],
                              bridge_active_tokens=r['bridge_active_tokens'], arms=r['arms']) for r in records])
    require(all(set(row) == {'key', 'old_active_tokens', 'bridge_active_tokens', 'arms'}
                for row in package['rows']), 'Unexpected package fields or labels')
    for role, path in sources.items():
        require(ref(path) == plan['sources'][role], 'Source changed during CPU construction: ' + role)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'inputs.jsonl').open('xb') as handle:
        for record in records:
            handle.write(json_bytes(record) + b'\n')
    with (output / 'encoded-holdout16.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=9) as handle:
            handle.write(json_bytes(package))
    train_id_bytes = json_bytes(sorted(train_ids))
    manifest = dict(status='complete_label_free_holdout_cpu_no_quality_result',
                    created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_sha256=sha(__file__), plan=ref(plan_path), sources=plan['sources'],
                    input_document_ids=sorted(input_ids), training_document_ids=sorted(train_ids),
                    training_document_ids_sha256=hashlib.sha256(train_id_bytes).hexdigest(),
                    training_document_ids_hash_encoding='UTF8 compact JSON of lexicographically sorted string IDs',
                    train48_holdout16_disjoint=True, documents=16, patients=47, fixed_entities=947,
                    old_prompt_rows=58, document_concepts=415, patient_concept_pairs=1166,
                    counts=dict(counts), selection_decisions=dict(decisions),
                    old_rows_header_evidence_order_exact=True, old_active_ids_types_positions_exact=True,
                    empty_bridge_arrays_exact=True, tensor_length=512,
                    max_old_active_tokens=max(r['old_active_tokens'] for r in records),
                    max_extra_bridge_tokens=max(r['bridge_active_tokens'] for r in records),
                    maximum_bridge_arm_active_tokens=max(r['old_active_tokens'] + r['bridge_active_tokens'] for r in records),
                    candidate_associations_used=False, gold_access=False, originaldev_access=False,
                    training_labels_access=False, labels_in_encoded_package=False,
                    model_loaded=False, gpu_used=False, evaluation_performed=False,
                    reused_controls=controls, tokenizer_backend_version=__import__('tokenizers').__version__,
                    artifacts={name: ref(output / name) for name in OUTPUT_NAMES[:2]},
                    cpu_seconds=time.monotonic() - started,
                    limitations=['Pure input construction and SHA binding only; no new quality score.',
                                 'Additional anchor clauses may contain clinical facts; no inferred ownership or coreference.',
                                 'This outerfold0 has participated in prior research; it is not blind validation.'])
    with (output / 'holdout-manifest.json').open('x') as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(manifest=ref(output / 'holdout-manifest.json'), counts=dict(counts),
                          encoded=manifest['artifacts']['encoded-holdout16.json.gz']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
