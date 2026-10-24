#!/usr/bin/env python3
"""Independent pure-JSON semantic and exposure audit. No model, tokenizer or gold."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random

ROOT = Path('/workspace/patientphex-runs/round2-20261023')
OUT = ROOT / 'validation/candidate-order-warmstart-entry-review'
PREP = ROOT / 'research/candidate-order-training-feasibility'
ENTRY = ROOT / 'research/candidate-order-warmstart-entry'

def load(p):
    return json.loads(Path(p).read_text())

def ref(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]

def main():
    plan = load(PREP/'plan.json')
    trainpath = Path(plan['references']['train']['path'])
    old, new = rows(trainpath), rows(PREP/'reverse124.jsonl')
    assert ref(trainpath) == plan['references']['train']
    assert ref(PREP/'reverse124.jsonl')['sha256'] == '34778f082d1057cab4646a3ef259fd0f15f4e48f13944ad5216bbde042e609b8'
    prep = load(plan['references']['preparation_manifest']['path'])
    train_ids = sorted({str(x['pmc_id']) for x in old})
    assert len(old) == len(new) == len({x['row_id'] for x in old}) == 124
    assert len(train_ids) == 32 and train_ids == prep['training_document_ids']
    assert not set(train_ids) & set(prep['excluded_document_ids'])
    for fold in prep['provenance']:
        generator, target = set(fold['generator_ids']), set(fold['target_ids'])
        assert len(generator) == len(target) == 16
        assert not generator & target and generator | target == set(train_ids)
    assert prep['no_F1_or_outer0_in_any_generator_or_SFT_training']
    total_pairs = positives = candidates = changed = 0
    for a, b in zip(old, new):
        assert set(a) == set(b)
        assert b['row_id'] == a['row_id'] + ':reverse-candidates'
        assert b['pmc_id'] == a['pmc_id'] and b['patient_ids'] == a['patient_ids']
        codes = list(a['code_map'])
        assert list(b['code_map']) == codes
        assert list(b['code_map'].values()) == list(reversed(list(a['code_map'].values())))
        assert len(set(a['code_map'].values())) == len(codes)
        assert a['messages'][:-1] == b['messages'][:-1]
        assert a['messages'][-1]['role'] == b['messages'][-1]['role'] == 'user'
        assert set(a['messages'][-1]) == set(b['messages'][-1]) == {'role', 'content'}
        divider = '\n\nCandidate phenotypes:\n'
        ap, aj = a['messages'][-1]['content'].split(divider)
        bp, bj = b['messages'][-1]['content'].split(divider)
        assert ap == bp
        ac, bc = json.loads(aj), json.loads(bj)
        assert [c['code'] for c in ac] == codes == [c['code'] for c in bc]
        expected = copy.deepcopy(list(reversed(ac)))
        for item, cid in zip(expected, codes):
            item['code'] = cid
        assert bc == expected  # all meaning/evidence/nested literal IDs stay exact
        aa, ba = json.loads(a['answer']), json.loads(b['answer'])
        assert list(aa) == list(ba) == codes
        assert a['answer'] == json.dumps(aa, ensure_ascii=False, separators=(',', ':'))
        assert b['answer'] == json.dumps(ba, ensure_ascii=False, separators=(',', ':'))
        original_labels = {a['code_map'][cid]: pids for cid, pids in aa.items()}
        reverse_labels = {b['code_map'][cid]: pids for cid, pids in ba.items()}
        assert original_labels == reverse_labels
        assert all(len(v) == len(set(v)) and set(v) <= set(a['patient_ids']) for v in aa.values())
        total_pairs += len(codes) * len(a['patient_ids'])
        positives += sum(len(v) for v in aa.values())
        candidates += len(codes)
        changed += a['messages'] != b['messages']
    assert (total_pairs, positives, candidates, changed) == (2506, 594, 880, 119)
    tokens = load(PREP/'row-inverse-and-token-audit.json')
    original_native = {x['row_id']:x for x in load(plan['references']['native_token_audit']['path'])['rows']}
    assert len(tokens) == 124
    for a, receipt in zip(old, tokens):
        assert receipt['original_row_id'] == a['row_id']
        assert receipt['original'] == original_native[a['row_id']]
    schedule = load(ENTRY/'matched-32-call-schedule.json')
    expected_schedule = []
    for epoch in range(2):
        order = list(range(124))
        random.Random(20261023 + epoch).shuffle(order)
        for offset in range(0, 124, 8):
            indices = order[offset:offset+8]
            step = len(expected_schedule)
            factor = step / 2 if step < 2 else (32-step) / 30
            variant = 'reversed' if epoch == 0 else 'original'
            expected_schedule.append(dict(optimizer_call=step+1, new_stage_epoch=epoch+1,
                sample_ids=[old[i]['row_id'] for i in indices], rows=len(indices),
                control_variant='O', augmented_variant='R' if epoch == 0 else 'O',
                lr_used=1e-4*factor,
                control_tokens=sum(tokens[i]['original']['full_tokens'] for i in indices),
                augmented_tokens=sum(tokens[i][variant]['full_tokens'] for i in indices),
                control_supervised_tokens=sum(tokens[i]['original']['supervised_tokens'] for i in indices),
                augmented_supervised_tokens=sum(tokens[i][variant]['supervised_tokens'] for i in indices)))
    assert schedule == expected_schedule
    assert len(schedule) == 32 and sum(x['rows'] for x in schedule) == 248
    assert sum(x['control_tokens'] for x in schedule) == 805188
    assert sum(x['augmented_tokens'] for x in schedule) == 805190
    assert sum(x['control_supervised_tokens'] for x in schedule) == sum(x['augmented_supervised_tokens'] for x in schedule) == 14340
    result = dict(status='PASS_INDEPENDENT_TRAIN_ONLY_REVERSE_SEMANTICS_AND_NEW_STAGE_SCHEDULE',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reviewer_source=ref(__file__),
        references={str(p.relative_to(ROOT)):ref(p) for p in [trainpath, PREP/'reverse124.jsonl',PREP/'plan.json',PREP/'row-inverse-and-token-audit.json',ENTRY/'matched-32-call-schedule.json',Path(plan['references']['preparation_manifest']['path'])]},
        training_documents=32, rows_per_variant=124, candidate_values=880,
        patient_concept_pairs=2506, positives=594, negatives=1912,
        exact_two16_to16_OOF_source_isolation=True, no_F1_or_original_dev_labels=True,
        candidate_only_reversal_and_top_level_code_rename=True,
        all_clinical_prefix_and_system_messages_byte_exact=True,
        meaning_evidence_nested_literals_patient_IDs_and_label_semantics_exact=True,
        independently_rebuilt_32_call_schedule_exact=True,
        stage=dict(control='O/O', augmented='R/O', epochs=2, optimizer_calls=32,
                   exposures=248, warmup_calls=2, nonzero_lr_calls=31,
                   control_tokens=805188, augmented_tokens=805190, supervised_tokens_each=14340),
        token_receipt_scope='Existing CPU receipt; every O row exactly matches old native token audit. No new tokenizer or model run.',
        no_model_or_tokenizer_import=True,no_gold_read=True,no_network=True)
    (OUT/'data-and-schedule.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],report=ref(OUT/'data-and-schedule.json'))))

if __name__ == '__main__':
    main()
