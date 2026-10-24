"""Single append-only full-report variant, no candidate/target/patient changes."""
from pathlib import Path
import copy
import hashlib
import json
import legacy_full_context as legacy

LEGACY_SHA = 'ae7f5853fc55ab430b080631496b73686cd3c6dd1a80ce181553d695ea3e01bb'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rows(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def jsha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())


def append_reports(original, documents, training):
    assert digest(Path(legacy.__file__).read_bytes()) == LEGACY_SHA
    assert legacy.function_sha(legacy.__file__, 'mark_patients') == legacy.MARKER_FUNCTION_SHA
    assert all(not {'entities', 'association'} & doc.keys() for doc in documents)
    by_id = {str(doc['pmc_id']): doc for doc in documents}
    assert len(by_id) == len(documents) and {str(row['pmc_id']) for row in original} == set(by_id)
    assert len({row['row_id'] for row in original}) == len(original)
    reports, audits = {}, []
    for pmc, document in by_id.items():
        report, audit = legacy.marked_report(document)
        assert audit['all_patients_marked'] and audit['original_text_reconstruction_exact']
        reports[pmc] = report
        audit['input_document_sha256'] = jsha(document)
        audits.append(audit)
    extended, row_audits = [], []
    for row in original:
        pmc = str(row['pmc_id'])
        assert ('answer' in row) is training
        assert row['patient_ids'] == [str(patient['patient_id']) for patient in by_id[pmc]['patient']]
        assert [message['role'] for message in row['messages']] == ['system', 'user']
        suffix = legacy.APPEND_PREFIX + reports[pmc]
        new = copy.deepcopy(row)
        new['messages'][-1]['content'] += suffix
        restored = copy.deepcopy(new)
        restored['messages'][-1]['content'] = restored['messages'][-1]['content'][:-len(suffix)]
        assert restored == row
        assert new['messages'][:-1] == row['messages'][:-1]
        assert new['messages'][-1]['content'] == row['messages'][-1]['content'] + suffix
        assert {k: v for k, v in new.items() if k != 'messages'} == {k: v for k, v in row.items() if k != 'messages'}
        row_audits.append({'row_id': row['row_id'], 'pmc_id': pmc, 'original_row_sha256': jsha(row),
                           'extended_row_sha256': jsha(new), 'suffix_sha256': digest(suffix.encode()),
                           'original_messages_sha256': jsha(row['messages']),
                           'answer_utf8_sha256': digest(row['answer'].encode()) if training else None,
                           'original_row_restored_exact': True})
        extended.append(new)
    return extended, audits, row_audits
