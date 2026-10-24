"""Build one reversible strict32/F1 append-only bundle from frozen pure inputs."""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import construct

D = Path(__file__).resolve().parent
R = D.parents[2]
P = R / 'case_retrieval/qwen35-preparation/sft-meta1-fusion'


def ref(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(name, value):
    with (D / name).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    pins = {
        'original_train': (P / 'train.jsonl', 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'),
        'original_infer': (P / 'meta.prompts.jsonl', 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'),
        'train_input': (P / 'public-training-candidate-preparation/train32.input.jsonl', 'a2c2baecb86ca425ae978688014073b69a415dbc9d3526d24be96f82b89e3e6d'),
        'infer_input': (Path('/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl'), '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'),
        'strict_preparation': (P / 'preparation-manifest.json', '356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'),
        'pure_input64': (R / 'research/public-phenotagger-train64-cache/train64.input.jsonl', 'ee2523843ff61f524db313fb2f9c37707a858e40f252870a7a05f4e7c1b3c246'),
    }
    texts, sources = {}, {}
    for name, (path, expected) in pins.items():
        assert ref(path)['sha256'] == expected, name
        texts[name] = path.read_text()
        sources[name] = ref(path)
    preparation = json.loads(texts['strict_preparation'])
    train_docs, infer_docs = construct.rows(texts['train_input']), construct.rows(texts['infer_input'])
    public = {str(doc['pmc_id']): doc for doc in construct.rows(texts['pure_input64'])}
    assert all(not {'entities', 'association'} & doc.keys() for doc in public.values())
    assert all(doc == public[str(doc['pmc_id'])] for doc in train_docs + infer_docs)
    assert sorted(str(doc['pmc_id']) for doc in train_docs) == preparation['training_document_ids']
    assert {str(doc['pmc_id']) for doc in infer_docs} <= set(preparation['excluded_document_ids'])
    assert len(train_docs) == 32 and len(infer_docs) == 16
    assert not {str(doc['pmc_id']) for doc in train_docs} & {str(doc['pmc_id']) for doc in infer_docs}
    train, infer = construct.rows(texts['original_train']), construct.rows(texts['original_infer'])
    assert len(train) == 124 and len(infer) == 80
    documents, row_audits, full = {}, {}, {}
    for split, original, docs, is_train in [('train', train, train_docs, True), ('infer', infer, infer_docs, False)]:
        extended, documents[split], row_audits[split] = construct.append_reports(original, docs, is_train)
        full[split] = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in extended)
        with (D / (split + '.fullcontext.jsonl')).open('x') as handle:
            handle.write(full[split])
    checks = ['actual_pure_inputs_match_frozen_input64', 'strict32_F1_document_isolation',
              'all_204_original_rows_restored_exact', 'all_original_answer_bytes_CID_PID_candidates_preserved',
              'all_original_message_prefixes_preserved', 'all_original_passages_and_offsets_preserved',
              'all_patient_mentions_exact_original_slices']
    toy = {'pmc_id': 'toy', 'patient': [{'patient_id': 'P1', 'mention': [{'offset': 10, 'length': 4, 'text': 'abcd'}]},
                                      {'patient_id': 'P2', 'mention': [{'offset': 10, 'length': 4, 'text': 'abcd'}]},
                                      {'patient_id': 'P3', 'mention': [{'offset': 12, 'length': 4, 'text': 'cdef'}, {'offset': 17, 'length': 2, 'text': 'hi'}]}],
           'full_text': [{'offset': 10, 'text': 'abcdef hi', 'section_type': 'case', 'type': 'paragraph'}]}
    marked, audit = construct.legacy.marked_report(toy)
    assert '[PATIENT P1/P2: abcd]' in marked and '[PATIENT P3: hi]' in marked
    assert len(audit['passages'][0]['overlap_skipped_markers']) == 1 and audit['original_text_reconstruction_exact']
    checks.append('shared_and_overlap_marker_policy_matches_legacy')
    bad = copy.deepcopy(toy); bad['patient'][0]['mention'][0]['text'] = 'WRONG'
    try:
        construct.legacy.marked_report(bad)
    except AssertionError:
        checks.append('wrong_patient_coordinate_text_rejected')
    else:
        raise AssertionError('Bad patient coordinate accepted')
    baseline_train = D.parent / 'actual-cpu/token-mask-audit.json'
    baseline_infer = D.parent / 'actual-cpu/inference-input-audit.json'
    measured = D.parent / 'formal-preparation/formal/train-001/progress.json'
    sources.update({name: ref(path) for name, path in [('baseline_train_audit', baseline_train), ('baseline_infer_audit', baseline_infer), ('measured_original_epoch1', measured)]})
    bundle = {'schema': 'medgemma-same12480-fullcontext-budget-v1', 'sources': sources,
              'original_train': texts['original_train'], 'original_infer': texts['original_infer'],
              'train_input': texts['train_input'], 'infer_input': texts['infer_input'],
              'extended_train': full['train'], 'extended_infer': full['infer'],
              'baseline_train_audit': json.loads(baseline_train.read_text()),
              'baseline_infer_audit': json.loads(baseline_infer.read_text()),
              'measured_original_epoch1': json.loads(measured.read_text())}
    payload = gzip.compress(json.dumps(bundle, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0)
    with (D / 'budget-inputs.json.gz').open('xb') as handle:
        handle.write(payload)
    save('document-audits.json', documents)
    save('row-construction-audits.json', row_audits)
    summary = {'status': 'CPU_prepared_append_only_not_tokenized_not_trained', 'source': ref(Path(__file__)),
               'construction_source': ref(D / 'construct.py'), 'legacy_source': ref(D / 'legacy_full_context.py'),
               'interface_source': ref(D / 'medgemma_interface.py'), 'sources': sources,
               'checks_passed': checks, 'train_rows': 124, 'inference_rows': 80,
               'training_documents': 32, 'inference_documents': 16,
               'append_prefix': construct.legacy.APPEND_PREFIX,
               'training_answers_unchanged': True, 'all_nonmessage_fields_unchanged': True,
               'documents': {split: {'patients': sum(len(doc['patient_ids']) for doc in values),
                                    'passages': sum(len(doc['passages']) for doc in values),
                                    'given_mentions': sum(doc['given_mentions'] for doc in values),
                                    'inserted_markers': sum(len(p['inserted_markers']) for doc in values for p in doc['passages']),
                                    'overlap_skipped': sum(len(p['overlap_skipped_markers']) for doc in values for p in doc['passages'])}
                             for split, values in documents.items()},
               'artifacts': {name: ref(D / name) for name in ['train.fullcontext.jsonl', 'infer.fullcontext.jsonl', 'budget-inputs.json.gz', 'document-audits.json', 'row-construction-audits.json']},
               'F1_gold_opened': False, 'new_model_encoding': False, 'quality_calls': 0}
    save('preparation.json', summary)
    print(json.dumps({key: summary[key] for key in ['status', 'checks_passed', 'documents', 'F1_gold_opened']}))


if __name__ == '__main__':
    main()
