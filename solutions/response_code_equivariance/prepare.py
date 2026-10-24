"""Prepare one complete cyclic1 held view and structural contracts. Stdlib only."""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, os, time
import original_response_codec as original
import cyclic1_codec as cyclic

CODED_SHA = '54c8cb8c3e273dca62d6972cf24fc520d921fde091f35626931adf5de2676e9b'
CLINICAL_SHA = '2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085'
CODEC_SHA = 'da6410ac3a93c6832de950ee03cdba6e7c0388151b10a050fc9859242c9faad6'


def sha(data): return hashlib.sha256(data).hexdigest()
def jbytes(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()
def ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
def save(path, value):
    with Path(path).open('x') as h:
        json.dump(value, h, ensure_ascii=False, indent=2); h.write('\n'); h.flush(); os.fsync(h.fileno())
def put(path, data):
    with Path(path).open('xb') as h:
        h.write(data); h.flush(); os.fsync(h.fileno())


def contracts():
    passed = []
    def reject(name, call):
        try: call()
        except (ValueError, TypeError, KeyError): passed.append(name); return
        raise AssertionError('Invalid fixture accepted: ' + name)
    clinical = {'row_id': 'synthetic', 'pmc_id': 'synthetic',
        'patient_ids': ['OII.1', 'P2', 'OII.2'], 'code_map': {'C001': 'HP:0000001', 'C002': 'HP:0000002'},
        'messages': [{'role': 'system', 'content': original.OLD_OUTPUT + '\n' + original.OLD_RELATIVE},
                     {'role': 'user', 'content': '[PATIENT OII.1: sibling] same text [PATIENT P2: sibling] [PATIENT OII.2: sibling]; clinical II.1 remains II.1.'}]}
    old = original.transform(clinical); new = cyclic.transform(old)
    assert list(new['response_code_map'].items()) == [('PT001', 'OII.2'), ('PT002', 'OII.1'), ('PT003', 'P2')]
    assert new['patient_ids'] == clinical['patient_ids'] and new['messages'][1]['content'].startswith(clinical['messages'][1]['content'])
    assert cyclic.decode_response('{"C001":["PT003","PT001"],"C002":[]}', new) == {'C001': ['P2', 'OII.2'], 'C002': []}
    assert original.decode_response('{"C001":["PT001"],"C002":[]}', old) != cyclic.decode_response('{"C001":["PT001"],"C002":[]}', new)
    passed += ['nontrivial_cyclic_direction', 'clinical_identity_same_mentions_preserved', 'ordered_inverse_not_set_decoder', 'old_and_cyclic_inverse_not_interchangeable']
    for name, text in [
        ('missing_CID', '{"C001":[]}'), ('extra_CID', '{"C001":[],"C002":[],"C003":[]}'),
        ('duplicate_CID', '{"C001":[],"C001":[],"C002":[]}'),
        ('unknown_response_code', '{"C001":["PT004"],"C002":[]}'),
        ('original_PID_instead_of_code', '{"C001":["OII.1"],"C002":[]}'),
        ('duplicate_response_code', '{"C001":["PT001","PT001"],"C002":[]}'),
        ('wrong_value_type', '{"C001":"PT001","C002":[]}'),
        ('nonstring_patient', '{"C001":[1],"C002":[]}'),
        ('markdown', '```json\n{"C001":[],"C002":[]}\n```')]:
        reject(name, lambda text=text: cyclic.decode_response(text, new))
    reject('double_transform', lambda: cyclic.transform(new))
    bad = copy.deepcopy(new); bad['response_code_map']['PT001'] = 'OII.1'
    reject('nonbijective_map', lambda: cyclic.decode_response('{"C001":[],"C002":[]}', bad))
    bad = copy.deepcopy(new); bad['messages'][1]['content'] = old['messages'][1]['content']
    reject('table_map_disagreement', lambda: cyclic.validate_view(bad))
    singleton = copy.deepcopy(clinical); singleton['patient_ids'] = ['OII.1']
    one = original.transform(singleton)
    assert cyclic.transform(one) == one
    assert cyclic.decode_response('{"C001":["PT001"],"C002":[]}', one) == original.decode_response('{"C001":["PT001"],"C002":[]}', one)
    passed.append('singleton_identical_representation_and_inverse')
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coded', required=True, type=Path)
    parser.add_argument('--clinical', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    coded_bytes, clinical_bytes = args.coded.read_bytes(), args.clinical.read_bytes()
    assert sha(coded_bytes) == CODED_SHA and sha(clinical_bytes) == CLINICAL_SHA
    assert sha(Path(original.__file__).read_bytes()) == CODEC_SHA
    lines = coded_bytes.splitlines(keepends=True)
    old, clinical = [json.loads(x) for x in lines], [json.loads(x) for x in clinical_bytes.splitlines()]
    assert len(old) == len(clinical) == 80 and len({x['row_id'] for x in old}) == 80
    assert len({str(x['pmc_id']) for x in old}) == 16
    output, audit, shifted_indices, cache_indices = [], [], [], []
    documents, candidate_count, pair_count = {}, 0, 0
    for index, (row, raw, line) in enumerate(zip(old, clinical, lines)):
        assert line.endswith(b'\n') and row == original.transform(raw)
        assert row['row_id'] == raw['row_id'] and row['pmc_id'] == raw['pmc_id'] and 'answer' not in row
        view = cyclic.transform(row)
        old_suffix = cyclic.table_text(row['response_code_map']); new_suffix = cyclic.table_text(view['response_code_map'])
        prefix = row['messages'][-1]['content'][:-len(old_suffix)]
        assert prefix.encode() == raw['messages'][-1]['content'].encode()
        assert view['messages'][-1]['content'][:-len(new_suffix)].encode() == prefix.encode()
        assert view['messages'][0] == row['messages'][0]
        assert {k:v for k,v in row.items() if k not in ['messages', 'response_code_map']} == {k:v for k,v in view.items() if k not in ['messages', 'response_code_map']}
        assert view['patient_ids'] == row['patient_ids'] and list(view['code_map'].items()) == list(row['code_map'].items())
        # Synthetic complete matrices verify every inverse without any gold or
        # original model answer; reverse list order exposes accidental sorting.
        wanted = {cid: list(reversed(row['patient_ids'])) for cid in row['code_map']}
        inverse = {pid:code for code,pid in view['response_code_map'].items()}
        synthetic = {cid:[inverse[pid] for pid in pids] for cid,pids in wanted.items()}
        assert cyclic.decode_response(json.dumps(synthetic), view) == wanted
        for pid in row['patient_ids']:
            single = {cid:[inverse[pid]] for cid in row['code_map']}
            assert cyclic.decode_response(json.dumps(single), view) == {cid:[pid] for cid in row['code_map']}
        n = len(row['patient_ids']); cache = n == 1
        if cache:
            assert view == row and jbytes(view['messages']) == jbytes(row['messages'])
            result_line = line; cache_indices.append(index)
        else:
            assert all(view['response_code_map'][code] != row['response_code_map'][code] for code in row['response_patient_codes'])
            result_line = (json.dumps(view, ensure_ascii=False) + '\n').encode(); shifted_indices.append(index)
        output.append(result_line)
        pmc = str(row['pmc_id']); signature = (row['patient_ids'], list(view['response_code_map'].items()))
        if pmc in documents: assert documents[pmc] == signature
        else: documents[pmc] = signature
        candidate_count += len(row['code_map']); pair_count += len(row['code_map']) * n
        audit.append({'index': index, 'row_id': row['row_id'], 'pmc_id': pmc, 'patients': n,
            'candidate_codes': len(row['code_map']), 'cache_original_raw': cache, 'needs_new_model_response': not cache,
            'original_line_sha256': sha(line), 'cyclic1_line_sha256': sha(result_line),
            'clinical_user_prefix_sha256': sha(prefix.encode()),
            'original_messages_sha256': sha(jbytes(row['messages'])), 'cyclic1_messages_sha256': sha(jbytes(view['messages'])),
            'clinical_PID_order_sha256': sha(jbytes(row['patient_ids'])),
            'CID_HPO_order_sha256': sha(jbytes(list(row['code_map'].items()))),
            'map_sha256': sha(jbytes(view['response_code_map'])),
            'all_clinical_user_prefix_UTF8_preserved': True, 'all_metadata_except_response_map_preserved': True,
            'inverse_all_patients_and_order_preserved_synthetic': True, 'singleton_line_bytes_exact': cache})
    assert (len(documents), sum(len(x[0]) for x in documents.values()), candidate_count, pair_count) == (16, 39, 587, 1557)
    tests = contracts(); args.output.mkdir(parents=True, exist_ok=False)
    combined = b''.join(output)
    put(args.output/'held.cyclic1.jsonl', combined)
    put(args.output/'original-coded80.jsonl.gz', gzip.compress(coded_bytes, mtime=0))
    put(args.output/'cyclic1-coded80.jsonl.gz', gzip.compress(combined, mtime=0))
    save(args.output/'row-contract.json', audit)
    result = {'status': 'PASS_LOCAL_STRUCTURE_CYCLIC1_NO_TOKENIZER_MODEL_OR_QUALITY', 'view': cyclic.VIEW,
        'rows': 80, 'documents': 16, 'patients': 39, 'candidate_codes': candidate_count, 'pairs': pair_count,
        'new_NN_rows': len(shifted_indices), 'exact_original_raw_cache_rows': len(cache_indices),
        'new_NN_indices': shifted_indices, 'original_cache_indices': cache_indices,
        'multi_patient_documents': sum(len(x[0]) > 1 for x in documents.values()),
        'single_patient_documents': sum(len(x[0]) == 1 for x in documents.values()),
        'rotation': 'patient i receives PT((i+1)%n+1); canonical PT table order unchanged; no alternative shifts',
        'all80_original_clinical_user_UTF8_prefix_preserved': True,
        'all80_system_marker_patient_CID_HPO_order_preserved': True,
        'all_singleton_serialized_lines_byte_exact': True,
        'synthetic_tests': tests, 'synthetic_tests_count': len(tests),
        'source': ref(__file__), 'codec': ref(cyclic.__file__), 'original_codec': ref(original.__file__),
        'inputs': {'coded': ref(args.coded), 'clinical': ref(args.clinical)},
        'outputs': {n:ref(args.output/n) for n in ['held.cyclic1.jsonl', 'original-coded80.jsonl.gz', 'cyclic1-coded80.jsonl.gz', 'row-contract.json']},
        'original_raw_responses_read': False, 'gold_read': False, 'tokenizer_model_GPU_loaded': False,
        'token_cost': None, 'quality': None, 'seconds': time.monotonic()-started}
    save(args.output/'structural-result.json', result)
    print(json.dumps({k:result[k] for k in ['status','rows','new_NN_rows','exact_original_raw_cache_rows','synthetic_tests_count','seconds']}))


if __name__ == '__main__': main()
