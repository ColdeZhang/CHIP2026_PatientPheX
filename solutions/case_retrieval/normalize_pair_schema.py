#!/usr/bin/env python3
"""A separate deterministic schema control; preserve immutable raw responses."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

POLICY = {
    'name': 'exact-patient-evidence-wrapper-v1',
    'required_object_keys': ['evidence', 'patient_id'],
    'patient_id': 'Exact supplied string ID; unique within this candidate list.',
    'evidence': 'String, retained verbatim in raw and audit; not checked for clinical correctness.',
    'eligible_value': 'Nonempty list entirely of exact two-field objects; no mixed values.',
    'scope': 'Only expected CID keys already present in a strict JSON object.',
    'reject_row': 'Token limit, non-strict JSON, duplicate JSON keys, non-object top level.',
    'unmodified': 'Missing/unknown CID or HPA keys, unknown/duplicate IDs, additional fields, '
                  'non-string fields, mixed/scalar values, and all originally valid lists.',
    'application': 'Independent paired base/adapter control; never replace original primary parsing.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def strict_object(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('non_json_constant:' + value)

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('non_object_top_level')
    return value


def normalize_row(prompt, response):
    assert response['row_id'] == prompt['row_id']
    assert str(response['pmc_id']) == str(prompt['pmc_id'])
    assert isinstance(response['response'], str)
    given = prompt['patient_ids']
    assert all(isinstance(pid, str) for pid in given) and len(set(given)) == len(given)
    patients = set(given)
    output = copy.deepcopy(response)
    audit = {'row_id': prompt['row_id'], 'pmc_id': str(prompt['pmc_id']),
             'original_response_sha256': text_sha(response['response']),
             'projected_candidates': [], 'rejected_candidates': [], 'row_rejection': None}
    if response.get('hit_token_limit', False):
        audit['row_rejection'] = 'hit_token_limit'
    else:
        try:
            parsed = strict_object(response['response'])
        except (ValueError, TypeError) as error:
            audit['row_rejection'] = str(error)
        else:
            audit['missing_codes'] = sorted(set(prompt['code_map']) - set(parsed))
            audit['extra_codes'] = sorted(set(parsed) - set(prompt['code_map']))
            for code in prompt['code_map']:
                value = parsed.get(code)
                # Ordinary lists of IDs and every unrecognized shape are left to the original merger.
                if not isinstance(value, list) or not value or not any(isinstance(x, dict) for x in value):
                    continue
                reason = None
                if not all(isinstance(x, dict) and set(x) == {'patient_id', 'evidence'} for x in value):
                    reason = 'mixed_or_unrecognized_object_fields'
                elif not all(isinstance(x['patient_id'], str) and x['patient_id'] in patients
                             and isinstance(x['evidence'], str) for x in value):
                    reason = 'unknown_or_nonstring_patient_or_evidence'
                else:
                    ids = [x['patient_id'] for x in value]
                    if len(ids) != len(set(ids)):
                        reason = 'duplicate_patient_id'
                if reason:
                    audit['rejected_candidates'].append({'code': code, 'reason': reason})
                    continue
                audit['projected_candidates'].append({'code': code, 'original_value': copy.deepcopy(value),
                                                       'projected_value': ids})
                parsed[code] = ids
            if audit['projected_candidates']:
                audit['original_response'] = response['response']
                output['response'] = json.dumps(parsed, ensure_ascii=False)
    audit['modified_response_sha256'] = text_sha(output['response'])
    audit['changed'] = bool(audit['projected_candidates'])
    return output, audit


def restore_row(normalized, audit):
    """Recover the exact response text from the sidecar; no inference or identity guessing."""
    assert normalized['row_id'] == audit['row_id'] and str(normalized['pmc_id']) == audit['pmc_id']
    assert text_sha(normalized['response']) == audit['modified_response_sha256']
    restored = copy.deepcopy(normalized)
    if audit['changed']:
        restored['response'] = audit['original_response']
    assert text_sha(restored['response']) == audit['original_response_sha256']
    return restored


def read_rows(path):
    data = Path(path).read_bytes()
    assert data.endswith(b'\n'), 'Require complete JSONL final newline'
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert len({row['row_id'] for row in rows}) == len(rows), 'Duplicate row ID'
    return rows


def normalize_file(args):
    out = Path(args.output)
    manifest_path = out.with_suffix('.schema-manifest.json')
    audit_path = out.with_suffix('.schema-audit.json')
    assert not any(p.exists() for p in [out, manifest_path, audit_path]), 'New output paths required'
    assert out.resolve() != Path(args.responses).resolve()
    prompts, responses = read_rows(args.prompts), read_rows(args.responses)
    by_id = {p['row_id']: p for p in prompts}
    assert {r['row_id']: str(r['pmc_id']) for r in responses} == {
        p['row_id']: str(p['pmc_id']) for p in prompts}, 'Response identity coverage mismatch'
    runner = json.loads(Path(args.runner_manifest).read_text())
    assert runner['output_sha256'] == sha(args.responses)
    assert runner['fingerprint']['prompts_sha256'] == sha(args.prompts)
    outputs, audits, counts = [], [], Counter()
    for response in responses:
        output, audit = normalize_row(by_id[response['row_id']], response)
        assert restore_row(output, audit) == response
        outputs.append(output)
        audits.append(audit)
        counts['rows'] += 1
        counts['modified_rows'] += audit['changed']
        counts['projected_candidate_decisions'] += len(audit['projected_candidates'])
        counts['projected_model_claimed_assignments'] += sum(len(x['projected_value']) for x in audit['projected_candidates'])
        counts['rejected_candidate_shapes'] += len(audit['rejected_candidates'])
        counts['row_rejections'] += audit['row_rejection'] is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in outputs))
    audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2) + '\n')
    manifest = {'source_sha256': sha(__file__), 'configuration': vars(args), 'policy': POLICY,
                'counts': dict(counts), 'exact_per_row_restore_verified': True,
                'files': {name: sha(path) for name, path in [
                    ('prompts', args.prompts), ('original_responses', args.responses),
                    ('runner_manifest', args.runner_manifest), ('normalized_responses', out), ('audit', audit_path)]},
                'raw_response_file_modified': False, 'gold_read': False, 'evaluated': False,
                'limitation': 'Only response strings may be normalized; generation token/time metadata describes '
                              'the original generation. Projection does not verify evidence or patient attribution.'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': dict(counts), 'output_sha256': sha(out), 'manifest_sha256': sha(manifest_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prompts', 'responses', 'runner-manifest', 'output']:
        parser.add_argument('--' + name, required=True)
    normalize_file(parser.parse_args())


if __name__ == '__main__':
    main()
