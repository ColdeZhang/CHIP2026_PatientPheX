"""Synthetic behavior tests; no actual predictions, patient labels or GPU."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import normalize_pair_schema as n


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {'row_id': 'doc:0', 'pmc_id': 'doc', 'patient_ids': ['P1', 'P2'],
                       'code_map': {'C000': 'HP:0000001', 'C001': 'HP:0000002'}}

    def apply(self, text, hit=False):
        row = {'row_id': 'doc:0', 'pmc_id': 'doc', 'response': text,
               'hit_token_limit': hit, 'output_tokens': 99}
        output, audit = n.normalize_row(self.prompt, row)
        self.assertEqual(n.restore_row(output, audit), row)
        self.assertEqual(output['output_tokens'], 99)
        return output, audit

    def test_exact_projection_and_lossless_sidecar(self):
        raw = '{ "C000": [{"patient_id":"P2","evidence":"claim"}], "C001": [] }'
        output, audit = self.apply(raw)
        self.assertEqual(json.loads(output['response']), {'C000': ['P2'], 'C001': []})
        self.assertEqual(audit['original_response'], raw)
        self.assertEqual(audit['projected_candidates'][0]['original_value'][0]['evidence'], 'claim')

    def test_ordinary_or_invalid_values_unchanged(self):
        for value in [[], ['P1'], ['P3'], ['P1', 'P1'], None, 'P1', {'patient_id': 'P1'}]:
            raw = json.dumps({'C000': value})
            output, audit = self.apply(raw)
            self.assertEqual(output['response'], raw)
            self.assertFalse(audit['changed'])

    def test_unknown_duplicate_mixed_and_extra_fields_rejected(self):
        obj = {'patient_id': 'P1', 'evidence': 'x'}
        values = [[dict(obj, patient_id='P3')], [obj, obj], [obj, 'P2'],
                  [dict(obj, present=False)], [{'patient_id': 'P1'}],
                  [dict(obj, evidence=None)], [dict(obj, patient_id=1)]]
        for value in values:
            raw = json.dumps({'C000': value})
            output, audit = self.apply(raw)
            self.assertEqual(output['response'], raw)
            self.assertEqual(len(audit['rejected_candidates']), 1)

    def test_unknown_codes_not_mapped_or_projected(self):
        raw = json.dumps({'HPA:0000001': [], 'C999': [{'patient_id': 'P1', 'evidence': 'x'}]})
        output, audit = self.apply(raw)
        self.assertEqual(output['response'], raw)
        self.assertEqual(audit['missing_codes'], ['C000', 'C001'])

    def test_truncated_duplicate_key_and_nonliteral_json_unchanged(self):
        valid = '{"C000":[{"patient_id":"P1","evidence":"x"}]}'
        for raw, hit in [(valid, True), ('{"C000":[],"C000":[]}', False),
                         ('```json\n' + valid + '\n```', False), ('[]', False),
                         ('{"C000":NaN}', False)]:
            output, audit = self.apply(raw, hit)
            self.assertEqual(output['response'], raw)
            self.assertIsNotNone(audit['row_rejection'])

    def test_multiple_patients_and_response_order_are_identity_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = [self.prompt, dict(self.prompt, row_id='doc:1')]
            replies = [{'row_id': p['row_id'], 'pmc_id': 'doc', 'response':
                        json.dumps({'C000': [{'patient_id': 'P1', 'evidence': 'a'},
                                             {'patient_id': 'P2', 'evidence': 'b'}]})}
                       for p in reversed(prompts)]
            for name, rows in [('prompts', prompts), ('responses', replies)]:
                (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'runner.json').write_text(json.dumps({'output_sha256': n.sha(root / 'responses.jsonl'),
                                                        'fingerprint': {'prompts_sha256': n.sha(root / 'prompts.jsonl')}}))
            n.normalize_file(argparse.Namespace(prompts=str(root / 'prompts.jsonl'), responses=str(root / 'responses.jsonl'),
                                                runner_manifest=str(root / 'runner.json'), output=str(root / 'normalized.jsonl')))
            outputs = n.read_rows(root / 'normalized.jsonl')
            self.assertEqual([r['row_id'] for r in outputs], ['doc:1', 'doc:0'])
            self.assertEqual(json.loads(outputs[0]['response'])['C000'], ['P1', 'P2'])
            self.assertEqual(n.read_rows(root / 'responses.jsonl'), replies)

    def test_wrong_identity_and_duplicate_rows_fail(self):
        with self.assertRaises(AssertionError):
            n.normalize_row(self.prompt, {'row_id': 'doc:0', 'pmc_id': 'other', 'response': '{}'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicates.jsonl'
            path.write_text(json.dumps(self.prompt) + '\n' + json.dumps(self.prompt) + '\n')
            with self.assertRaises(AssertionError):
                n.read_rows(path)


if __name__ == '__main__':
    unittest.main()
