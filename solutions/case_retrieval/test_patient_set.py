"""Patient-set supervision and parser boundaries on synthetic documents."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import patient_set as subject


class PatientSetTests(unittest.TestCase):
    def setUp(self):
        text = 'Alice had short stature. Bob had shortness and unusual gait.'
        self.doc = {'pmc_id': 'synthetic', 'pmid': None,
                    'full_text': [{'offset': 100, 'text': text, 'section_type': 'CASE', 'type': 'paragraph'}],
                    'patient': [{'patient_id': 'OII:3', 'mention': [{'text': 'Alice', 'offset': 100, 'length': 5}]},
                                {'patient_id': 'P1', 'mention': [{'text': 'Bob', 'offset': 100+text.index('Bob'), 'length': 3}]}],
                    'entities': [], 'association': [{'patient_id': 'OII:3', 'phenotype': []},
                                                    {'patient_id': 'P1', 'phenotype': []}]}
        self.valid = {'HP:0004322'}

    def entity(self, text, identifier, note=None):
        return {'offset': 100+self.doc['full_text'][0]['text'].index(text), 'length': len(text),
                'text': text, 'identifier': identifier, 'note': note}

    def response(self, *items):
        return json.dumps({'phenotypes': list(items)})

    def item(self, identifier, quote, segment='S000'):
        return {'identifier': identifier, 'segment_id': segment, 'quote': quote}

    def test_prompt_independent_of_all_gold(self):
        prompts = subject.make_prompts(self.doc)
        altered = copy.deepcopy(self.doc)
        altered['entities'] = [self.entity('short stature', 'HP:0004322')]
        altered['association'][0]['phenotype'] = ['HP:0004322']
        self.assertEqual(prompts, subject.make_prompts(altered))
        self.assertEqual(prompts[0]['patient_map'], {'P001': 'OII:3', 'P002': 'P1'})
        self.assertIn('[PATIENT P002: Bob]', prompts[0]['messages'][1]['content'])
        self.assertEqual(prompts[0]['segments'][0]['text'], self.doc['full_text'][0]['text'])

    def test_target_needs_positive_literal_gold_evidence(self):
        self.doc['association'][0]['phenotype'] = ['HP:0004322', 'HP:9999999', 'HP:0004322;HP:000328', 'missing']
        self.doc['entities'] = [self.entity('short stature', 'HP:0004322', 'NO')]
        rows, decisions = subject.make_targets(self.doc, subject.make_prompts(self.doc), {k: k for k in self.valid})
        self.assertEqual(json.loads(rows[0]['answer']), {'phenotypes': []})
        self.assertEqual({x.get('reason') for x in decisions}, {'no_positive_entity', 'hpo_outside_branch', 'malformed_hpo'})
        self.doc['entities'][0]['note'] = None
        rows, _ = subject.make_targets(self.doc, subject.make_prompts(self.doc), {k: k for k in self.valid})
        self.assertEqual(json.loads(rows[0]['answer'])['phenotypes'], [self.item('HP:0004322', 'short stature')])
        self.assertEqual(json.loads(rows[1]['answer']), {'phenotypes': []})

    def test_missing_training_labels_not_silently_empty(self):
        del self.doc['association']
        with self.assertRaisesRegex(ValueError, 'must provide'):
            subject.make_targets(self.doc, subject.make_prompts(self.doc), {})

    def test_quote_boundary_segment_and_marker_validation(self):
        row = subject.make_prompts(self.doc)[0]
        response = self.response(self.item('HP:0004322', 'short', 'S000'),
                                 self.item('HP:0004322', '[PATIENT P001: Alice]'),
                                 self.item('HP:0004322', 'short stature', 'S999'))
        raw, grounded, _ = subject.parse_response(response, row, self.valid)
        # "short" exists as a standalone word in short stature, so it is literal
        # support only; no claim that the quote is equivalent to the HPO ID.
        self.assertEqual(raw, self.valid); self.assertEqual(grounded, self.valid)
        raw, grounded, stats = subject.parse_response(self.response(self.item('HP:0004322', 'shortn')), row, self.valid)
        self.assertEqual(raw, self.valid); self.assertEqual(grounded, set())
        self.assertEqual(stats['unsupported_quote'], 1)

    def test_nil_is_grounded_text_never_bare_minus_one(self):
        row = subject.make_prompts(self.doc)[0]
        response = self.response(self.item('-1', 'unusual gait'), self.item('-1', 'hallucinated finding'),
                                 self.item('-1', '-1'), self.item('HP:9999999', 'unusual gait'))
        raw, grounded, stats = subject.parse_response(response, row, self.valid)
        self.assertEqual(raw, {'unusual gait'}); self.assertEqual(grounded, raw)
        self.assertEqual(stats['invalid_nil_text'], 1); self.assertEqual(stats['invalid_hpo'], 1)

    def test_invalid_json_not_repaired(self):
        row = subject.make_prompts(self.doc)[0]
        self.assertEqual(subject.parse_response('```json\n{"phenotypes":[]}\n```', row, self.valid)[2], {'invalid_json': 1})

    def test_multi_patient_merge_and_token_limit_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [subject.strip_document(self.doc)]
            prompts = subject.make_prompts(self.doc)
            baseline = dict(self.doc, entities=[self.entity('short stature', 'HP:0004322')])
            responses = [dict(row_id=r['row_id'], pmc_id=r['pmc_id'],
                              response=self.response(self.item('HP:0004322', 'short stature')),
                              hit_token_limit=(i == 1)) for i, r in enumerate(prompts)]
            for name, rows in [('input', docs), ('baseline', [baseline]), ('prompts', prompts), ('responses', responses)]:
                subject.write_rows(root / (name+'.jsonl'), rows)
            (root / 'hpo.obo').write_text('synthetic loader mocked')
            args = SimpleNamespace(**{name: str(root / (name+'.jsonl')) for name in
                                      ['input', 'baseline', 'prompts', 'responses']},
                                   hpo=str(root/'hpo.obo'), output_dir=str(root/'out'))
            with patch.object(subject, 'load_hpo', return_value=({}, self.valid, {}, 'synthetic')):
                subject.merge(args)
                raw = subject.read_rows(root/'out/raw.predictions.jsonl')[0]
                grounded = subject.read_rows(root/'out/grounded.predictions.jsonl')[0]
                self.assertEqual(raw['entities'], baseline['entities'])
                self.assertEqual(grounded['entities'], baseline['entities'])
                self.assertEqual(raw['association'], [{'patient_id': 'OII:3', 'phenotype': ['HP:0004322']},
                                                     {'patient_id': 'P1', 'phenotype': ['HP:0004322']}])
                self.assertEqual(grounded['association'][1]['phenotype'], [])
                audit = subject.read_rows(root/'out/parse-audit.jsonl')
                self.assertEqual(audit[1]['grounded_row_status'], 'token_limit_rejected')
                subject.write_rows(root/'responses.jsonl', responses[:1])
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    subject.merge(args)
                subject.write_rows(root/'responses.jsonl', responses+[responses[0]])
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    subject.merge(args)


if __name__ == '__main__':
    unittest.main()
