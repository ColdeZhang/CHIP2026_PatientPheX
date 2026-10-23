"""Small synthetic checks for the fixed filter; no corpus labels or GPU."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import sap_span_control as control


class FixedFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root / 'request'; self.request.mkdir()

        def entity(offset, text, identifier):
            return dict(offset=offset, length=len(text), text=text,
                        identifier=identifier, type='Phenotype', note=None)
        self.base = entity(0, 'abc', 'HP:0000001')
        self.edge = entity(3, 'def', 'HP:0000002')
        self.low = entity(8, 'ghi', 'HP:0000003')
        self.overlap = entity(2, 'cde', 'HP:0000004')
        self.association = [{'patient_id': 'p1', 'phenotype': ['HP:0000001']}]
        cs = [{k: e[k] for k in ['offset', 'length', 'text']}
              for e in [self.edge, self.low, self.overlap]]
        self.write(self.request / 'requests.jsonl', [{'pmc_id': '1', 'candidates': cs}])
        self.write(self.root / 'base.jsonl', [{'pmc_id': '1', 'entities': [self.base],
                                              'association': self.association}])
        self.write(self.root / 'linked.jsonl', [{'pmc_id': '1', 'entities':
                                               [self.edge, self.low, self.overlap]}])
        self.request_manifest = {
            'request_sha256': control.sha(self.request / 'requests.jsonl'),
            'files': {'baseline': {'sha256': control.sha(self.root / 'base.jsonl')},
                      'train': {'sha256': 'train48'}, 'input': {'sha256': 'input16'},
                      'frozen_candidates': {'sha256': 'reference'}}}
        control.save_json(self.request / 'request.manifest.json', self.request_manifest)
        self.rebuilt = {'passed': True, 'inputs': {'train': 'train48', 'input': 'input16'},
                        'reference_candidate_sha256': 'reference'}
        self.scores = [{'pmc_id': '1', 'candidates': [dict(c, probability=p, similarity=s)
            for c, (p, s) in zip(cs, [(.65, .92), (.649999, .99), (.99, .99)])]}]
        self.args = argparse.Namespace(
            request_dir=str(self.request), linked=str(self.root / 'linked.jsonl'),
            scores=str(self.root / 'scores.jsonl'), score_manifest=str(self.root / 'scores.manifest.json'),
            reconstruction_manifest=str(self.root / 'reconstruction.json'),
            baseline=str(self.root / 'base.jsonl'), output=str(self.root / 'output.jsonl'))
        self.refresh()

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def refresh(self):
        control.save_json(self.args.reconstruction_manifest, self.rebuilt)
        self.write(Path(self.args.scores), self.scores)
        control.save_json(self.args.score_manifest, {
            'request_sha256': self.request_manifest['request_sha256'],
            'reconstruction_manifest_sha256': control.sha(self.args.reconstruction_manifest),
            'output_sha256': control.sha(self.args.scores)})

    def test_threshold_inclusive_and_touching_spans_retained(self):
        control.merge(self.args)
        self.assertEqual(control.read(self.args.output)[0]['entities'], [self.base, self.edge])

    def test_existing_entity_and_association_preserved(self):
        control.merge(self.args)
        result = control.read(self.args.output)[0]
        self.assertEqual(result['entities'][0], self.base)
        self.assertEqual(result['association'], self.association)

    def test_failed_reconstruction_rejected(self):
        self.rebuilt['passed'] = False; self.refresh()
        with self.assertRaisesRegex(ValueError, 'has not passed'):
            control.merge(self.args)

    def test_wrong_training_fold_rejected(self):
        self.rebuilt['inputs']['train'] = 'train64'; self.refresh()
        with self.assertRaisesRegex(ValueError, 'fold differs'):
            control.merge(self.args)

    def test_missing_scores_rejected(self):
        self.scores[0]['candidates'].pop(); self.refresh()
        with self.assertRaisesRegex(ValueError, 'each exact requested occurrence'):
            control.merge(self.args)

    def test_unbound_scores_rejected(self):
        self.scores[0]['candidates'][0]['probability'] = .99
        self.write(Path(self.args.scores), self.scores)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            control.merge(self.args)


if __name__ == '__main__':
    unittest.main()
