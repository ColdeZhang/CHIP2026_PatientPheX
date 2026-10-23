import copy
import json
import unittest
from retry_links import errors, split_request


class LinkRetryTests(unittest.TestCase):
    def setUp(self):
        self.items = {f'M{i}': {'candidates': [{'identifier': 'HP:0001250'}]} for i in range(3)}
        self.blocks = [f'M{i}: quote="seizures"; context="a context with \\n escaped text"\nCandidates: ["HP:0001250"]' for i in range(3)]
        self.prompt = {'row_id': 'link:0', 'item_codes': list(self.items),
                       'messages': [{'role': 'system', 'content': 'fixed instructions'},
                                    {'role': 'user', 'content': '\n\n'.join(self.blocks)}]}

    def test_only_structural_failures_trigger_retry(self):
        valid = {'response': json.dumps(dict(M0='HP:0001250', M1='-1', M2=['HP:0001250'])), 'hit_token_limit': False}
        self.assertEqual(errors(self.prompt, valid, self.items), [])
        invalid = {'response': json.dumps(dict(M0='HP:0001250', M2='HP:9999999')), 'hit_token_limit': True}
        self.assertEqual(errors(self.prompt, invalid, self.items),
                         ['token_limit', 'item_key_mismatch', 'invalid_candidate:M2'])

    def test_smaller_requests_preserve_every_original_block(self):
        original = copy.deepcopy(self.prompt)
        retries = split_request(self.prompt, 2)
        self.assertEqual(self.prompt, original)
        self.assertEqual([c for p in retries for c in p['item_codes']], list(self.items))
        self.assertEqual([b for p in retries for b in p['messages'][1]['content'].split('\n\n')], self.blocks)
        self.assertTrue(all(p['messages'][0] == self.prompt['messages'][0] for p in retries))
        with self.assertRaises(ValueError):
            split_request(retries[0], 1)


if __name__ == '__main__':
    unittest.main()
