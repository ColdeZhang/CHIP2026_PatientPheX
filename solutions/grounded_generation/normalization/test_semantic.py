"""Policy counterexamples and exact concept aggregation, without a neural model."""
import unittest
import numpy as np

from .semantic import PublicConceptBank, alias_conflict, decide, load_policy, precheck


class SemanticContract(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.strict = {"quote": "unusual seizure", "status": "abstain", "ids": [], "identifier": None,
                       "reason": "unmatched", "match": None}
        self.record = {"quote": self.strict["quote"], "token_count": 6, "candidates": [
            {"identifier": "HP:0001250", "score": 0.95, "alias": "seizure"},
            {"identifier": "HP:0001251", "score": 0.89, "alias": "ataxia"}]}

    def test_confident_unknown_can_map(self):
        self.assertEqual(decide(self.strict, self.record, self.policy)["ids"], ["HP:0001250"])

    def test_low_cosine_does_not_invent_nil(self):
        self.record["candidates"][0]["score"] = 0.899
        self.record["candidates"][1]["score"] = 0.70
        result = decide(self.strict, self.record, self.policy)
        self.assertEqual(result["status"], "abstain")
        self.assertIsNone(result["identifier"])

    def test_margin_rejects_close_distinct_concept(self):
        self.record["candidates"][1]["score"] = 0.93
        self.assertEqual(decide(self.strict, self.record, self.policy)["reason"], "semantic_distinct_concept_margin_too_small")

    def test_known_ambiguity_not_overridden(self):
        self.strict["reason"] = "ambiguous_mapping"
        self.assertEqual(decide(self.strict, self.record, self.policy)["reason"], "ambiguous_mapping")

    def test_known_nil_is_preserved(self):
        self.strict.update(status="nil", ids=["-1"], identifier="-1", reason="unanimous_whole_span_alias")
        self.assertEqual(decide(self.strict, self.record, self.policy)["status"], "nil")

    def test_query_truncation_is_abstention(self):
        self.record.update(token_count=33, candidates=[])
        self.assertEqual(decide(self.strict, self.record, self.policy)["reason"], "semantic_query_would_truncate")

    def test_missing_scores_not_silent_strict_only(self):
        with self.assertRaises(ValueError):
            decide(self.strict, None, self.policy)

    def test_nonfinite_scores_rejected(self):
        self.record["candidates"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            decide(self.strict, self.record, self.policy)

    def test_unknown_compound_abbreviation_normal_negation(self):
        for phrase in ["seizures and ataxia", "normal hearing", "no hypotonia", "new finding; ataxia", "ABC"]:
            self.assertIsNotNone(precheck(phrase, self.policy), phrase)
        self.assertIsNone(precheck("abnormal hearing", self.policy))

    def test_direction_laterality_and_numbers(self):
        for quote, alias in [("increased muscle tone", "reduced muscle tone"), ("hypoplasia", "hyperplasia"),
                             ("left retinal lesion", "right retinal lesion"), ("type 2 disorder", "type 1 disorder")]:
            self.assertIsNotNone(alias_conflict(quote, alias, self.policy))
        self.assertIsNone(alias_conflict("increased muscle tone", "hypertonia", self.policy))

    def test_top2_is_concept_not_two_synonyms(self):
        bank = PublicConceptBank.__new__(PublicConceptBank)
        vectors = np.zeros((3, 768), dtype=np.float32)
        vectors[:, 0] = [0.96, 0.95, 0.80]
        vectors[:, 1] = np.sqrt(1 - vectors[:, 0] ** 2)
        bank.bank = vectors
        bank.rows = [{"text": "alias1"}, {"text": "alias2"}, {"text": "other"}]
        bank.concepts = ["HP:0001250", "HP:0001251"]
        bank.by_concept = {"HP:0001250": [0, 1], "HP:0001251": [2]}
        bank.alias_indices = np.array([0, 1, 2])
        bank.concept_indices = np.array([0, 0, 1])
        query = np.zeros(768, dtype=np.float32)
        query[0] = 1
        ranked = bank.rank(query)
        self.assertEqual([row["identifier"] for row in ranked], bank.concepts)
        self.assertGreater(ranked[0]["score"] - ranked[1]["score"], .15)

    def test_ambiguous_public_alias_gives_tied_concepts(self):
        bank = PublicConceptBank.__new__(PublicConceptBank)
        bank.bank = np.zeros((1, 768), dtype=np.float32)
        bank.bank[0, 0] = 1
        bank.rows = [{"text": "ambiguous"}]
        bank.concepts = ["HP:0001250", "HP:0001251"]
        bank.by_concept = {hid: [0] for hid in bank.concepts}
        bank.alias_indices = np.array([0, 0])
        bank.concept_indices = np.array([0, 1])
        ranked = bank.rank(bank.bank[0])
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])


if __name__ == "__main__":
    unittest.main()
