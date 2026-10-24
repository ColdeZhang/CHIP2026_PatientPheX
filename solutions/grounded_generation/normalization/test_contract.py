"""Semantic boundary checks; no held labels, model loading or quality evaluation."""
import os
import unittest

from .strict import HPO, StrictNormalizer, verify_span
from .public_bank import public_rows


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hpo = HPO(os.environ["GROUNDED_HPO"])
        cls.normalizer = StrictNormalizer(os.environ["GROUNDED_HPO"], os.environ["GROUNDED_TRAIN32"])

    def test_valid_compound_dedup_and_whitespace(self):
        value = " HP:0001250 ; HP:0001251;HP:0001250 "
        self.assertEqual(self.hpo.entity_keys(value, "irrelevant"), self.hpo.relation_keys(value))
        self.assertEqual(self.hpo.identifiers(value), ("HP:0001250", "HP:0001251"))

    def test_compound_invalid_component_never_partially_maps(self):
        for value in ["HP:0001250;-1", "HP:0001250;HP:9999999", "HP:0001250;MESH:123", "HP:0001250;"]:
            with self.assertRaises(ValueError):
                self.hpo.identifiers(value)

    def test_natural_semicolon_relation_is_literal_nil(self):
        phrase = "low serum cholesterol; elevated triglycerides"
        self.assertEqual(self.hpo.relation_keys(phrase), (("nil", phrase),))

    def test_nil_typed_join_uses_text(self):
        self.assertEqual(self.hpo.entity_keys("-1", "unusual finding"), self.hpo.relation_keys("unusual finding"))
        with self.assertRaises(ValueError):
            self.hpo.relation_keys("-1")

    def test_alt_id_is_canonical(self):
        alt, canonical = next((a, c) for a, c in self.hpo.canonical.items() if a != c)
        self.assertEqual(self.hpo.identifiers(alt), (canonical,))
        self.assertEqual(self.hpo.entity_keys(alt, "alias"), self.hpo.relation_keys(canonical))

    def test_obsolete_and_outside_branch_rejected(self):
        for hid in ["HP:0000001", "HP:0031135", "HP:9999999"]:
            with self.assertRaises(ValueError):
                self.hpo.identifiers(hid)

    def test_unmatched_is_abstention_not_nil(self):
        result = self.normalizer.normalize("zzzz_unmapped_observation_zzzz")
        self.assertEqual(result["status"], "abstain")
        self.assertIsNone(result["identifier"])

    def test_alias_whole_span_not_substring(self):
        self.assertEqual(self.normalizer.normalize("Seizure")["ids"], ["HP:0001250"])
        self.assertEqual(self.normalizer.normalize("an entirely novel modifier Seizure")["status"], "abstain")

    def test_casefold_conflict_not_hidden_by_literal_priority(self):
        key, options = next((key, options) for key, options in self.normalizer.folded.items() if len(options) > 1)
        row = next(iter(options.values()))[0]
        result = self.normalizer.normalize(row["text"])
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["reason"], "ambiguous_mapping")

    def test_nil_requires_literal_evidence(self):
        surface = next(key for key, options in self.normalizer.exact.items()
                       if list(options) == [("-1",)] and len(self.normalizer.folded[" ".join(key.casefold().split())]) == 1)
        result = self.normalizer.normalize(surface)
        self.assertEqual(result["status"], "nil")
        self.assertEqual(result["identifier"], "-1")

    def test_grounding_global_offset_and_unicode(self):
        doc = {"full_text": [{"offset": 100, "text": "α Seizure."}], "pmc_id": "synthetic"}
        self.assertEqual(verify_span(doc, "Seizure", 102), 0)
        with self.assertRaises(ValueError):
            verify_span(doc, "Seizure", 103)

    def test_grounding_rejects_duplicate_passages(self):
        doc = {"full_text": [{"offset": 100, "text": "Seizure"}] * 2}
        with self.assertRaises(ValueError):
            verify_span(doc, "Seizure", 100)

    def test_training_hash_guard(self):
        with self.assertRaises(ValueError):
            StrictNormalizer(os.environ["GROUNDED_HPO"], os.environ["GROUNDED_TRAIN32"], expected_train_sha256="0" * 64)

    def test_public_rows_never_contain_nil_or_compound_labels(self):
        rows = public_rows(self.hpo)
        self.assertTrue(rows)
        self.assertTrue(all(hid in self.hpo.allowed and hid != "HP:0000118" for row in rows for hid in row["identifiers"]))
        self.assertEqual([row["text"] for row in rows], sorted(set(row["text"] for row in rows)))
        self.assertTrue(all(row["provenance"] for row in rows))


if __name__ == "__main__":
    unittest.main()
