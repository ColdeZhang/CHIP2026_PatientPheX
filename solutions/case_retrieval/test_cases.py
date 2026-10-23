"""Small behavioural checks for boundaries, negative evidence, and patient identity."""
import unittest

from run import LexicalMemory, normalize_id, patient_pair, mentions_by_patient


class Cases(unittest.TestCase):
    def test_association_units_preserve_non_hpo_text(self):
        from run import association_ids
        self.assertEqual(association_ids(["HP:0000252;HP:0001250", "some; text", "MESH:X;Y", "HP:123"]), {"HP:0000252", "HP:0001250", "some; text", "MESH:X;Y", "HP:123"})

    def setUp(self):
        self.terms = {"HP:0000508": {"aliases": ["ptosis"]}, "HP:0001250": {"aliases": ["seizures"]}}
        self.canonical = {key: key for key in self.terms}
        self.canonical["HP:9999999"] = "HP:0000508"
        self.memory = LexicalMemory(self.terms, set(self.terms), self.canonical, [])

    def test_word_boundary_and_global_offset(self):
        text = "Apoptosis was measured; ptosis was present."
        doc = {"full_text": [{"text": text, "offset": 400}]}
        entities = self.memory.predict(doc)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["offset"], 400 + text.rindex("ptosis was"))
        self.assertEqual(entities[0]["text"], "ptosis")

    def test_local_negation_and_new_sentence(self):
        doc = {"full_text": [{"text": "No ptosis. Seizures developed later.", "offset": 0}]}
        self.assertEqual([e["identifier"] for e in self.memory.predict(doc)], ["HP:0001250"])

    def test_compound_canonicalization(self):
        self.assertEqual(normalize_id("HP:9999999;HP:0001250", self.canonical), "HP:0000508;HP:0001250")
        self.assertIsNone(normalize_id("HP:0000508;HP:1111111", self.canonical))

    def test_patient_specific_retrieval_features(self):
        text = "Patient A had ptosis. Later, Patient B was examined."
        doc = {"patient": [
            {"patient_id": "A", "mention": [{"offset": 500, "length": 9}]},
            {"patient_id": "B", "mention": [{"offset": 500 + text.index("Patient B"), "length": 9}]}],
            "full_text": [{"offset": 500, "text": text, "section_type": "CASE"}]}
        entity = {"identifier": "HP:0000508", "offset": 500+text.index("ptosis"), "length": 6, "text": "ptosis"}
        mentions = mentions_by_patient(doc)
        _, features_a = patient_pair(doc, "A", [entity], mentions)
        _, features_b = patient_pair(doc, "B", [entity], mentions)
        self.assertIn("target_same_sentence=True", features_a)
        self.assertIn("target_same_sentence=False", features_b)
        self.assertIn("target_closest=True", features_a)
        self.assertIn("other_closest=True", features_b)


if __name__ == "__main__":
    unittest.main()
