"""Discriminating public-metric and submission-integrity counterexamples."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import evaluate, mention_counts
from validate_predictions import index_documents, read_hpo, validate_predictions


def entity(identifier="HP:0000001", offset=100, length=5, text="alpha", note=None):
    return {"identifier": identifier, "offset": offset, "length": length, "text": text,
            "type": "Phenotype", "note": note}


def document(pmc_id="PMC1", entities=None, phenotypes=None):
    return {"pmc_id": pmc_id, "pmid": pmc_id[3:], "patient": [{"patient_id": "P1", "mention": []}],
            "full_text": [{"offset": 100, "text": "alpha beta gamma", "type": "paragraph", "section_type": "CASE"}],
            "entities": entities if entities is not None else [entity()],
            "association": [{"patient_id": "P1", "phenotype": phenotypes if phenotypes is not None else ["HP:0000001"]}]}


class EvaluationTests(unittest.TestCase):
    def test_wrong_hpo_fails_mentions_despite_matching_span(self):
        gold = document()
        pred = copy.deepcopy(gold)
        pred["entities"][0]["identifier"] = "HP:0000002"
        result = evaluate([gold], [pred])
        self.assertEqual(result["metrics"]["menF1"], 0)
        self.assertEqual(result["metrics"]["docF1"], 0)

    def test_document_concepts_ignore_position_but_mentions_do_not(self):
        gold = document()
        pred = copy.deepcopy(gold)
        pred["entities"] = [entity(offset=106, length=4, text="beta")]
        result = evaluate([gold], [pred])
        self.assertEqual(result["metrics"]["docF1"], 1)
        self.assertEqual(result["metrics"]["menF1"], 0)

    def test_document_metric_aggregates_counts_instead_of_averaging_document_f1(self):
        first = document()
        second = document("PMC2", [entity(), entity("HP:0000002", 106, 4, "beta"), entity("HP:0000003", 111, 5, "gamma")])
        pred1 = copy.deepcopy(first)
        pred2 = copy.deepcopy(second)
        pred2["entities"] = []
        result = evaluate([first, second], [pred1, pred2])
        self.assertEqual(result["metrics"]["docF1"], 0.4)
        self.assertEqual(result["counts"]["document"]["fn"], 3)

    def test_missing_or_duplicate_documents_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([document(), document("PMC2")], [document()])
        with self.assertRaises(ValueError):
            index_documents([document(), document()])

    def test_reordered_documents_match_by_id(self):
        rows = [document(), document("PMC2")]
        self.assertEqual(evaluate(rows, list(reversed(rows)))["metrics"]["Score"], 1)

    def test_no_and_compound_are_handled(self):
        gold = document(entities=[entity("HP:0000001;HP:0000002"), entity("HP:0000003", 106, 4, "beta", "NO")])
        pred = copy.deepcopy(gold)
        pred["entities"] = [entity("HP:0000001")]
        result = evaluate([gold], [pred])
        self.assertAlmostEqual(result["metrics"]["menF1"], 2 / 3)
        pred["entities"].append(entity("HP:0000003", 106, 4, "beta"))
        result = evaluate([gold], [pred])
        self.assertEqual(result["counts"]["mention"]["fp"], 1)

    def test_unlinked_gold_can_match_one_span_but_not_cheat_linked_gold(self):
        self.assertEqual(mention_counts({(1, 3): {"-1"}}, {(1, 3): {"HP:0000001", "HP:0000002"}}), (1, 1, 0))
        self.assertEqual(mention_counts({(1, 3): {"HP:0000001"}}, {(1, 3): {"-1"}}), (0, 1, 1))

    def test_empty_patient_macro_is_one(self):
        gold = document(entities=[], phenotypes=[])
        self.assertEqual(evaluate([gold], [copy.deepcopy(gold)])["metrics"]["macF1"], 1)

    def test_patient_coverage_and_offsets_are_enforced(self):
        gold = document()
        pred = copy.deepcopy(gold)
        pred["association"] = []
        self.assertFalse(validate_predictions([gold], [pred])["valid"])
        pred = copy.deepcopy(gold)
        pred["entities"][0]["offset"] = 0
        self.assertFalse(validate_predictions([gold], [pred])["valid"])
        pred = copy.deepcopy(gold)
        pred["entities"][0]["text"] = "wrong"
        self.assertFalse(validate_predictions([gold], [pred])["valid"])

    def test_patients_with_same_id_in_different_documents_stay_separate(self):
        gold = [document(), document("PMC2", phenotypes=["HP:0000002"])]
        pred = copy.deepcopy(gold)
        pred[1]["association"][0]["phenotype"] = ["HP:0000001"]
        result = evaluate(gold, pred)
        self.assertEqual(result["metrics"]["micF1"], 0.5)
        self.assertEqual(result["metrics"]["macF1"], 0.5)

    def test_hpo_membership_obsolete_and_alt_ids(self):
        content = """format-version: 1.2
data-version: hp/releases/2026-06-23

[Term]
id: HP:0000118
name: root

[Term]
id: HP:0000001
name: child
alt_id: HP:0000009
is_a: HP:0000118 ! root

[Term]
id: HP:0000002
name: obsolete
is_a: HP:0000118 ! root
is_obsolete: true

[Term]
id: HP:0000003
name: unrelated
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hp.obo"
            path.write_text(content)
            hpo = read_hpo(path)
        self.assertEqual(hpo["allowed"], {"HP:0000118", "HP:0000001"})
        self.assertEqual(hpo["canonical"]["HP:0000009"], "HP:0000001")
        gold = document()
        self.assertTrue(validate_predictions([gold], [copy.deepcopy(gold)], hpo)["valid"])
        pred = copy.deepcopy(gold)
        pred["entities"][0]["identifier"] = "HP:0000002"
        self.assertFalse(validate_predictions([gold], [pred], hpo)["valid"])


if __name__ == "__main__":
    unittest.main()
