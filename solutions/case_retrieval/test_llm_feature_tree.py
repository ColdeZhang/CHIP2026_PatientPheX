"""Identity/leakage/failed-answer controls; no tree fitting or gold files."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import llm_feature_tree as tree


def request(row_id="a", pmc="d1", concept="HP:0000001", patients=None):
    return dict(row_id=row_id, pmc_id=pmc, code_map={"C000": concept},
                patient_ids=patients or ["P1", "P2"],
                messages=[{"role": "system", "content": "schema"},
                          {"role": "user", "content": "unlabelled input"}])


def response(prompt, value, **kwargs):
    return dict(row_id=prompt["row_id"], pmc_id=prompt["pmc_id"],
                response=value, hit_token_limit=False, removed_demonstrations=0,
                input_tokens=10, output_tokens=5, **kwargs)


class PairedTreeControls(unittest.TestCase):
    def test_local_cid_and_response_order_cannot_cross_documents(self):
        prompts = [request(), request("b", "d2", "unlinked text")]
        raw = [response(prompts[1], '{"C000":["P2"]}'),
               response(prompts[0], '{"C000":["P1"]}')]
        pairs = tree.pair_rows(prompts)
        flags, _ = tree.llm_features(prompts, raw, pairs)
        self.assertEqual(flags[:, 0].tolist(), [1, 0, 0, 1])
        self.assertEqual([p["concept"] for p in pairs],
                         ["HP:0000001", "HP:0000001", "unlinked text", "unlinked text"])

    def test_empty_failed_and_other_patient_are_different(self):
        p = request()
        cases = [(' {"C000":[]} ', [0, 1, 0, 0]),
                 ('{"C000":["P2"]}', [0, 0, 0, 1]),
                 ('{"HPA:0000001":[]}', [0, 0, 1, 0]),
                 ('unparsable', [0, 0, 1, 0]),
                 ('{"C000":["UNKNOWN"]}', [0, 0, 1, 0]),
                 ('{"C000":[{"patient_id":"P1","evidence":"quote"}]}', [0, 0, 1, 0])]
        for text, expected in cases:
            with self.subTest(text=text):
                flags, _ = tree.llm_features([p], [response(p, text)], tree.pair_rows([p]))
                self.assertEqual(flags[0].tolist(), expected)

    def test_numeric_patient_ids_and_duplicate_selections_match_original_merger(self):
        p = request(patients=["1", "2"])
        r = response(p, '```json\n{"C000":[1,"1"],"EXTRA":["unknown"]}\n```')
        flags, counts = tree.llm_features([p], [r], tree.pair_rows([p]))
        self.assertEqual(flags.tolist(), [[1, 0, 0, 1], [0, 0, 0, 1]])
        self.assertEqual(counts["valid_candidate_decisions"], 1)
        r["hit_token_limit"] = True
        failed, _ = tree.llm_features([p], [r], tree.pair_rows([p]))
        self.assertEqual(failed.tolist(), [[0, 0, 1, 0], [0, 0, 1, 0]])

    def test_missing_duplicate_or_wrong_document_response_rejected(self):
        p, q = request(), request("b", "d2")
        r = response(p, '{"C000":[]}')
        for rows in ([r], [r, r]):
            with self.assertRaises(ValueError):
                tree.align_responses([p, q], rows)
        r["pmc_id"] = "d2"
        with self.assertRaisesRegex(ValueError, "pmc_id"):
            tree.align_responses([p], [r])

    def test_answer_not_in_request_and_cached_labels_must_match(self):
        p = request()
        source = dict(p, answer='{"C000":["P1"]}')
        with self.assertRaises(ValueError):
            tree.keyed_requests([source])
        pairs = tree.pair_rows([p], [source])
        cache = [dict(pmc_id="d1", patient_id="P1", concept="HP:0000001",
                      features={"x": 1}, target=False),
                 dict(pmc_id="d1", patient_id="P2", concept="HP:0000001",
                      features={"x": 2}, target=False)]
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            tree.join_features(pairs, cache, True)
        with self.assertRaisesRegex(ValueError, "Validation target"):
            tree.join_features(tree.pair_rows([p]), cache, False)

    def test_validation_columns_cannot_expand_training_and_base_prefix_is_identical(self):
        vec, x, z = tree.base_matrices([{"x": 1.0}, {"x": 2.0}],
                                      [{"x": 3.0, "validation_only": 999.0}])
        self.assertEqual(vec.feature_names_, ["x"])
        self.assertEqual(z.tolist(), [[3.0]])
        augmented = tree.append_llm(x, np.asarray([[1, 0, 0, 1], [0, 1, 0, 0]]))
        np.testing.assert_array_equal(augmented[:, :x.shape[1]], x)
        with self.assertRaisesRegex(ValueError, "shape"):
            tree.append_llm(x, np.zeros((1, 4)))

    def test_predictions_keep_entities_all_patients_and_freeze_both_arms(self):
        p = request()
        doc = dict(pmc_id="d1", patient=[{"patient_id": "P1"}, {"patient_id": "P2"}])
        entity = dict(pmc_id="d1", entities=[{"text": "abc", "offset": 0}],
                      association=[{"patient_id": "P1", "phenotype": ["old"]}])
        pred = tree.construct_predictions([doc], [entity], tree.pair_rows([p]), np.asarray([.5, .49]))
        self.assertEqual(pred[0]["entities"], entity["entities"])
        self.assertEqual(pred[0]["association"], [
            {"patient_id": "P1", "phenotype": ["HP:0000001"]},
            {"patient_id": "P2", "phenotype": []}])
        self.assertEqual(entity["association"][0]["phenotype"], ["old"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(ValueError):
                tree.freeze_predictions(out, {"without_llm": pred})
            self.assertFalse(list(out.iterdir()))
            freeze = tree.freeze_predictions(out, {"without_llm": pred, "with_llm": pred})
            self.assertFalse(freeze["heldout_gold_read"])
            for item in freeze["files"].values():
                self.assertEqual(item["sha256"], tree.sha(item["path"]))

    def test_manifest_rejects_adapter_wrong_prompts_partial_rows_or_weights(self):
        p = request()
        r = response(p, '{"C000":[]}')
        source = {"files": [{"name": str(i), "size": i, "sha256": "sha" + str(i)} for i in range(15)]}
        with tempfile.TemporaryDirectory() as directory:
            prompts, responses = Path(directory)/"p.jsonl", Path(directory)/"r.jsonl"
            prompts.write_text(json.dumps(p) + "\n"); responses.write_text(json.dumps(r) + "\n")
            manifest = dict(fingerprint=dict(configuration=dict(adapter=None, max_input_tokens=14000,
                    max_new_tokens=1600), source_sha256=tree.RUNNER_SHA, prompts_sha256=tree.sha(prompts),
                    source_manifest_sha256=tree.PINS["model_source"], batch_size=1, thinking=False,
                    precision="text_only_bfloat16_sdpa_no_quantization",
                    model_files={x["name"]:{"bytes":x["size"],"sha256":x["sha256"]} for x in source["files"]},
                    runtime=dict(transformers="5.20.2", peft="0.21.2", tokenizers="0.23.1")),
                output_sha256=tree.sha(responses), expected_rows=1, base_parameters=8953803264,
                total_parameters=8953803264, eos_token_ids=[248044,248046],
                load_information={k:[] for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")})
            tree.verify_response_manifest(manifest, responses, prompts, [r], source)
            for kind in ("adapter", "prompt", "partial", "weights"):
                changed = copy.deepcopy(manifest)
                if kind == "adapter": changed["fingerprint"]["configuration"]["adapter"] = "trained-on48"
                elif kind == "prompt": changed["fingerprint"]["prompts_sha256"] = "wrong"
                elif kind == "partial": changed["expected_rows"] = 2
                else: changed["fingerprint"]["model_files"]["0"]["sha256"] = "wrong"
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    tree.verify_response_manifest(changed, responses, prompts, [r], source)


if __name__ == "__main__":
    unittest.main()
