"""CPU-only behavior checks for fold isolation, answer masking and resume guards."""
import json
from pathlib import Path
import tempfile
import unittest

from train_lora import batch_plan, encode_example, load_training_rows, verify_fingerprint


class TinyTokenizer:
    """Small deterministic chat serializer, not a replacement for Qwen token audits."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        text = "".join("<" + x["role"] + ">" + x["content"] + "</end>" for x in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


class WrongPrefixTokenizer(TinyTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        output = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"]:
            output[0] += 1
        return output


def example(identifier="1", row="1:chunk:0", answer='["seizures"]'):
    return {"pmc_id": identifier, "row_id": row,
            "messages": [{"role": "system", "content": "Extract"},
                         {"role": "user", "content": "TEXT: seizures"}], "answer": answer}


class LoraDataTests(unittest.TestCase):
    def test_answer_only_loss_and_empty_answer(self):
        for answer in ('["seizures"]', '[]'):
            result = encode_example(example(answer=answer), TinyTokenizer(), 500)
            n = result["prefix_tokens"]
            self.assertTrue(all(x == -100 for x in result["labels"][:n]))
            self.assertEqual(result["labels"][n:], result["input_ids"][n:])
            self.assertIn(ord('['), result["labels"][n:])
            self.assertGreater(result["supervised_tokens"], 0)

    def test_no_silent_truncation_or_mismatched_template(self):
        with self.assertRaisesRegex(ValueError, "No truncation"):
            encode_example(example(), TinyTokenizer(), 10)
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            encode_example(example(), WrongPrefixTokenizer(), 500)

    def test_document_isolation_not_chunk_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, heldout = Path(temporary) / "train.jsonl", Path(temporary) / "input.jsonl"
            train.write_text(json.dumps(example()) + "\n" + json.dumps(example(row="1:chunk:1")) + "\n")
            heldout.write_text(json.dumps({"pmc_id": "2", "entities": [], "association": []}) + "\n")
            rows, documents = load_training_rows(train, heldout)
            self.assertEqual((len(rows), documents), (2, ["1"]))
            heldout.write_text(json.dumps({"pmc_id": "1"}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_training_rows(train, heldout)

    def test_excluded_gold_and_duplicate_training_rows_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, heldout = Path(temporary) / "train.jsonl", Path(temporary) / "input.jsonl"
            train.write_text(json.dumps(example()) + "\n")
            heldout.write_text(json.dumps({"pmc_id": "2", "association": [{"patient_id": "P1"}]}) + "\n")
            with self.assertRaisesRegex(ValueError, "labels"):
                load_training_rows(train, heldout)
            heldout.write_text(json.dumps({"pmc_id": "2"}) + "\n")
            train.write_text((json.dumps(example()) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_training_rows(train, heldout)

    def test_batches_cover_each_example_and_resume_epoch_order(self):
        plan = batch_plan(19, 2, 4, 20261023, 1)
        flat = [index for group in plan for batch in group for index in batch]
        self.assertEqual(sorted(flat), list(range(19)))
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan, batch_plan(19, 2, 4, 20261023, 1))
        self.assertNotEqual(plan, batch_plan(19, 2, 4, 20261023, 0))

    def test_smoke_longest_and_formal_fingerprint_separation(self):
        self.assertEqual(batch_plan(3, 2, 4, 7, 0, True, [5, 9, 6]), [[[1]]])
        verify_fingerprint({"smoke": False, "sha": "a"}, {"smoke": False, "sha": "a"})
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_fingerprint({"smoke": True, "sha": "a"}, {"smoke": False, "sha": "a"})
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_fingerprint({"smoke": False, "sha": "a"}, {"smoke": False, "sha": "b"})


if __name__ == "__main__":
    unittest.main()
