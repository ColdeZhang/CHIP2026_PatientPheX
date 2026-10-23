"""CPU checks for expensive inference recovery and inference-label separation."""
import json
from pathlib import Path
import tempfile
import unittest

import qwen35_runner as runner


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "responses.jsonl"
        self.checkpoint = self.output.with_suffix(".checkpoint.json")
        self.fingerprint = {"weights": "same", "prompts": "fixed"}
        self.checkpoint.write_text(json.dumps(self.fingerprint))
        self.rows = [{"row_id": "paper:window:1", "pmc_id": "paper"}]
        self.record = {**self.rows[0], "response": "{}"}

    def test_partial_completion_resumes_known_row(self):
        self.output.write_text(json.dumps(self.record) + "\n")
        self.assertEqual(runner.validate_existing(self.output, self.checkpoint,
                                                  self.fingerprint, self.rows),
                         {"paper:window:1"})

    def test_changed_weights_are_not_reused(self):
        self.output.write_text(json.dumps(self.record) + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.validate_existing(self.output, self.checkpoint, {"weights": "new"}, self.rows)

    def test_trailing_partial_write_is_preserved_and_rejected(self):
        value = json.dumps(self.record) + '\n{"row_id":'
        self.output.write_text(value)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner.validate_existing(self.output, self.checkpoint, self.fingerprint, self.rows)
        self.assertEqual(self.output.read_text(), value)

    def test_duplicate_or_wrong_patient_does_not_count_as_complete(self):
        for records in ([self.record, self.record], [{**self.record, "pmc_id": "different"}]):
            self.output.write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaises(ValueError):
                runner.validate_existing(self.output, self.checkpoint, self.fingerprint, self.rows)

    def test_training_answer_is_not_accepted_as_inference_prompt(self):
        row = {**self.rows[0], "messages": [{"role": "user", "content": "Input"}], "answer": "[]"}
        self.output.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "training answers"):
            runner.read_prompts(self.output)


if __name__ == "__main__":
    unittest.main()
