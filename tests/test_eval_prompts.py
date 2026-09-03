"""Offline tests for held-out evaluation prompt selection."""
from __future__ import annotations

import unittest

from data.build_eval_prompts import normalize, select_prompts, training_prompts
from evaluation.eval_refusal import artifact_name, summarize


class EvalPromptTest(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_training_prompts(self) -> None:
        rows = [{"prompt": "Train me"}, {"prompt": "Held out one"}, {"prompt": "Held out two"}]
        first = select_prompts(rows, lambda row: row["prompt"], 2, "source", {normalize("Train me")}, 42)
        second = select_prompts(rows, lambda row: row["prompt"], 2, "source", {normalize("Train me")}, 42)
        self.assertEqual(first, second)
        self.assertEqual([row["source"] for row in first], ["source", "source"])
        self.assertNotIn("train me", {normalize(row["prompt"]) for row in first})

    def test_result_names_are_unique_and_summaries_require_both_splits(self) -> None:
        self.assertEqual(artifact_name("Qwen/Qwen2.5-0.5B", 2), "02_Qwen2.5-0.5B")
        self.assertEqual(
            summarize([
                {"split": "harmful", "refusal": True},
                {"split": "benign", "refusal": False},
            ]),
            (1.0, 1.0),
        )
        with self.assertRaises(ValueError):
            summarize([{"split": "harmful", "refusal": True}])

    def test_training_prompts_strips_serialized_template(self) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps({"prompt": "### Instruction:\nExample\n\n### Response:\n"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(training_prompts((path,)), {"example"})


if __name__ == "__main__":
    unittest.main()
