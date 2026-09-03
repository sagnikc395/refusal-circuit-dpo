"""Regression tests for dataset safety and preference-pair preparation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.build_dpo_dataset import pair, truncate_completion
from data.filter_sft_safety import filter_file, match_reasons
from rcdpo.refusal import refusal_reasons


class DatasetSafetyTest(unittest.TestCase):
    def test_refusal_vocabulary_is_shared(self) -> None:
        row = {"instruction": "Explain this", "input": "", "output": "I cannot help with that."}
        self.assertIn("i cannot", match_reasons(row))
        self.assertIn("i cannot", refusal_reasons(row["output"]))

    def test_filter_can_enforce_clean_sample_size(self) -> None:
        rows = [
            {"instruction": "unsafe", "input": "", "output": "I cannot help."},
            {"instruction": "safe", "input": "", "output": "Here is an answer."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.jsonl", Path(directory) / "clean.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            filter_file(source, destination, sample_size=1)
            self.assertEqual(len(destination.read_text(encoding="utf-8").splitlines()), 1)

    def test_pair_uses_final_conversation_turn(self) -> None:
        row = {
            "chosen": "\n\nHuman: first\n\nAssistant: old\n\nHuman: final\n\nAssistant: safe",
            "rejected": "\n\nHuman: first\n\nAssistant: old\n\nHuman: final\n\nAssistant: unsafe",
        }
        self.assertEqual(pair(row), ("final", "safe", "unsafe"))

    def test_truncation_preserves_prompt(self) -> None:
        class Tokenizer:
            def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
                tokens = text.split()
                if truncation:
                    tokens = tokens[:max_length]
                return {"input_ids": tokens}

            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(tokens)

        tokenizer = Tokenizer()
        result = truncate_completion(tokenizer, "prompt", "one two three", max_length=3)
        self.assertEqual(result, "one two")


if __name__ == "__main__":
    unittest.main()
