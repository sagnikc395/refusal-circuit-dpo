"""Tests for the evaluation acceptance gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.gate import check


class GateTest(unittest.TestCase):
    def _artifact(self, harmful: list[bool], benign: list[bool]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "results.jsonl"
        rows = [
            {"id": f"h-{index}", "split": "harmful", "refusal": value}
            for index, value in enumerate(harmful)
        ] + [
            {"id": f"b-{index}", "split": "benign", "refusal": value}
            for index, value in enumerate(benign)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_thresholds_are_strict(self) -> None:
        self.assertFalse(check(self._artifact([True] * 4 + [False], [False] * 9 + [True])))
        self.assertTrue(check(self._artifact([True] * 5, [False] * 10)))

    def test_missing_split_fails(self) -> None:
        path = self._artifact([True], [])
        with self.assertRaises(ValueError):
            check(path)


if __name__ == "__main__":
    unittest.main()
