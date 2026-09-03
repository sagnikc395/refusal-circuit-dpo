"""Regression tests for raw-hook intervention semantics."""
from __future__ import annotations

import torch
from torch import nn
import unittest

from interp.activation_patching import normalized_disruption
from interp.hooks import AddVector, CacheActivations, PatchActivations


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return (self.mlp(self.self_attn(x)),)


class ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block()])
        self.head = nn.Linear(4, 5, bias=False)

    def forward(self, x):
        return self.head(self.model.layers[0](x)[0])


class HooksTest(unittest.TestCase):
    def test_patching_cached_activation_is_a_noop(self) -> None:
        torch.manual_seed(0)
        model, inputs = ToyQwen(), torch.randn(2, 3, 4)
        expected = model(inputs)
        with CacheActivations(model, [(0, "residual")]) as cache:
            model(inputs)
        with PatchActivations(model, {(0, "residual"): cache[(0, "residual")]}):
            actual = model(inputs)
        torch.testing.assert_close(actual, expected)

    def test_vector_changes_only_last_position(self) -> None:
        torch.manual_seed(1)
        model, inputs, vector = ToyQwen(), torch.randn(1, 3, 4), torch.ones(4)
        expected = model(inputs)
        with AddVector(model, (0, "residual"), vector, 1.0):
            actual = model(inputs)
        torch.testing.assert_close(actual[:, :-1], expected[:, :-1])
        self.assertFalse(torch.equal(actual[:, -1], expected[:, -1]))

    def test_all_qwen_components_are_hookable(self) -> None:
        model = ToyQwen()
        for component in ("residual", "attention", "mlp"):
            with CacheActivations(model, [(0, component)]) as cache:
                model(torch.randn(1, 2, 4))
            self.assertEqual(cache[(0, component)].shape, (1, 2, 4))

    def test_vector_can_be_applied_once_for_generation(self) -> None:
        torch.manual_seed(2)
        model, inputs, vector = ToyQwen(), torch.randn(1, 3, 4), torch.ones(4)
        baseline = model(inputs)
        with AddVector(model, (0, "residual"), vector, 1.0, apply_once=True):
            first = model(inputs)
            second = model(inputs)
        self.assertFalse(torch.equal(first, baseline))
        torch.testing.assert_close(second, baseline)

    def test_normalized_disruption_is_scale_invariant(self) -> None:
        self.assertAlmostEqual(normalized_disruption(10, 7, 0), 0.3)
        self.assertAlmostEqual(normalized_disruption(100, 70, 0), 0.3)
        self.assertEqual(normalized_disruption(4, 3, 4), 1.0)

    def test_cache_hooks_are_removed_after_forward_exception(self) -> None:
        model = ToyQwen()
        with self.assertRaises(RuntimeError):
            with CacheActivations(model, [(0, "residual")]):
                raise RuntimeError("forward failed")
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
