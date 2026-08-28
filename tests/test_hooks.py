"""Regression tests for raw-hook intervention semantics."""
from __future__ import annotations

import torch
from torch import nn
import unittest

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
