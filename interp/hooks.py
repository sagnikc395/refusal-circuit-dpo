"""Raw Hugging Face hooks for Qwen2 residual, attention, and MLP points."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

Point = tuple[int, str]
_COMPONENTS = {"residual": "", "attention": "self_attn", "mlp": "mlp"}


def _module_for(model: nn.Module, point: Point) -> nn.Module:
    layer, component = point
    try:
        block = model.model.layers[layer]
    except AttributeError as exc:
        raise ValueError("Expected a Qwen-style model.model.layers structure") from exc
    if component == "residual":
        return block
    if component not in _COMPONENTS:
        raise ValueError(f"Unknown component {component!r}")
    return getattr(block, _COMPONENTS[component])


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output

@dataclass
class _Hooks:
    handles: list[Any]
    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

class CacheActivations:
    """Cache detached outputs keyed by ``(layer, component)``."""
    def __init__(self, model: nn.Module, points: list[Point]):
        self.model, self.points, self.activations = model, points, {}
        self._hooks = _Hooks([])
    def __enter__(self):
        for point in self.points:
            def capture(_module, _inputs, output, point=point):
                self.activations[point] = _output_tensor(output).detach().clone()
            self._hooks.handles.append(_module_for(self.model, point).register_forward_hook(capture))
        return self.activations
    def __exit__(self, *_):
        self._hooks.close()

class PatchActivations:
    """Substitute supplied activations for selected module outputs."""
    def __init__(self, model: nn.Module, patches: Mapping[Point, torch.Tensor]):
        self.model, self.patches = model, patches
        self._hooks = _Hooks([])
    def __enter__(self):
        for point, value in self.patches.items():
            def patch(_module, _inputs, output, value=value):
                if isinstance(output, tuple):
                    return (value,) + output[1:]
                return value
            self._hooks.handles.append(_module_for(self.model, point).register_forward_hook(patch))
        return self
    def __exit__(self, *_):
        self._hooks.close()

class AddVector:
    """Add ``alpha * vec`` at the final prompt position of an activation."""
    def __init__(self, model: nn.Module, point: Point, vec: torch.Tensor, alpha: float):
        self.model, self.point, self.vec, self.alpha = model, point, vec, alpha
        self._hooks = _Hooks([])
    def __enter__(self):
        def add(_module, _inputs, output):
            original = _output_tensor(output)
            value = original.clone()
            vector = self.alpha * self.vec.to(device=original.device, dtype=original.dtype)
            value[:, -1, :] = value[:, -1, :] + vector
            return (value,) + output[1:] if isinstance(output, tuple) else value
        self._hooks.handles.append(_module_for(self.model, self.point).register_forward_hook(add))
        return self
    def __exit__(self, *_):
        self._hooks.close()
