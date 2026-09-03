"""Raw Hugging Face hooks for Qwen2 residual, attention, and MLP points.

Points target the decoder layer at ``model.model.layers[layer]`` for a raw
Qwen model and the equivalent base-model path for a PEFT wrapper. Residual
points observe the decoder block output; attention and MLP points observe the
corresponding submodule outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

Point = tuple[int, str]
_COMPONENTS = {"residual": "", "attention": "self_attn", "mlp": "mlp"}


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find Qwen's decoder layers on raw, PEFT, and common wrapper models."""
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
    )
    for path in candidates:
        current: Any = model
        try:
            for name in path:
                current = getattr(current, name)
        except AttributeError:
            continue
        if isinstance(current, (nn.ModuleList, list, tuple)):
            return current
    raise ValueError("Expected a Qwen-style model with decoder layers")


def _module_for(model: nn.Module, point: Point) -> nn.Module:
    layer, component = point
    if layer < 0:
        raise ValueError(f"Layer must be non-negative, got {layer}")
    layers = decoder_layers(model)
    if layer >= len(layers):
        raise ValueError(f"Layer {layer} is outside model with {len(layers)} layers")
    block = layers[layer]
    if component == "residual":
        return block
    if component not in _COMPONENTS:
        raise ValueError(f"Unknown component {component!r}; expected one of {tuple(_COMPONENTS)}")
    return getattr(block, _COMPONENTS[component])


def _output_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim < 3:
        raise TypeError("Qwen hook outputs must contain a [batch, sequence, hidden] tensor")
    return value


def _replace_output(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value,) + output[1:]
    return value


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

    def __enter__(self) -> dict[Point, torch.Tensor]:
        try:
            for point in self.points:
                def capture(_module, _inputs, output, point=point):
                    self.activations[point] = _output_tensor(output).detach().clone()

                self._hooks.handles.append(_module_for(self.model, point).register_forward_hook(capture))
        except BaseException:
            self._hooks.close()
            raise
        return self.activations

    def __exit__(self, *_: Any) -> None:
        self._hooks.close()


class PatchActivations:
    """Substitute supplied activations for selected module outputs."""

    def __init__(self, model: nn.Module, patches: Mapping[Point, torch.Tensor]):
        self.model, self.patches = model, patches
        self._hooks = _Hooks([])

    def __enter__(self) -> "PatchActivations":
        try:
            for point, value in self.patches.items():
                def patch(_module, _inputs, output, value=value):
                    original = _output_tensor(output)
                    if value.shape != original.shape:
                        raise ValueError(f"Patch shape {tuple(value.shape)} does not match output {tuple(original.shape)}")
                    return _replace_output(output, value.to(device=original.device, dtype=original.dtype))

                self._hooks.handles.append(_module_for(self.model, point).register_forward_hook(patch))
        except BaseException:
            self._hooks.close()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self._hooks.close()


class AddVector:
    """Add ``alpha * vec`` at the final prompt position of an activation."""

    def __init__(self, model: nn.Module, point: Point, vec: torch.Tensor, alpha: float, *, apply_once: bool = False):
        self.model, self.point, self.vec, self.alpha = model, point, vec, alpha
        self.apply_once = apply_once
        self._applied = False
        self._hooks = _Hooks([])

    def __enter__(self) -> "AddVector":
        try:
            def add(_module, _inputs, output):
                if self.apply_once and self._applied:
                    return output
                original = _output_tensor(output)
                value = original.clone()
                vector = self.alpha * self.vec.to(device=original.device, dtype=original.dtype)
                if vector.ndim != 1 or vector.shape[0] != value.shape[-1]:
                    raise ValueError(f"Steering vector must have shape [{value.shape[-1]}], got {tuple(vector.shape)}")
                value[:, -1, :] = value[:, -1, :] + vector
                self._applied = True
                return _replace_output(output, value)

            self._hooks.handles.append(_module_for(self.model, self.point).register_forward_hook(add))
        except BaseException:
            self._hooks.close()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self._hooks.close()
