"""Device and dtype selection shared by every runtime script."""
from __future__ import annotations

import logging
import os

import torch

_LOG = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the preferred available accelerator in MPS, CUDA, CPU order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Return bf16 for accelerators and fp32 for CPU execution."""
    resolved = torch.device(device) if device is not None else get_device()
    return torch.bfloat16 if resolved.type in {"mps", "cuda"} else torch.float32


def configure_mps() -> None:
    """Report the optional MPS memory knob without changing user settings.

    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` must be exported before Python starts
    if it is needed, for example ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``.
    """
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        _LOG.info("MPS high-watermark ratio=%s", os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])


DEVICE = get_device()
DTYPE = get_dtype(DEVICE)
configure_mps()
_LOG.info("Resolved device=%s dtype=%s", DEVICE, DTYPE)
