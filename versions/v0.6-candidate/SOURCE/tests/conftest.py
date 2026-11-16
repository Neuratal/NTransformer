"""Shared fixtures.

Every test uses the full 262,144-token NTokenizer vocabulary (per project
requirement — no reduced-vocabulary shortcuts). Only layer count and width
vary, which is exactly what the architecture is parameterized by.
"""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    yield


@pytest.fixture(params=DEVICES)
def device(request) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture()
def dev_config() -> NeuratalConfig:
    """The approved development/testing preset (full NTokenizer vocabulary)."""
    return NeuratalConfig()
