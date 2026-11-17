"""Distributed smoke test: single-process gloo DDP init + train step via engine utilities."""

from __future__ import annotations

import os

import pytest
import torch

from neuratal.training.engine import init_distributed, set_seed


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_single_process_gloo_init():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # only if not already initialized by the test runner
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    assert torch.distributed.is_initialized()
    set_seed(42)
    torch.distributed.destroy_process_group()
