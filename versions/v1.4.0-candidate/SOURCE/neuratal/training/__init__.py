"""Neuratal training engine — data, optimization, scheduling, state management.

Scope: engine is independent of model family and provides:
- dataset protocol with sequence packing and shuffling
- AdamW optimizer factory (pluggable; Muon/Lion can be added later)
- warmup + cosine LR schedule driven by global step
- micro-batch gradient accumulation
- gradient clipping at the correct point (after accumulation)
- mixed precision (bf16 preferred; fp16 fallback)
- full training-state checkpointing (model + optimizer + scheduler + RNG + step)
- resume that truly restores all state

No global state, no training run is started by this module — only by callers.
"""

from __future__ import annotations

from neuratal.training.chunked_loss import chunked_cross_entropy, chunked_cross_entropy_batch

__all__ = ["chunked_cross_entropy", "chunked_cross_entropy_batch"]
