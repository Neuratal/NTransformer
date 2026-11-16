"""KVCache: lazy init, views, overflow, reset, batch isolation, commit protocol."""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig
from neuratal.model.cache import CacheOverflowError, KVCache


def make_cache(capacity: int | None = 8, **kwargs) -> KVCache:
    cfg = NeuratalConfig(cache_capacity=capacity, max_position_embeddings=64, **kwargs)
    return KVCache(cfg)


def fake_kv(cache: KVCache, batch=1, tokens=1, layer_heads=2, head_dim=64, device="cpu", dtype=torch.float32):
    return (
        torch.randn(batch, layer_heads, tokens, head_dim, device=device, dtype=dtype),
        torch.randn(batch, layer_heads, tokens, head_dim, device=device, dtype=dtype),
    )


class TestInitialization:
    def test_lazy_init_on_first_update(self):
        cache = make_cache(capacity=8)
        assert not cache.is_initialized
        k, v = fake_kv(cache, tokens=3)
        cache.update(0, k, v)
        assert cache.is_initialized
        assert cache.batch_size == 1
        # both layers allocated
        assert len(cache._keys) == 2

    def test_views_return_full_prefix(self):
        cache = make_cache(capacity=8)
        k1, v1 = fake_kv(cache, tokens=3)
        k2, v2 = fake_kv(cache, tokens=2)
        kf, vf = cache.update(0, k1, v1)
        assert kf.shape[2] == 3
        cache.commit(3)
        kf2, _ = cache.update(0, k2, v2)
        cache.commit(2)
        assert kf2.shape[2] == 5
        # original data preserved
        assert torch.equal(kf2[:, :, :3], k1)
        assert torch.equal(kf2[:, :, 3:], k2)

    def test_buffered_zeros_before_write(self):
        cache = make_cache(capacity=4)
        k, v = fake_kv(cache, tokens=1)
        cache.update(0, k, v)
        cache.commit(1)
        assert torch.equal(cache._keys[0][:, :, 1:], torch.zeros_like(cache._keys[0][:, :, 1:]))


class TestCommitProtocol:
    def test_commit_without_update_raises(self):
        cache = make_cache()
        with pytest.raises(RuntimeError, match="without a preceding update"):
            cache.commit(1)

    def test_commit_mismatch_raises(self):
        cache = make_cache()
        k, v = fake_kv(cache, tokens=3)
        cache.update(0, k, v)
        with pytest.raises(RuntimeError, match="does not match"):
            cache.commit(1)

    def test_commit_advances_length(self):
        cache = make_cache()
        k, v = fake_kv(cache, tokens=2)
        cache.update(0, k, v)
        assert cache.get_seq_length() == 0  # not yet committed
        cache.commit(2)
        assert cache.get_seq_length() == 2

    def test_multi_layer_same_window(self):
        cache = make_cache()
        k0, v0 = fake_kv(cache, tokens=2)
        k1, v1 = fake_kv(cache, tokens=2)
        cache.update(0, k0, v0)
        cache.update(1, k1, v1)
        cache.commit(2)
        assert cache.get_seq_length() == 2
        # next write lands at offset 2 in both layers
        k0b, v0b = fake_kv(cache, tokens=1)
        kf, _ = cache.update(0, k0b, v0b)
        assert kf.shape[2] == 3
        cache.commit(1)
        assert torch.equal(cache._keys[0][:, :, 2], k0b[:, :, 0])


class TestOverflow:
    def test_overflow_raises_and_does_not_wrap(self):
        cache = make_cache(capacity=3)
        k, v = fake_kv(cache, tokens=3)
        cache.update(0, k, v)
        cache.commit(3)
        k2, v2 = fake_kv(cache, tokens=1)
        with pytest.raises(CacheOverflowError, match="capacity"):
            cache.update(0, k2, v2)

    def test_partial_fit_rejected_entirely(self):
        cache = make_cache(capacity=4)
        k, v = fake_kv(cache, tokens=3)
        cache.update(0, k, v)
        cache.commit(3)
        k2, v2 = fake_kv(cache, tokens=2)
        with pytest.raises(CacheOverflowError):
            cache.update(0, k2, v2)


class TestResetAndIsolation:
    def test_reset_reuses_buffers_and_rewinds(self):
        cache = make_cache(capacity=8)
        k, v = fake_kv(cache, tokens=4)
        cache.update(0, k, v)
        cache.commit(4)
        ptr = cache._keys[0].data_ptr()
        cache.reset()
        assert cache.get_seq_length() == 0
        assert cache.is_initialized
        assert cache._keys[0].data_ptr() == ptr

    def test_reset_produces_fresh_sequence_equivalence(self):
        cache = make_cache(capacity=8)
        k1, v1 = fake_kv(cache, tokens=2)
        cache.update(0, k1, v1)
        cache.commit(2)
        cache.reset()
        # write new data; must overwrite the old window completely
        k2, v2 = fake_kv(cache, tokens=2)
        kf, _ = cache.update(0, k2, v2)
        cache.commit(2)
        assert torch.equal(kf[:, :, :2], k2)
        assert not torch.equal(kf[:, :, :2], k1[:, :, :2])

    def test_batch_size_change_rejected(self):
        cache = make_cache(capacity=8)
        k, v = fake_kv(cache, batch=1, tokens=1)
        cache.update(0, k, v)
        cache.commit(1)
        k2, v2 = fake_kv(cache, batch=2, tokens=1)
        with pytest.raises(RuntimeError, match="batch size changed"):
            cache.update(0, k2, v2)

    def test_wrong_head_count_rejected(self):
        cache = make_cache(capacity=8)
        k = torch.randn(1, 3, 1, 64)  # 3 kv heads, configured for 2
        v = torch.randn(1, 3, 1, 64)
        with pytest.raises(ValueError, match="kv_heads"):
            cache.update(0, k, v)

    def test_wrong_head_dim_rejected(self):
        cache = make_cache(capacity=8)
        k = torch.randn(1, 2, 1, 32)
        v = torch.randn(1, 2, 1, 32)
        with pytest.raises(ValueError, match="head_dim"):
            cache.update(0, k, v)

    def test_layer_idx_out_of_range(self):
        cache = make_cache(capacity=8)
        k, v = fake_kv(cache, tokens=1)
        with pytest.raises(ValueError, match="layer_idx"):
            cache.update(7, k, v)

    def test_key_value_mismatch_rejected(self):
        cache = make_cache(capacity=8)
        k, _ = fake_kv(cache, tokens=1)
        v2 = torch.randn(1, 2, 2, 64)
        with pytest.raises(ValueError, match="shape mismatch"):
            cache.update(0, k, v2)


class TestDeviceMismatch:
    def test_device_mismatch_rejected_when_cuda(self, device):
        if device.type != "cuda":
            pytest.skip("needs CUDA for a device-mismatch probe")
        cache = make_cache(capacity=8)
        k, v = fake_kv(cache, tokens=1, device=str(device))
        cache.update(0, k, v)
        cache.commit(1)
        k2 = torch.randn(1, 2, 1, 64, device="cpu")
        v2 = torch.randn(1, 2, 1, 64, device="cpu")
        with pytest.raises(RuntimeError, match="device/dtype mismatch"):
            cache.update(0, k2, v2)
