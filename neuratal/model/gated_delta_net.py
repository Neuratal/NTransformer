"""Gated DeltaNet attention (Kimi-K3 / Qwen3.8-Flash-Next "KDA" lineage) — experimental, opt-in.

Implements the gated delta rule, Eq. 10 of Yang, Kautz & Hatamizadeh,
"Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv:2412.06464,
ICLR 2025, camera-ready v3):

    S_t = S_{t-1} · ( α_t ( I − β_t k_t k_tᵀ ) ) + β_t v_t k_tᵀ
    o_t = S_t q_t

with data-dependent gates α, β ∈ (0,1) (sigmoid), L2-normalized queries/keys,
an SiLU output gate, and per-head state S ∈ R^{d_v × d_k} (head_dim x head_dim).

Why this exists here: KDA is the dominant mixer in Kimi K3 (69 of 93 layers)
and Qwen3.8-Flash-Next (3:1 GDN:QSA). It is the only frontier-2026 mixer that
is fully testable without training, and it targets long-context decode cost:
per-token work is O(d_k·d_v) with NO context-length term, and per-layer state
is O(d_k·d_v) regardless of sequence length (vs O(L · d_h) KV growth).

Prototype scope (deliberate, documented):
* Sequential recurrence (Eq. 10 verbatim). The WY-representation chunkwise
  parallel form (the paper's training algorithm) is NOT implemented; this is an
  inference prototype. Prefill is therefore O(L) small matmuls and will measure
  slower than dense SDPA — by design, not as a defect.
* No short causal convolution on q/k/v, no asymmetric head counts, no MoE —
  those are orthogonal ingredients; noted as future work.
* State is kept in fp32 even under bf16 models; gates are sigmoid (bounded);
  (I − β·k·kᵀ) is a contraction ⇒ random-init states are stable by construction.
* Positional information is implicit through the decay gate; RoPE is not applied.

Correctness contract (tested): shapes/finite, strict causality, incremental
decode == full-sequence forward (fp32, ≤ 1e-5), gradients flow, alpha/beta
bounded, checkpoint roundtrip, batch isolation between cache instances.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig


class DeltaNetCache:
    """Recurrent-state cache; interface-compatible with KVCache for NeuratalModel.

    Holds one fp32 state tensor per layer: [batch, heads, head_dim, head_dim].
    Bounded memory, no growth with sequence length. ``commit`` enforces the same
    contract as KVCache.commit (set by the attention module at end of forward).
    """

    def __init__(self, config: NeuratalConfig):
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self._states: list[torch.Tensor] = []
        self._length = 0
        self._pending = 0
        self.batch_size = None
        self.device = None
        self.dtype = None

    @property
    def is_initialized(self) -> bool:
        return bool(self._states)

    def get_seq_length(self) -> int:
        return self._length

    def initialize(self, batch_size: int, device, dtype) -> None:
        if self.is_initialized:
            raise RuntimeError("DeltaNetCache already initialized; call reset() instead")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        shape = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        self._states = [torch.zeros(shape, device=device, dtype=torch.float32) for _ in range(self.num_layers)]
        self.batch_size, self.device, self.dtype = batch_size, torch.device(device), dtype

    def layer_state(self, layer_idx: int) -> torch.Tensor:
        return self._states[layer_idx]

    def write_layer(self, layer_idx: int, state: torch.Tensor) -> None:
        self._states[layer_idx] = state

    def mark_pending(self, num_tokens: int) -> None:
        self._pending = num_tokens

    def commit(self, num_tokens: int) -> None:
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if self._pending == 0:
            raise RuntimeError("commit() without a preceding forward in this pass")
        if num_tokens != self._pending:
            raise RuntimeError(f"commit({num_tokens}) != tokens written ({self._pending})")
        self._length += num_tokens
        self._pending = 0
        if self._length > self.max_position_embeddings:
            raise ValueError(
                f"DeltaNetCache stream length {self._length} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}"
            )

    def reset(self) -> None:
        self._length = 0
        self._pending = 0
        if self._states:
            for layer in self._states:
                layer.zero_()

    def __repr__(self) -> str:
        n = self.num_heads * self.head_dim * self.head_dim * 4
        return (f"DeltaNetCache(len={self._length}, layers={self.num_layers}, "
                f"state_per_layer={n} bytes, initialized={self.is_initialized})")


class GatedDeltaNetAttention(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self._gdn_chunk_size = int(getattr(config, "gdn_chunk_size", 0) or 0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.out_norm = RMSNorm2d(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, positions=None, *, past_key_values=None, layer_idx=None):
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be 3-D [batch, seq, hidden], got {hidden_states.dim()}-D")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_states.size(-1)}")
        batch, seq_len, _ = hidden_states.shape
        x = hidden_states
        # fp32 internal math for the recurrence regardless of model dtype,
        # including fp32 copies of the weights (never mutate the module params)
        xf = x.float()

        q = F.normalize(F.linear(xf, self.q_proj.weight.float()).view(batch, seq_len, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(F.linear(xf, self.k_proj.weight.float()).view(batch, seq_len, self.num_heads, self.head_dim), dim=-1)
        v = F.linear(xf, self.v_proj.weight.float()).view(batch, seq_len, self.num_heads, self.head_dim)
        alpha = torch.sigmoid(F.linear(xf, self.alpha_proj.weight.float(), self.alpha_proj.bias.float()))
        beta = torch.sigmoid(F.linear(xf, self.beta_proj.weight.float(), self.beta_proj.bias.float()))
        z = F.silu(F.linear(xf, self.gate_proj.weight.float())).view(batch, seq_len, self.num_heads, self.head_dim)

        d = self.head_dim
        chunk_size = getattr(self, "_gdn_chunk_size", 0) or 0
        use_chunkwise = chunk_size >= 2 and seq_len >= 2
        if past_key_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when past_key_values is provided")
            if not isinstance(past_key_values, DeltaNetCache):
                raise TypeError(
                    f"GatedDeltaNetAttention requires DeltaNetCache, got {type(past_key_values).__name__}"
                )
            if not past_key_values.is_initialized:
                past_key_values.initialize(batch, x.device, x.dtype)
            elif past_key_values.batch_size != batch:
                raise RuntimeError(
                    f"batch size changed from {past_key_values.batch_size} to {batch}; use a fresh cache"
                )
            if x.device != past_key_values.device:
                raise RuntimeError(f"cache device mismatch: {past_key_values.device} vs {x.device}")
            state = past_key_values.layer_state(layer_idx)
        else:
            state = torch.zeros(batch, self.num_heads, d, d, device=x.device, dtype=torch.float32)

        if use_chunkwise:
            o_all, state = self._forward_chunkwise(q, k, v, alpha, beta, state, chunk_size)
        else:
            outs = []
            for t in range(seq_len):
                kt, vt, qt = k[:, t], v[:, t], q[:, t]              # [B,H,d]
                a = alpha[:, t].unsqueeze(-1).unsqueeze(-1)        # [B,H,1,1]
                b = beta[:, t].unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
                Sk = torch.einsum("bhvd,bhd->bhv", state, kt)        # [B,H,d]
                state = a * state - a * b * torch.einsum("bhv,bhd->bhvd", Sk, kt) \
                    + b * torch.einsum("bhv,bhd->bhvd", vt, kt)
                o = torch.einsum("bhvd,bhd->bhv", state, qt)         # [B,H,d]
                outs.append(o)
            o_all = torch.stack(outs, dim=1)

        if past_key_values is not None:
            past_key_values.write_layer(layer_idx, state)
            past_key_values.mark_pending(seq_len)

        o = o_all.to(x.dtype)                                    # [B,T,H,d]
        o = self.out_norm(o)
        o = (o * z.to(x.dtype)).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return F.linear(o.float(), self.o_proj.weight.float()).to(x.dtype)

    def _forward_chunkwise(self, q, k, v, alpha, beta, state, C):
        """WY chunkwise parallel form of the gated delta rule (arXiv:2412.06464, §3.3, Eqs. 5–9).

        Per chunk (tokens t0..t0+C-1, all batch×head lanes vectorized):
            γ_r  = ∏_{i≤r} α_i   (within-chunk cumulative decay)
            Γ_ij = γ_i/γ_j  (i ≥ j, else 0)
            T    = (I + strictlyLower(diag(β)·(Γ ⊙ K Kᵀ)))⁻¹ diag(β)
            W    = T·K ;  U = T·V
            W←,K→,Q← = γ-scaled/rev-scaled copies
            O    = Q← S_0ᵀ + (Q Kᵀ ⊙ M_causal)(U − W← S_0ᵀ)
            S_1  = γ_C S_0 + (U − W← S_0ᵀ)ᵀ K→

        Sequential and chunkwise are the same math with different summation
        order; fp32 is used throughout (fp32 for bf16 models) and outputs are
        compared against the sequential oracle in tests with justified fp32
        tolerances (~1e-5 rel) — never bitwise-forced.
        """
        B, T, H, D = q.shape
        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))          # [B,H,T,D]
        ah, bh = (t.transpose(1, 2) for t in (alpha, beta))          # [B,H,T]
        state = state.float().clone()
        outs: list[torch.Tensor] = []

        for t0 in range(0, T, C):
            Ci = min(C, T - t0)
            qc, kc, vc = qh[:, :, t0:t0 + Ci], kh[:, :, t0:t0 + Ci], vh[:, :, t0:t0 + Ci]
            ac, bc = ah[:, :, t0:t0 + Ci].float(), bh[:, :, t0:t0 + Ci].float()

            gamma = torch.cumprod(ac, dim=-1)                          # [B,H,Ci]
            gamma_C = gamma[..., -1:].unsqueeze(-1)                     # [B,H,1,1]
            gamma_safe = gamma.clamp_min(1e-30)
            Gamma = torch.tril(gamma_safe.unsqueeze(-1) * (1.0 / gamma_safe).unsqueeze(-2))       # [B,H,Ci,Ci]
            G = torch.matmul(kc.float(), kc.float().transpose(-2, -1))   # [B,H,Ci,Ci]
            LG = bc.unsqueeze(-1) * (Gamma * G)                          # diag(β) scales rows
            Lmat = torch.tril(LG, diagonal=-1)                           # strictly below diagonal
            I = torch.eye(Ci, device=Lmat.device, dtype=torch.float32).expand(B, H, Ci, Ci)
            Tmat = torch.linalg.solve_triangular(
                I + Lmat, torch.diag_embed(bc), upper=False, unitriangular=True
            )                                                              # [B,H,Ci,Ci]
            W = torch.matmul(Tmat, kc.float())                           # [B,H,Ci,D]
            U = torch.matmul(Tmat, vc.float())                           # [B,H,Ci,D]

            W_dec = gamma.unsqueeze(-1) * W                              # W← [B,H,Ci,D]
            gammaC_over_gamma = gamma[..., -1:] / gamma_safe                # [B,H,Ci]
            K_dec = gammaC_over_gamma.unsqueeze(-1) * kc.float()            # K→ [B,H,Ci,D]
            Q_dec = gamma.unsqueeze(-1) * qc.float()                      # Q← [B,H,Ci,D]

            WS = torch.matmul(W_dec, state.transpose(-2, -1))            # [B,H,Ci,D]
            R = U - WS                                                   # Ũ − W←Sᵀ [B,H,Ci,D]

            P = torch.matmul(qc.float(), kc.float().transpose(-2, -1))   # [B,H,Ci,Ci]
            P = torch.tril(Gamma * P)                                    # γ-weighted causal QKᵀ (j ≤ i)
            O_causal = torch.matmul(P, R)                                # [B,H,Ci,D]
            O_state = torch.matmul(Q_dec, state.transpose(-2, -1))       # [B,H,Ci,D]
            outs.append(O_state + O_causal)

            state = gamma_C * state + torch.matmul(R.transpose(-2, -1), K_dec)

        out = torch.cat(outs, dim=2).transpose(1, 2)                     # [B,T,H,D]
        return out, state

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim}, gated_delta=Eq.10-arXiv:2412.06464")


class RMSNorm2d(nn.Module):
    """Per-head RMSNorm over the head_dim axis (no bias), fp32 statistics."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True)
        return (x32 * torch.rsqrt(rms + self.eps) * self.weight.float()).to(dtype)
