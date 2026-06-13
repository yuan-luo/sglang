from __future__ import annotations

"""
Copyright 2023-2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
LinearDeltaPool — compact, token/chunk-granular state storage for linear
attention (KDA / GDN / gated-delta-rule) layers.

Motivation
----------
A linear-attention layer summarizes all seen tokens in a single recurrent state
matrix ``S in R^{d_k x d_v}`` (per head, per layer). That makes prefix caching
coarse: the existing ``MambaRadixCache`` stores one FULL state per reuse point
(tens of MB each), so the mamba state pool holds only ~O(1000) checkpoints and
saturates at a small fraction of the token-KV capacity — re-accessed prefixes
whose token KV is still cached must nonetheless recompute the SSM recurrence
(measured: reuse collapses 0.94 -> 0.05 once distinct prefixes exceed the pool).

Key identity (gated delta rule, shared by GDN scalar-decay and KDA per-channel
decay): the per-step state increment is rank-1,

    S_t = diag(alpha_t) S_{t-1} + k_t u_t^T,

so a token range can be reconstructed from a SPARSE full checkpoint plus the
compact per-token triple ``(k_g, v_new, decay)`` by REPLAY — which is exactly
``chunk_gated_delta_rule_fwd_h`` run with ``w = 0`` and ``u = v_new`` starting
from the checkpoint (verified bit-exact when checkpoints are stored fp32).

This pool stores, per linear-attention layer:
  * per-token compact deltas ``k_g`` (=gated key), ``v_new`` (=WY-corrected delta
    value), ``decay`` (=cumulative gate; ``[d_k]`` per-channel for KDA, scalar for
    GDN), and
  * a sparse set of FULL fp32 checkpoints (state at every ``ckpt_interval`` tokens).

It owns ONLY the storage tensors and the store/reconstruct math. Slot lifecycle
(allocation, eviction, copy-on-write into a request's state slot) is owned by the
caller (``MambaRadixCache`` / ``HybridReqToTokenPool``), mirroring how
``MambaPool`` owns storage while ``MambaSlotAllocator`` owns the free-list.

Replay reuses the production recurrence kernel verbatim, so GDN and KDA share one
path; the only difference is ``per_channel_decay`` (KDA ``USE_GK`` vs GDN
``USE_G``).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)


@dataclass(frozen=True, kw_only=True)
class LinearDeltaShape:
    """Per-head dimensions of one linear-attention layer."""

    num_k_heads: int  # Hg (key/gate heads; may be < Hv under GQA)
    num_v_heads: int  # Hv (value heads = state head count)
    head_k_dim: int  # d_k
    head_v_dim: int  # d_v


class LinearDeltaPool:
    """Storage + replay for compact linear-attention state deltas.

    Tensors (all indexed by an absolute *delta-token slot* on the token axis, and a
    *checkpoint slot* on the checkpoint axis; both are handed out by the caller):

        kg     : [L, max_delta_tokens, Hg, d_k]   (delta dtype, e.g. bf16)
        v_new  : [L, max_delta_tokens, Hv, d_v]   (delta dtype)
        decay  : [L, max_delta_tokens, Hv, d_k]   if per_channel_decay (KDA)
                 [L, max_delta_tokens, Hv, 1]     otherwise (GDN scalar)   (fp32)
        ckpt   : [L, max_ckpt_slots, Hv, d_v, d_k] (fp32, lossless replay base)

    The decay buffer holds the SAME cumulative-within-chunk gate that
    ``chunk_gated_delta_rule_fwd_h`` consumes as ``gk`` (per-channel) / ``g``
    (scalar); only its value at the last token of each replayed chunk matters for
    the inter-chunk state decay, but we store it per-token so any chunk-aligned
    sub-range can be replayed.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_delta_tokens: int,
        max_ckpt_slots: int,
        shape: LinearDeltaShape,
        ckpt_interval: int,
        device: str,
        per_channel_decay: bool = True,
        delta_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = CHUNK_SIZE,
    ):
        assert ckpt_interval % chunk_size == 0, (
            f"ckpt_interval ({ckpt_interval}) must be a multiple of chunk_size "
            f"({chunk_size}); checkpoints land on chunk boundaries."
        )
        self.L = num_layers
        self.shape = shape
        self.ckpt_interval = ckpt_interval
        self.chunk_size = chunk_size
        self.per_channel_decay = per_channel_decay
        self.device = device
        Hg, Hv = shape.num_k_heads, shape.num_v_heads
        dk, dv = shape.head_k_dim, shape.head_v_dim

        self.kg = torch.empty(
            num_layers, max_delta_tokens, Hg, dk, dtype=delta_dtype, device=device
        )
        self.v_new = torch.empty(
            num_layers, max_delta_tokens, Hv, dv, dtype=delta_dtype, device=device
        )
        self.decay = torch.empty(
            num_layers,
            max_delta_tokens,
            Hv,
            dk if per_channel_decay else 1,
            dtype=torch.float32,
            device=device,
        )
        # fp32 checkpoints (lossless replay base), per layer. Per-slot layout
        # [Hv, d_v, d_k] matches the `initial_state` chunk_gated_delta_rule_fwd_h wants.
        self.ckpt = torch.empty(
            num_layers, max_ckpt_slots, Hv, dv, dk, dtype=torch.float32, device=device
        )
        self.max_delta_tokens = max_delta_tokens
        self.max_ckpt_slots = max_ckpt_slots

    # ----- write paths (caller supplies destination slot indices) -----

    def store_deltas(
        self,
        layer_idx: int,
        token_slots: torch.Tensor,  # [T] int, destination delta-token slots
        kg: torch.Tensor,  # [T, Hg, d_k]
        v_new: torch.Tensor,  # [T, Hv, d_v]
        decay: torch.Tensor,  # [T, Hv, d_k] (KDA) or [T, Hv, 1]/[T, Hv] (GDN)
    ) -> None:
        self.kg[layer_idx, token_slots] = kg.to(self.kg.dtype)
        self.v_new[layer_idx, token_slots] = v_new.to(self.v_new.dtype)
        if decay.dim() == 2:  # [T, Hv] scalar -> [T, Hv, 1]
            decay = decay.unsqueeze(-1)
        self.decay[layer_idx, token_slots] = decay.to(torch.float32)

    def store_checkpoint(
        self, layer_idx: int, ckpt_slot: int, state: torch.Tensor  # [Hv, d_v, d_k]
    ) -> None:
        self.ckpt[layer_idx, ckpt_slot] = state.to(torch.float32)

    # ----- read path: reconstruct state by replay -----

    def reconstruct(
        self,
        layer_idx: int,
        ckpt_slot: int,
        replay_token_slots: torch.Tensor,  # ordered delta-token slots from the ckpt
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct the fp32 state after applying the deltas at
        ``replay_token_slots`` (in order, a whole number of chunks) on top of the
        checkpoint in ``ckpt_slot``. Returns ``[Hv, d_v, d_k]`` fp32.

        ``replay_token_slots`` MUST start at the checkpoint's token position and be
        chunk-aligned in length, so the chunk decay applies at the right boundaries.
        Empty -> returns a copy of the checkpoint.
        """
        base = self.ckpt[layer_idx, ckpt_slot]
        if replay_token_slots.numel() == 0:
            return base.clone() if out is None else out.copy_(base)

        kg = self.kg[layer_idx, replay_token_slots].unsqueeze(0).contiguous()
        v_new = self.v_new[layer_idx, replay_token_slots].unsqueeze(0).contiguous()
        decay = self.decay[layer_idx, replay_token_slots]  # [T, Hv, dk|1]
        w0 = torch.zeros_like(kg)
        state = base.clone().unsqueeze(0)  # [1, Hv, d_v, d_k] fp32 (in-place updated)
        idx = torch.zeros(1, dtype=torch.int32, device=self.device)
        if self.per_channel_decay:
            gk = decay.unsqueeze(0).contiguous()  # [1, T, Hv, dk]
            g = None
        else:
            gk = None
            g = decay.squeeze(-1).unsqueeze(0).contiguous()  # [1, T, Hv]
        chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w0,
            u=v_new,
            g=g,
            gk=gk,
            initial_state=state,
            initial_state_indices=idx,
            save_new_value=False,
            cu_seqlens=None,
        )
        result = state[0]
        return result if out is None else out.copy_(result)

    # ----- introspection -----

    def mem_usage_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.kg, self.v_new, self.decay, self.ckpt)
        )
