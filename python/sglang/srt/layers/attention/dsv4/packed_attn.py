# SPDX-License-Identifier: Apache-2.0
"""Head-packed sparse attention for DeepSeek-V4 C128A prefill layers.

At TP8 DeepSeek-V4 has 8 real heads per rank, but FlashMLA's sparse prefill
kernel only accepts h_q=64, so the stock path pads q with 56 zero heads and
throws away 7/8 of every tensor-core tile. For C128A layers the per-token
index row is structurally

    [0, n_c)                    compressed prefix, shared by nearby tokens
    [w, w + swa)                a contiguous sliding window

and n_c is constant over runs of exactly 128 query tokens, so 16 adjacent
tokens x 8 real heads pack into ONE 128-row tile whose union index list is
barely longer than a single token's (measured blow-up 1.002x). The packed
kernel takes two per-query ranges instead of a bitmask, which is what makes
the union exact: query g attends union ranks [0, pref_g) U [lo_g, hi_g).

This module derives those ranges directly from positions -- the same inputs
combine_topk_swa_indices consumes -- so the [num_tokens, topk+window] index
matrix is never materialized, and the result is cached per (metadata, chunk)
because it depends only on positions, not on the layer.

Enable with SGLANG_DSV4_PACKED_ATTN=1; SGLANG_DSV4_PACKED_SO overrides the
kernel .so path. Any shape the packed path cannot serve exactly falls back
to the stock kernel.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

B_H = 128  # packed rows per tile -- fixed by the kernel
B_TOPK = 64  # kernel index-width granularity
MAX_G = 16  # kernel limit: at most 16 queries share a tile

_ENABLED = os.environ.get("SGLANG_DSV4_PACKED_ATTN", "0") == "1"
# The kernel ships in-tree next to the other DSA sources. Iteration builds
# under /tmp must be selected explicitly so a stale process-global file can
# never silently replace the vendored artifact.
_SO = os.environ.get("SGLANG_DSV4_PACKED_SO", "")
_MOD = None
_FN = None
_FAILED = False
_SCRATCH: dict = {}
_EXECUTION_LOGGED = False


_CHECK = os.environ.get("SGLANG_DSV4_PACKED_CHECK", "0") == "1"
_CHECK_STATS = {"n": 0, "worst": 0.0}


def enabled() -> bool:
    return _ENABLED


def checking() -> bool:
    """Run the stock path too and compare (validation only; halves speed)."""
    return _CHECK


def report_check(packed, stock, n_local_heads):
    """Compare a packed result against the stock kernel's, in place."""
    if packed is None:
        return
    h = int(n_local_heads)
    packed_real = packed[:, :h].float()
    stock_real = stock[:, :h].float()
    if not (torch.isfinite(packed_real).all() and torch.isfinite(stock_real).all()):
        raise RuntimeError("DSv4 packed-attention check found non-finite output")
    d = (packed_real - stock_real).abs()
    scale = stock_real.abs().max().clamp(min=1e-6)
    max_abs = float(d.max().item())
    rel = float(max_abs / scale.item())
    _CHECK_STATS["n"] += 1
    if rel > _CHECK_STATS["worst"]:
        _CHECK_STATS["worst"] = rel
    if rel > 0.05:
        raise RuntimeError(
            "DSv4 packed-attention check exceeded relative-error tolerance: "
            f"rel={rel:.5f}, limit=0.05000, maxabs={max_abs:.5f}"
        )
    if _CHECK_STATS["n"] % 20 == 0:
        logger.info(
            "[dsv4-packed] check n=%d rel=%.5f worst=%.5f " "maxabs=%.5f",
            _CHECK_STATS["n"],
            rel,
            _CHECK_STATS["worst"],
            max_abs,
        )


def _fn():
    """Load the packed-attention .so once per process."""
    global _MOD, _FN, _FAILED
    if _FN is None and not _FAILED:
        try:
            if not _SO:
                raise RuntimeError("SGLANG_DSV4_PACKED_SO is required")
            import tvm_ffi

            _MOD = tvm_ffi.load_module(_SO)
            _FN = _MOD["dsv4_masked_mla_bf16"]
            logger.info("DSv4 packed attention loaded from %s", _SO)
        except Exception as e:  # noqa: BLE001
            _FAILED = True
            logger.warning(
                "DSv4 packed attention unavailable (%s); "
                "falling back to the stock kernel",
                e,
            )
    return _FN


def _build_ranges(
    query_start_loc,
    seq_lens,
    gather_lens,
    chunk_M,
    chunk_N,
    window_size,
    compress_ratio,
    top_k,
    num_tokens,
    G,
):
    """Union list + per-query ranges for one chunk, from positions alone.

    Mirrors _combine_topk_swa_indices_kernel exactly: token at absolute
    position ``pos`` of request ``b`` attends compressed entries
    ``M*b + [0, min((pos+1)//ratio, top_k))`` and window entries starting at
    ``M*b + N + pos - swa + 1 - (seq_len - gather_len)``.
    """
    dev = seq_lens.device
    qsl = query_start_loc - query_start_loc[0]
    query_len = qsl[1:] - qsl[:-1]  # [num_reqs]
    start_pos = seq_lens - query_len

    tok = torch.arange(num_tokens, dtype=torch.int64, device=dev)
    # request id per token (chunk-local)
    b = torch.searchsorted(qsl[1:].to(torch.int64), tok, right=True)
    pos = start_pos.to(torch.int64)[b] + (tok - qsl.to(torch.int64)[b])

    topk_len = torch.clamp(
        torch.div(pos + 1, compress_ratio, rounding_mode="floor"), max=top_k
    )
    swa_len = torch.clamp(pos + 1, max=window_size)
    gather_start = (seq_lens - gather_lens).to(torch.int64)[b]
    base = chunk_M * b
    win_start = base + chunk_N + pos - swa_len + 1 - gather_start

    ng = num_tokens // G
    pl = topk_len.view(ng, G)
    ws = win_start.view(ng, G)
    we = ws + swa_len.view(ng, G)
    P = pl.max(1).values  # [ng]
    wlo = ws.min(1).values
    whi = we.max(1).values
    ulen = P + (whi - wlo)
    cap = (int(ulen.max().item()) + B_TOPK - 1) // B_TOPK * B_TOPK

    # union rank -> workspace slot: prefix first (offset by this request's
    # base), then the shared window span.
    gbase = base.view(ng, G)[:, 0].view(-1, 1)
    rank = torch.arange(cap, dtype=torch.int64, device=dev).view(1, cap)
    u = torch.where(
        rank < P.view(-1, 1), gbase + rank, wlo.view(-1, 1) + (rank - P.view(-1, 1))
    )
    u = (
        torch.where(rank < ulen.view(-1, 1), u, torch.full_like(u, -1))
        .to(torch.int32)
        .contiguous()
    )

    pref = pl.to(torch.int32).contiguous()
    w_lo = (P.view(-1, 1) + (ws - wlo.view(-1, 1))).to(torch.int32).contiguous()
    w_hi = (w_lo + swa_len.view(ng, G).to(torch.int32)).contiguous()
    counts = ulen.to(torch.int32).contiguous()
    return u.view(ng, 1, cap), pref, w_lo, w_hi, counts


def _scratch_key(q, kv, attn_sink, h):
    # A worker normally executes attention on one stream, but keeping the
    # stream in the key makes direct eager overlap safe: a later dispatch on a
    # different stream cannot overwrite a still-live output view.
    stream = int(torch.cuda.current_stream(q.device).cuda_stream) if q.is_cuda else 0
    return (
        str(q.device),
        stream,
        int(h),
        q.dtype,
        kv.dtype,
        attn_sink.dtype,
    )


def _get_scratch(q, kv, attn_sink, ng, h):
    """Return shape views into one power-of-two growable scratch allocation."""
    key = _scratch_key(q, kv, attn_sink, h)
    buf = _SCRATCH.get(key)
    if buf is None or buf["capacity"] < ng:
        capacity = 1 << (int(ng) - 1).bit_length()
        buf = {
            "capacity": capacity,
            "o": torch.empty((capacity, B_H, 512), dtype=q.dtype, device=q.device),
            "lse": torch.empty((capacity, B_H), dtype=torch.float32, device=q.device),
            "q": torch.empty((capacity, B_H, 512), dtype=q.dtype, device=q.device),
            "sink": torch.empty(B_H, dtype=attn_sink.dtype, device=q.device),
        }
        _SCRATCH[key] = buf
    return (
        buf["o"][:ng],
        buf["lse"][:ng],
        buf["q"][:ng],
        buf["sink"],
    )


def try_packed_prefill(
    *,
    q,
    kv,
    attn_sink,
    sm_scale,
    query_start_loc,
    seq_lens,
    gather_lens,
    chunk_M,
    chunk_N,
    window_size,
    compress_ratio,
    top_k,
    n_local_heads,
    cache_owner,
    cache_key,
):
    """Run one C128A prefill chunk on the packed kernel.

    Returns the packed output view when it ran; ``None`` means the caller must
    use the stock path. Returning persistent output scratch directly avoids
    allocating and copying another 64 MiB tensor for every C128 layer at TP8.
    Same-stream ordering keeps the view alive until its consumer completes.
    """
    # The TVM-FFI entry casts raw pointers to these exact types.  Decline
    # unsupported dtypes before loading or dispatching the module rather than
    # silently reinterpreting memory.
    if (
        q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or attn_sink.dtype != torch.float32
    ):
        return None
    fn = _fn()
    if fn is None:
        return None
    num_tokens = q.shape[0]
    num_reqs = seq_lens.shape[0]
    # One tile is B_H rows = G query tokens x n_local_heads real heads, so the
    # packing factor follows the TP degree: G = 128 / heads-per-rank. Anything
    # that does not tile the 128 rows exactly (or would need more than the
    # kernel's 16 queries per tile) falls back.
    h = int(n_local_heads)
    if h <= 0 or h > q.shape[1] or h > attn_sink.numel() or B_H % h != 0:
        return None
    G = B_H // h
    if G > MAX_G:
        return None
    # Groups must not straddle a request (each request has its own workspace
    # base offset).  SGLang's tiny model warmup may pad q beyond the real
    # query_start_loc extent; the stock path understands those padding rows,
    # while the packed range builder intentionally does not.
    real_query_tokens = int((query_start_loc[-1] - query_start_loc[0]).item())
    if (
        num_reqs != 1
        or real_query_tokens != num_tokens
        or num_tokens % G != 0
        or num_tokens == 0
    ):
        return None
    if q.shape[-1] != 512 or kv.shape[-1] != 512:
        return None

    ng = num_tokens // G
    cached = getattr(cache_owner, "_dsv4_packed_cache", None)
    if cached is None:
        cached = {}
        cache_owner._dsv4_packed_cache = cached
    plan = cached.get(cache_key)
    if plan is None:
        plan = _build_ranges(
            query_start_loc,
            seq_lens,
            gather_lens,
            chunk_M,
            chunk_N,
            window_size,
            compress_ratio,
            top_k,
            num_tokens,
            G,
        )
        cached[cache_key] = plan
    idxp, pref, w_lo, w_hi, counts = plan
    if any(tensor.dtype != torch.int32 for tensor in (idxp, pref, w_lo, w_hi, counts)):
        return None

    # Persistent per-(device, stream, heads, dtypes) scratch grows to a
    # power-of-two capacity.  Different tail-query lengths therefore reuse one
    # allocation instead of pinning one large Q/O pair per exact shape.
    o_packed, lse, qp, sink = _get_scratch(q, kv, attn_sink, ng, h)
    # attn_sink is a distinct learned parameter for every C128A layer. The
    # large Q/O scratch is shape-shared, but its tiny row-expanded sink must
    # be refreshed before each layer rather than retaining the first layer's.
    sink.view(G, h).copy_(attn_sink[:h])
    # The packed row layout (token-major within a group, then head) is a
    # two-level stride over the 64-head tensor, so it cannot be a view; this
    # copy is what a 3D Q tensor map would remove.
    qp.view(ng, G, h, 512).copy_(q[:, :h].view(ng, G, h, 512))
    kvp = kv.view(-1, 1, kv.shape[-1])

    fn(qp, kvp, idxp, pref, w_lo, w_hi, counts, float(sm_scale), h, o_packed, lse, sink)

    out = o_packed.view(num_tokens, h, 512)
    global _EXECUTION_LOGGED
    if not _EXECUTION_LOGGED:
        logger.info("DSV4_PACKED_KERNEL_EXECUTED packed BF16 MLA kernel dispatched")
        _EXECUTION_LOGGED = True
    return out
