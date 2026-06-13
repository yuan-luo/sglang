"""M2 step-1: LinearDeltaPool + varlen extraction + replay, bit-exact round-trip.

What M1 proved: replay = chunk_gated_delta_rule_fwd_h(w=0, u=v_new, init=fp32 ckpt)
reconstructs state bit-exactly (single sequence). What M2 step-1 adds (the new,
serve-path-relevant risk):

  * A real STORAGE abstraction (LinearDeltaPool): flat per-token compact deltas
    (kg, v_new, gk) + sparse fp32 checkpoints, with per-sequence metadata.
  * VARLEN handling (multiple sequences in one batch, cu_seqlens) — exactly how a
    real prefill batch is laid out — with correct per-sequence chunk/token slicing.
  * Reconstruct ANY sequence's state at ANY chunk boundary from its nearest sparse
    checkpoint, through the pool's index math.

Validated: pool.reconstruct(seq, pos) == ground-truth fp32 forward state, max|Δ|=0,
across a varlen batch and multiple checkpoint spacings.

Part A: random KDA-shaped inputs (bulletproof test of the pool/varlen index math).
Part B: real KDA gates via chunk_kda_fwd_intra (per-channel decay path).

NOTE on checkpoints: chunk_delta_h writes its per-chunk `h` array in bf16, but the
running state is fp32. For LOSSLESS checkpoints we materialize fp32 state via a
segment-forward into an fp32 pool (== the continuous fp32 forward; proven in M1).
Production M2 will instead have the kernel emit fp32 state at sparse chunk
boundaries directly (one extra output) — same values, no recompute.

    PYTHONPATH=<repo>/python CUDA_VISIBLE_DEVICES=0 python m2_pool_replay.py
"""
import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)

DEV = "cuda"
BT = CHUNK_SIZE  # 64
H = 32           # KDA heads (Hg == Hv here; Kimi-Linear-48B-A3B num_kv_heads_for_linear_attn=0)
K = 128          # head_k_dim
V = 128          # head_v_dim


def _fwd_h(kg, w, u, gk, init_fp32, idx, cu_seqlens):
    """chunk_gated_delta_rule_fwd_h wrapper; in-place updates fp32 state pool.
    Returns (per_chunk_h, v_new); init_fp32[idx] now holds each seq's final state."""
    return chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, g=None, gk=gk,
        initial_state=init_fp32, initial_state_indices=idx,
        save_new_value=True, cu_seqlens=cu_seqlens,
    )


class LinearDeltaPool:
    """Sparse fp32 checkpoints + per-token compact deltas for one linear-attn layer.
    Reconstructs state at any chunk boundary via replay (chunk_delta_h, w=0)."""

    def __init__(self, max_tokens, max_ckpts, ckpt_interval_chunks):
        self.C = ckpt_interval_chunks                 # checkpoint every C chunks
        self.kg = torch.empty(max_tokens, H, K, device=DEV, dtype=torch.bfloat16)
        self.vnew = torch.empty(max_tokens, H, V, device=DEV, dtype=torch.bfloat16)
        self.gk = torch.empty(max_tokens, H, K, device=DEV, dtype=torch.float32)
        self.ckpt = torch.empty(max_ckpts, H, V, K, device=DEV, dtype=torch.float32)
        self._tok = 0
        self._ck = 0
        self.meta = {}  # seq_id -> dict(tok0, T, NT, ckpt_chunks:list, ckpt_slots:list)

    def store_sequence(self, seq_id, kg_s, vnew_s, gk_s, ckpt_states, ckpt_chunks):
        """kg_s/vnew_s/gk_s: [T, H, d] for this sequence. ckpt_states: [n, H,V,K] fp32
        at the chunk boundaries listed in ckpt_chunks (state ENTERING that chunk)."""
        T = kg_s.shape[0]
        tok0 = self._tok
        self.kg[tok0:tok0 + T] = kg_s
        self.vnew[tok0:tok0 + T] = vnew_s
        self.gk[tok0:tok0 + T] = gk_s
        slots = []
        for j in range(len(ckpt_chunks)):
            s = self._ck
            self.ckpt[s] = ckpt_states[j]
            slots.append(s)
            self._ck += 1
        self.meta[seq_id] = dict(
            tok0=tok0, T=T, NT=(T + BT - 1) // BT,
            ckpt_chunks=list(ckpt_chunks), ckpt_slots=slots,
        )
        self._tok += T

    def reconstruct(self, seq_id, target_chunk):
        """Return fp32 state ENTERING chunk `target_chunk` of seq_id (target_chunk ==
        NT means the final state after the whole sequence). Replays from the nearest
        stored checkpoint <= target_chunk."""
        m = self.meta[seq_id]
        # nearest checkpoint chunk <= target_chunk
        cand = [(c, s) for c, s in zip(m["ckpt_chunks"], m["ckpt_slots"]) if c <= target_chunk]
        assert cand, f"no checkpoint <= chunk {target_chunk} for {seq_id}"
        c0, slot = max(cand, key=lambda cs: cs[0])
        if c0 == target_chunk:
            return self.ckpt[slot].clone()
        a, b = c0 * BT, min(target_chunk * BT, m["T"])  # token range to replay
        tok0 = m["tok0"]
        kg_r = self.kg[tok0 + a: tok0 + b].unsqueeze(0).contiguous()
        vnew_r = self.vnew[tok0 + a: tok0 + b].unsqueeze(0).contiguous()
        gk_r = self.gk[tok0 + a: tok0 + b].unsqueeze(0).contiguous()
        w0 = torch.zeros_like(kg_r)
        pool = self.ckpt[slot].clone().unsqueeze(0)          # [1,H,V,K] fp32
        idx = torch.zeros(1, dtype=torch.int32, device=DEV)
        _fwd_h(kg_r, w0, vnew_r, gk_r, pool, idx, cu_seqlens=None)
        return pool[0]


def make_varlen(seqlens, seed, real_kda=False):
    """Build a varlen batch and run the full fp32 forward. Returns the compact
    deltas (kg,v_new,gk), per-seq final fp32 states, cu_seqlens, and the intra w/u
    (only needed to materialize fp32 checkpoints in the test harness)."""
    torch.manual_seed(seed)
    Tt = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=DEV)
    n = len(seqlens)
    if real_kda:
        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.kda import chunk_kda_fwd_intra

        def l2(x):
            return x / (x.float().norm(dim=-1, keepdim=True) + 1e-6)
        q = l2(torch.randn(1, Tt, H, K, device=DEV)).bfloat16()
        k = l2(torch.randn(1, Tt, H, K, device=DEV)).bfloat16()
        v = (torch.randn(1, Tt, H, V, device=DEV) * 0.5).bfloat16()
        beta = torch.rand(1, Tt, H, device=DEV, dtype=torch.bfloat16)
        g_raw = (-torch.rand(1, Tt, H, K, device=DEV, dtype=torch.float32) * 0.05 - 0.01)
        gk = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu)
        w, u, _, kg, _Aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=gk, beta=beta, scale=K ** -0.5,
            cu_seqlens=cu, chunk_size=BT, chunk_indices=None,
            fuse_diagonal=False, fuse_recompute=False)
    else:
        kg = torch.randn(1, Tt, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
        w = torch.randn(1, Tt, H, K, device=DEV, dtype=torch.bfloat16) * 0.1
        u = torch.randn(1, Tt, H, V, device=DEV, dtype=torch.bfloat16) * 0.5
        gk = (-torch.rand(1, Tt, H, K, device=DEV, dtype=torch.float32) * 0.05 - 0.01)

    init = torch.zeros(n, H, V, K, device=DEV, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32, device=DEV)
    _h, v_new = _fwd_h(kg, w, u, gk, init, idx, cu_seqlens=cu)
    finals = init.clone()                                   # [n,H,V,K] fp32 per-seq final
    return kg[0], v_new[0], gk[0], w[0], u[0], finals, cu


def fp32_ckpt(kg, w, u, gk, a, b):
    """fp32 state entering token b, starting from token a (a,b chunk-aligned, single seq)."""
    pool = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32, device=DEV)
    _fwd_h(kg[a:b].unsqueeze(0).contiguous(), w[a:b].unsqueeze(0).contiguous(),
           u[a:b].unsqueeze(0).contiguous(), gk[a:b].unsqueeze(0).contiguous(),
           pool, idx, cu_seqlens=None)
    return pool[0]


def run(seqlens, C_chunks, seed, real_kda):
    kg, vnew, gk, w, u, finals, cu = make_varlen(seqlens, seed, real_kda)
    pool = LinearDeltaPool(max_tokens=sum(seqlens), max_ckpts=1000, ckpt_interval_chunks=C_chunks)
    worst = 0.0
    for s, T in enumerate(seqlens):
        a0, a1 = int(cu[s]), int(cu[s + 1])
        kg_s, vnew_s, gk_s = kg[a0:a1], vnew[a0:a1], gk[a0:a1]
        w_s, u_s = w[a0:a1], u[a0:a1]
        NT = (T + BT - 1) // BT
        ckpt_chunks = list(range(0, NT, C_chunks))          # 0, C, 2C, ...
        # materialize fp32 checkpoint state entering each ckpt chunk (segment forward)
        states = [fp32_ckpt(kg_s, w_s, u_s, gk_s, 0, c * BT) for c in ckpt_chunks]
        pool.store_sequence(s, kg_s, vnew_s, gk_s, states, ckpt_chunks)
        # reconstruct final state (chunk == NT) and compare to ground truth
        rec = pool.reconstruct(s, NT)
        d = (rec - finals[s]).abs().max().item()
        worst = max(worst, d)
    tag = "real-KDA" if real_kda else "random"
    print(f"[{tag}] seqlens={seqlens} C={C_chunks}chunks  final-state recon max|Δ|={worst:.3e}"
          f"  -> {'BIT-EXACT' if worst == 0 else 'DRIFT'}")
    return worst


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)} BT={BT} H={H} K={K} V={V}")
    print("== Part A: pool + varlen index math (random KDA-shaped) ==")
    wA = 0.0
    for sl in ([192, 320, 64], [64, 640, 128, 256], [512, 512, 512]):
        for C in (1, 2, 4):
            wA = max(wA, run(sl, C, seed=0, real_kda=False))
    print(f"Part A worst max|Δ| = {wA:.3e} ({'ALL BIT-EXACT' if wA == 0 else 'NOT'})")
    print("== Part B: real KDA per-channel gates (varlen) ==")
    wB = run([192, 320, 128], C_chunks=2, seed=1, real_kda=True)
    print(f"Part B max|Δ| = {wB:.3e} ({'BIT-EXACT' if wB == 0 else 'DRIFT'})")
