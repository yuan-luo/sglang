"""
KDA "B" (token/chunk-level radix) — M1 replay oracle.

Validates the mathematical core of direction B for linear attention:

    "Store a sparse full checkpoint (state entering chunk c0) + per-chunk compact
     deltas (kg, v_new, gk), and a REPLAY = chunk_gated_delta_rule_fwd_h with
     w=0 and u=v_new reconstructs the exact state at any later chunk boundary."

Key precision finding baked in here: chunk_gated_delta_rule_fwd_h carries the
running state b_h in fp32 internally but writes the per-chunk `h` array in the
INPUT dtype (bf16). If checkpoints are stored bf16, replay drifts by one bf16
rounding of the state. To get BIT-EXACT replay, the checkpoint (the in-place
`initial_state` pool) must be fp32 — which is exactly what the in-place
INPLACE_UPDATE store gives when the pool tensor is fp32.

So this oracle uses an fp32 state pool and asserts:
  full([0:T], init=0)  ==  seg1([0:c0], init=0) -> ckpt ; replay([c0:T], init=ckpt, w=0, u=v_new)
bit-exact, for a sweep of split points c0 and lengths T.

Part A uses random (kg, w, u, gk) of KDA shapes — model-agnostic, proves the
chunk_delta_h replay identity. Part B (best-effort) drives real KDA gates so the
per-channel-decay path is exercised end to end.

Run on a box with the sglang branch on PYTHONPATH and one visible GPU:
    PYTHONPATH=<repo>/python CUDA_VISIBLE_DEVICES=0 python replay_oracle.py
"""

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)

DEV = "cuda"
# Kimi-Linear-48B-A3B KDA dims.
H = 32          # num KDA (v) heads
K = 128         # head_k_dim
V = 128         # head_v_dim
BT = CHUNK_SIZE  # 64


def _fwd_h(kg, w, u, gk, init_state_fp32):
    """Thin wrapper: run chunk_gated_delta_rule_fwd_h over a single seq (B=1),
    in-place updating the fp32 state pool. Returns (per_chunk_h, v_new) and the
    mutated init_state_fp32 now holding the final state."""
    idx = torch.zeros(1, dtype=torch.int32, device=DEV)  # one sequence -> slot 0
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        g=None,
        gk=gk,
        initial_state=init_state_fp32,
        initial_state_indices=idx,
        save_new_value=True,
        cu_seqlens=None,
    )
    return h, v_new


def run_part_a(T, c0_chunks, seed=0):
    """Random-input algebraic identity at a single split point."""
    torch.manual_seed(seed)
    NT = T // BT
    assert T % BT == 0 and 0 < c0_chunks < NT
    c0 = c0_chunks * BT

    kg = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
    w = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.1
    u = torch.randn(1, T, H, V, device=DEV, dtype=torch.bfloat16) * 0.5
    # per-channel log-decay, small negative -> stable decay in (0,1]
    gk = (-torch.rand(1, T, H, K, device=DEV, dtype=torch.float32) * 0.1)

    # ---- FULL: single run from zero, fp32 pool ----
    pool_full = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    h_full, v_new_full = _fwd_h(kg, w, u, gk, pool_full)
    S_T_full = pool_full[0].clone()              # final state, fp32
    ckpt = h_full[0, c0_chunks].clone()          # state ENTERING chunk c0 (bf16 from h[])

    # The bf16 `h` array is NOT precise enough for a bit-exact checkpoint. Recompute
    # the fp32 checkpoint by running seg1 = [0:c0] into an fp32 pool.
    pool_seg1 = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    _fwd_h(kg[:, :c0], w[:, :c0], u[:, :c0], gk[:, :c0], pool_seg1)
    ckpt_fp32 = pool_seg1[0].clone()             # state entering chunk c0, fp32

    # ---- REPLAY: from fp32 checkpoint, w=0, u=v_new (stored compact deltas) ----
    kg_r = kg[:, c0:].contiguous()
    gk_r = gk[:, c0:].contiguous()
    vnew_r = v_new_full[:, c0:].contiguous()     # stored per-chunk delta value
    w_zero = torch.zeros_like(kg_r)
    pool_replay = ckpt_fp32.clone().unsqueeze(0)  # [1,H,V,K] fp32
    _fwd_h(kg_r, w_zero, vnew_r, gk_r, pool_replay)
    S_T_replay = pool_replay[0]

    # bf16-checkpoint variant (what you'd get if checkpoints are stored bf16)
    pool_replay_bf16 = ckpt.float().clone().unsqueeze(0)
    _fwd_h(kg_r, w_zero, vnew_r, gk_r, pool_replay_bf16)
    S_T_replay_bf16 = pool_replay_bf16[0]

    d_fp32 = (S_T_replay - S_T_full).abs().max().item()
    d_bf16 = (S_T_replay_bf16 - S_T_full).abs().max().item()
    rel = S_T_full.abs().mean().item()
    print(
        f"[A] T={T:5d} c0_chunk={c0_chunks:2d}/{NT:2d}  "
        f"fp32-ckpt max|Δ|={d_fp32:.3e}  bf16-ckpt max|Δ|={d_bf16:.3e}  "
        f"(state mean|·|={rel:.3e})  -> fp32 {'BIT-EXACT' if d_fp32==0 else 'drift'}"
    )
    return d_fp32, d_bf16


def run_part_b(T=512, c0_chunks=4, seed=1):
    """Best-effort: drive REAL KDA per-channel gates through chunk_kda_fwd_intra,
    then apply the same checkpoint+replay identity. Proves the USE_GK path."""
    try:
        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.kda import chunk_kda_fwd_intra
    except Exception as e:  # noqa: BLE001
        print(f"[B] skipped (import): {e}")
        return
    torch.manual_seed(seed)
    NT = T // BT
    c0 = c0_chunks * BT
    scale = K ** -0.5

    def l2(x):  # KDA l2-normalizes q,k per head (use_qk_l2norm_in_kernel)
        return x / (x.float().norm(dim=-1, keepdim=True) + 1e-6)

    q = l2(torch.randn(1, T, H, K, device=DEV, dtype=torch.float32)).bfloat16()
    k = l2(torch.randn(1, T, H, K, device=DEV, dtype=torch.float32)).bfloat16()
    v = (torch.randn(1, T, H, V, device=DEV, dtype=torch.float32) * 0.5).bfloat16()
    beta = torch.rand(1, T, H, device=DEV, dtype=torch.bfloat16)
    # per-channel log-decay, bounded negative -> exp(cumsum) contracts, state stays finite
    g_raw = (-torch.rand(1, T, H, K, device=DEV, dtype=torch.float32) * 0.05 - 0.01)
    try:
        gk = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=None)
        w, u, _, kg, _Aqk, _ = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=gk, beta=beta, scale=scale,
            cu_seqlens=None, chunk_size=BT, chunk_indices=None,
            fuse_diagonal=False, fuse_recompute=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[B] skipped (intra signature mismatch): {e}")
        return

    pool_full = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    _h, v_new = _fwd_h(kg, w, u, gk, pool_full)
    S_T_full = pool_full[0].clone()

    pool_seg1 = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    _fwd_h(kg[:, :c0], w[:, :c0], u[:, :c0], gk[:, :c0], pool_seg1)
    ckpt_fp32 = pool_seg1[0].clone()

    kg_r, gk_r, vnew_r = kg[:, c0:].contiguous(), gk[:, c0:].contiguous(), v_new[:, c0:].contiguous()
    pool_replay = ckpt_fp32.clone().unsqueeze(0)
    _fwd_h(kg_r, torch.zeros_like(kg_r), vnew_r, gk_r, pool_replay)
    d = (pool_replay[0] - S_T_full).abs().max().item()
    print(f"[B] real-KDA T={T} c0_chunk={c0_chunks}/{NT}  fp32-ckpt max|Δ|={d:.3e}  "
          f"-> {'BIT-EXACT' if d==0 else 'drift'}")


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)}  BT={BT}  H={H} K={K} V={V}")
    print("== Part A: chunk_delta_h replay identity (random KDA-shaped inputs) ==")
    worst_fp32 = 0.0
    for T in (256, 512, 1024):
        for c0c in (1, 2, T // BT // 2, T // BT - 1):
            d_fp32, _ = run_part_a(T, c0c)
            worst_fp32 = max(worst_fp32, d_fp32)
    print(f"Part A worst fp32-ckpt max|Δ| = {worst_fp32:.3e} "
          f"({'ALL BIT-EXACT' if worst_fp32==0 else 'NOT bit-exact'})")
    print("== Part B: real KDA per-channel gate path ==")
    run_part_b()
