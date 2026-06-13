"""Validate the promoted LinearDeltaPool module (bit-exact varlen round-trip)."""
import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)
from sglang.srt.mem_cache.linear_delta_pool import LinearDeltaPool, LinearDeltaShape

DEV = "cuda"
BT = CHUNK_SIZE
H = 32
K = 128
V = 128


def fwd(kg, w, u, gk, init, idx, cu):
    return chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, g=None, gk=gk, initial_state=init,
        initial_state_indices=idx, save_new_value=True, cu_seqlens=cu,
    )


def fp32_ckpt(kg, w, u, gk, a, b):
    pool = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32, device=DEV)
    fwd(kg[a:b][None].contiguous(), w[a:b][None].contiguous(),
        u[a:b][None].contiguous(), gk[a:b][None].contiguous(), pool, idx, None)
    return pool[0]


def make(seqlens, seed, real):
    torch.manual_seed(seed)
    Tt = sum(seqlens)
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()],
                      dtype=torch.int32, device=DEV)
    n = len(seqlens)
    if real:
        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.kda import chunk_kda_fwd_intra
        l2 = lambda x: x / (x.float().norm(dim=-1, keepdim=True) + 1e-6)
        q = l2(torch.randn(1, Tt, H, K, device=DEV)).bfloat16()
        k = l2(torch.randn(1, Tt, H, K, device=DEV)).bfloat16()
        v = (torch.randn(1, Tt, H, V, device=DEV) * 0.5).bfloat16()
        beta = torch.rand(1, Tt, H, device=DEV, dtype=torch.bfloat16)
        g_raw = -torch.rand(1, Tt, H, K, device=DEV, dtype=torch.float32) * 0.05 - 0.01
        gk = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu)
        w, u, _, kg, _a, _b = chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=gk, beta=beta, scale=K ** -0.5, cu_seqlens=cu,
            chunk_size=BT, chunk_indices=None, fuse_diagonal=False, fuse_recompute=False)
    else:
        kg = torch.randn(1, Tt, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
        w = torch.randn(1, Tt, H, K, device=DEV, dtype=torch.bfloat16) * 0.1
        u = torch.randn(1, Tt, H, V, device=DEV, dtype=torch.bfloat16) * 0.5
        gk = -torch.rand(1, Tt, H, K, device=DEV, dtype=torch.float32) * 0.05 - 0.01
    init = torch.zeros(n, H, V, K, device=DEV, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32, device=DEV)
    _h, vnew = fwd(kg, w, u, gk, init, idx, cu)
    return kg[0], vnew[0], gk[0], w[0], u[0], init.clone(), cu


def run(seqlens, C_chunks, real):
    kg, vnew, gk, w, u, finals, cu = make(seqlens, 0, real)
    pool = LinearDeltaPool(
        num_layers=1, max_delta_tokens=sum(seqlens), max_ckpt_slots=2000,
        shape=LinearDeltaShape(num_k_heads=H, num_v_heads=H, head_k_dim=K, head_v_dim=V),
        ckpt_interval=C_chunks * BT, device=DEV, per_channel_decay=True)
    worst = 0.0
    ck = 0
    for s, T in enumerate(seqlens):
        a0, a1 = int(cu[s]), int(cu[s + 1])
        tok = torch.arange(a0, a1, device=DEV)
        pool.store_deltas(0, tok, kg[a0:a1], vnew[a0:a1], gk[a0:a1])
        NT = (T + BT - 1) // BT
        ckchunks = list(range(0, NT, C_chunks))
        slot_of = {}
        for c in ckchunks:
            st = fp32_ckpt(kg[a0:a1], w[a0:a1], u[a0:a1], gk[a0:a1], 0, c * BT)
            pool.store_checkpoint(0, ck, st)
            slot_of[c] = ck
            ck += 1
        c0 = max(ckchunks)  # nearest ckpt <= final
        replay = torch.arange(a0 + c0 * BT, a1, device=DEV)
        rec = pool.reconstruct(0, slot_of[c0], replay)
        worst = max(worst, (rec - finals[s]).abs().max().item())
    tag = "real" if real else "rand"
    print(f"[{tag}] seqlens={seqlens} C={C_chunks}  max|Δ|={worst:.3e} "
          f"-> {'BIT-EXACT' if worst == 0 else 'DRIFT'}")
    return worst


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)} module=LinearDeltaPool")
    worst = 0.0
    for sl in ([192, 320, 64], [64, 640, 128, 256]):
        for C in (1, 2, 4):
            worst = max(worst, run(sl, C, real=False))
    worst = max(worst, run([192, 320, 128], 2, real=True))
    print(f"WORST max|Δ| = {worst:.3e} ({'ALL BIT-EXACT' if worst == 0 else 'FAIL'})")
