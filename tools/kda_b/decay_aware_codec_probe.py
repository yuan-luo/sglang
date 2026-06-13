"""Does a DECAY-AWARE checkpoint codec beat uniform int8?

Kimi-Linear-48B-A3B KDA channels span memory horizons 1..40000+ tokens (measured). Slow
channels (large, persistent state) are the int8 precision bottleneck; fast
channels (small, transient) tolerate int4. So allocate bits by static decay
alpha0 (from real A_log/dt_bias): slow -> int8, fast -> int4.

Build a realistic state with REAL per-channel gates (real A_log/dt_bias + synth
gate-projection input), then compare codecs on decode-output error vs avg bits.

    PYTHONPATH=<repo>/python CUDA_VISIBLE_DEVICES=0 python decay_aware_codec_probe.py
"""
import json
import os

import torch
from safetensors import safe_open

from sglang.srt.layers.attention.fla.kda import fused_recurrent_kda

DEV = "cuda"
MODEL = os.environ.get("KDA_MODEL_PATH", "/home/models/Kimi-Linear-48B-A3B")
H, K, V = 32, 128, 128
LB = -5.0
LAYER = 8
T_PRE, T_DEC = 512, 128


def load_layer(li):
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    out = {}
    for name in ("A_log", "dt_bias"):
        key = f"model.layers.{li}.attention.{name}"
        with safe_open(f"{MODEL}/{wm[key]}", framework="pt") as f:
            out[name] = f.get_tensor(key).float().to(DEV)
    return out["A_log"].reshape(H), out["dt_bias"].reshape(H, K)


def real_gate(raw_g, A_log, dt_bias):
    # gate = LB * sigmoid(exp(A_log_h) * (raw_g + dt_bias_{h,c})), per token log-decay
    sharp = torch.exp(A_log).view(1, 1, H, 1)
    return LB * torch.sigmoid(sharp * (raw_g + dt_bias.view(1, 1, H, K)))


def decode_from(state, q, k, v, g, beta):
    st = state.clone()
    o, _ = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5,
                              initial_state=st, inplace_final_state=True,
                              use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    return o.float()


def quant_per_kchan(S, bits, mask=None):
    """Quantize S [1,H,V,K] per (head,k-channel) to `bits`. mask [H,K] selects which
    k-channels to quantize at this bit width (others untouched)."""
    qmax = (1 << (bits - 1)) - 1
    amax = S.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # per (H,1,K)
    scale = amax / qmax
    Sq = (torch.round(S / scale).clamp(-qmax, qmax)) * scale
    if mask is None:
        return Sq
    m = mask.view(1, H, 1, K)
    return torch.where(m, Sq, S)


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def main():
    A_log, dt_bias = load_layer(LAYER)
    alpha0 = torch.exp(LB * torch.sigmoid(torch.exp(A_log).view(H, 1) * dt_bias))  # [H,K]
    horizon = 1.0 / (1 - alpha0).clamp(min=1e-6)
    print(f"layer {LAYER}: frac fast(<32tok)={ (horizon<32).float().mean():.3f} "
          f"slow(>256)={ (horizon>256).float().mean():.3f}")

    torch.manual_seed(0)
    def synth(T, s):
        torch.manual_seed(s)
        q = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
        k = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
        v = (torch.randn(1, T, H, V, device=DEV) * 0.5).bfloat16()
        beta = torch.rand(1, T, H, device=DEV, dtype=torch.bfloat16)
        raw_g = torch.randn(1, T, H, K, device=DEV, dtype=torch.float32)
        g = real_gate(raw_g, A_log, dt_bias)
        return q, k, v, g, beta

    qp, kp, vp, gp, bp = synth(T_PRE, 0)
    init = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    S = decode_from(init, qp, kp, vp, gp, bp)  # this returns outputs, need final state
    # rebuild to capture final state explicitly
    st = init.clone()
    fused_recurrent_kda(q=qp, k=kp, v=vp, g=gp, beta=bp, scale=K ** -0.5,
                        initial_state=st, inplace_final_state=True,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    S = st
    qd, kd, vd, gd, bd = synth(T_DEC, 1)
    o_ref = decode_from(S, qd, kd, vd, gd, bd)

    fast = horizon < 32   # ~46% channels
    slow = ~fast
    bits_fast = fast.float().mean().item()
    codecs = {
        "bf16             (16b)": (S.bfloat16().float(), 16.0),
        "uniform int8     ( 8b)": (quant_per_kchan(S, 8), 8.0),
        "uniform int4     ( 4b)": (quant_per_kchan(S, 4), 4.0),
        "decay int4fast/int8slow": (
            quant_per_kchan(quant_per_kchan(S, 8, mask=slow), 4, mask=fast),
            4.0 * bits_fast + 8.0 * (1 - bits_fast)),
        "decay int4fast/bf16slow": (
            torch.where(fast.view(1, H, 1, K), quant_per_kchan(S, 4), S.bfloat16().float()),
            4.0 * bits_fast + 16.0 * (1 - bits_fast)),
    }
    print(f"\n{'codec':26s} {'avg_bits':>8s} {'out_relerr':>11s} {'vs_bf16':>8s}")
    eb = rel(decode_from(codecs['bf16             (16b)'][0], qd, kd, vd, gd, bd), o_ref)
    for name, (Sq, bits) in codecs.items():
        oe = rel(decode_from(Sq, qd, kd, vd, gd, bd), o_ref)
        print(f"{name:26s} {bits:8.2f} {oe:11.3e} {oe/max(eb,1e-12):7.2f}x")
    print("\n(want: decay-aware at LOWER avg_bits but out_relerr near uniform-int8)")


if __name__ == "__main__":
    main()
