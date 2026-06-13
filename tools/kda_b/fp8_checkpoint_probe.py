"""Is fp8/int8 compression of a CACHED KDA state viable?

Scenario: a prefix's recurrent state S is cached, stored COMPRESSED, then loaded
once and decoding continues in fp32/bf16. Unlike the decode-kernel-fp8 case (where
fp8 error compounds through the recurrence + amax overhead), here S is quantized
ONCE on store and never re-quantized — so the only error is a single rounding of S.

Test: build a realistic prefix state via fused_recurrent_kda over T_pre tokens;
store it as {fp32(ref), bf16, fp8-e4m3 per-head, fp8 per-(head,row), int8 per-head};
decode T_dec tokens from each with identical inputs; compare decode-output error to
the fp32 reference. If fp8_err ~ bf16_err, fp8 checkpoints are safe -> 2-4x more
cached prefixes fit -> M0 collapse point moves right.

    PYTHONPATH=<repo>/python CUDA_VISIBLE_DEVICES=0 python fp8_checkpoint_probe.py
"""
import torch

from sglang.srt.layers.attention.fla.kda import fused_recurrent_kda

DEV = "cuda"
H = 32
K = 128
V = 128
T_PRE = 512
T_DEC = 128


def synth(T, seed):
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
    v = (torch.randn(1, T, H, V, device=DEV) * 0.5).bfloat16()
    beta = torch.rand(1, T, H, device=DEV, dtype=torch.bfloat16)
    # per-channel log-decay (negative -> decay<1). Spread to mimic A_log structure.
    g = (-torch.rand(1, T, H, K, device=DEV, dtype=torch.float32) * 0.1 - 0.005)
    return q, k, v, g, beta


def decode_from(state_fp32, q, k, v, g, beta):
    st = state_fp32.clone()
    o, _ = fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5,
        initial_state=st, inplace_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )
    return o.float(), st  # outputs, final state


def q_bf16(S):
    return S.bfloat16().float()


def q_fp8(S, dims):
    amax = S.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    scale = amax / 448.0
    return (S / scale).to(torch.float8_e4m3fn).float() * scale


def q_int8(S, dims):
    amax = S.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    return torch.round(S / scale).clamp(-127, 127) * scale


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def main():
    print(f"device={torch.cuda.get_device_name(0)} H={H} K={K} V={V} T_pre={T_PRE} T_dec={T_DEC}")
    # build a realistic prefix state (fp32)
    qp, kp, vp, gp, bp = synth(T_PRE, seed=0)
    init = torch.zeros(1, H, V, K, device=DEV, dtype=torch.float32)
    _o, S = decode_from(init, qp, kp, vp, gp, bp)  # S = fp32 prefix state
    print(f"prefix state |S|: mean={S.abs().mean():.3e} max={S.abs().max():.3e}")

    # decode inputs (shared across all variants)
    qd, kd, vd, gd, bd = synth(T_DEC, seed=1)
    o_ref, sf_ref = decode_from(S, qd, kd, vd, gd, bd)

    variants = {
        "bf16              ": q_bf16(S),
        "fp8-e4m3 per-head ": q_fp8(S, dims=(-2, -1)),
        "fp8-e4m3 per-row  ": q_fp8(S, dims=(-1,)),
        "int8 per-head     ": q_int8(S, dims=(-2, -1)),
        "int8 per-row      ": q_int8(S, dims=(-1,)),
    }
    print(f"\n{'variant':20s} {'state_relerr':>12s} {'out_relerr@dec':>14s} {'final_relerr':>13s}  bytes/elem")
    sizes = {"bf16": 2, "fp8": 1, "int8": 1}
    for name, Sq in variants.items():
        o_q, sf_q = decode_from(Sq, qd, kd, vd, gd, bd)
        se = rel(Sq, S)             # state quantization error (store)
        oe = rel(o_q, o_ref)        # decode output error vs fp32-state ref
        fe = rel(sf_q, sf_ref)      # final state error after T_dec decode
        b = 1 if ("fp8" in name or "int8" in name) else 2
        print(f"{name} {se:12.3e} {oe:14.3e} {fe:13.3e}  {b}B")

    # headline: fp8 vs bf16 ratio on decode output error
    o_bf16, _ = decode_from(variants["bf16              "], qd, kd, vd, gd, bd)
    o_fp8h, _ = decode_from(variants["fp8-e4m3 per-head "], qd, kd, vd, gd, bd)
    o_fp8r, _ = decode_from(variants["fp8-e4m3 per-row  "], qd, kd, vd, gd, bd)
    eb = rel(o_bf16, o_ref)
    print(f"\nVERDICT: fp8-per-head out_err / bf16 out_err = {rel(o_fp8h, o_ref)/max(eb,1e-12):.2f}x"
          f"  | fp8-per-row / bf16 = {rel(o_fp8r, o_ref)/max(eb,1e-12):.2f}x")
    print("(<~2x and small absolute -> fp8 checkpoint viable: 2-4x more cached prefixes)")


if __name__ == "__main__":
    main()
