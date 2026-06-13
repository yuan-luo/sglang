"""Does Kimi-Linear-48B-A3B's KDA have exploitable fast/slow per-channel decay structure?

Kimi-Linear-48B-A3B uses the SAFE gate (kda_safe_gate=True, lower_bound=-5.0). Per-token,
per-channel log-decay:
    gate = lower_bound * sigmoid(exp(A_log_h) * (raw_g + dt_bias_{h,c})),  alpha = exp(gate)
where A_log is PER-HEAD [H] (sigmoid sharpness) and dt_bias is PER-CHANNEL [H,K]
(the bias point that sets each channel fast vs slow). At static input (raw_g=0)
each channel has alpha0 = exp(lower_bound * sigmoid(exp(A_log_h)*dt_bias_{h,c})),
ranging exp(-5)=0.0067 (fast, ~1-token memory) .. ~1 (slow, long memory).

If alpha0 spreads widely across channels, a decay-aware codec (precise store for
slow/persistent channels, coarse or recompute-from-recent for fast channels) can
beat uniform int8.

    python alog_structure_probe.py   # on the box; reads /home/models/Kimi-Linear-48B-A3B
"""
import json
import os

import torch
from safetensors import safe_open

LOWER_BOUND = -5.0
MODEL = os.environ.get("KDA_MODEL_PATH", "/home/models/Kimi-Linear-48B-A3B")


def load_kda_params():
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    layers = {}
    for k, shard in wm.items():
        if k.endswith(".attention.A_log") or k.endswith(".attention.dt_bias"):
            li = int(k.split(".layers.")[1].split(".")[0])
            layers.setdefault(li, {})[k.split(".")[-1]] = (shard, k)
    out = {}
    cache = {}
    for li, d in layers.items():
        rec = {}
        for name, (shard, key) in d.items():
            if shard not in cache:
                cache[shard] = safe_open(f"{MODEL}/{shard}", framework="pt")
            rec[name] = cache[shard].get_tensor(key).float()
        out[li] = rec
    return out


def main():
    params = load_kda_params()
    lis = sorted(params)
    print(f"KDA layers: {len(lis)}  (layer ids {lis[0]}..{lis[-1]})")
    a0 = params[lis[0]]["A_log"]
    db0 = params[lis[0]]["dt_bias"]
    print(f"A_log shape={tuple(a0.shape)}  dt_bias shape={tuple(db0.shape)}")

    H = a0.numel()
    all_alpha = []
    print(f"\n{'layer':>5s} {'static_decay_alpha0 p1/p50/p99':>34s}  {'frac fast(<32tok)':>17s}")
    for li in lis:
        A_log = params[li]["A_log"].flatten()  # [H]
        dt_bias = params[li]["dt_bias"].flatten().reshape(H, -1)  # [H, K]
        sharp = torch.exp(A_log).unsqueeze(1)  # [H,1]
        gate0 = LOWER_BOUND * torch.sigmoid(sharp * dt_bias)  # static log-decay [H,K]
        alpha = torch.exp(gate0).flatten()  # per-channel static decay
        all_alpha.append(alpha)
        q = lambda t, p: torch.quantile(t, p).item()
        horizon = 1.0 / (1 - alpha).clamp(min=1e-6)
        print(f"{li:5d}  {q(alpha,.01):.4f}/{q(alpha,.5):.4f}/{q(alpha,.99):.4f}"
              f"          {(horizon<32).float().mean().item():.3f}")

    alpha = torch.cat(all_alpha)
    print(f"\n=== ALL channels ({alpha.numel()}) ===")
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        a = torch.quantile(alpha, p).item()
        print(f"  static_decay_alpha0 p{int(p*100):02d} = {a:.5f}   "
              f"~memory_horizon = {1.0/max(1e-6, 1-a):.1f} tok")
    # fraction of "fast" channels (short memory horizon) vs "slow"
    horizon = 1.0 / (1 - alpha).clamp(min=1e-6)
    frac_fast = (horizon < 32).float().mean().item()
    frac_slow = (horizon > 256).float().mean().item()
    print(f"  frac channels memory<32tok (fast) = {frac_fast:.3f}   "
          f"memory>256tok (slow) = {frac_slow:.3f}")
    print("  -> wide spread + sizable fast fraction = decay-aware codec has headroom")


if __name__ == "__main__":
    main()
