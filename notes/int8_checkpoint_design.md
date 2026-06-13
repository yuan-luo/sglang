# int8 Checkpoint Compression for Linear-Attention Prefix Cache

> Compress the linear-attention (KDA / GDN) recurrent states that the radix
> prefix cache stores, from bf16 to **int8-per-(head,k-channel)** → ~2× cached
> capacity at fixed GPU memory, no measurable quality loss. Attacks the measured
> mamba-pool bottleneck. Validated end-to-end on Kimi-Linear-48B-A3B.

Symbols follow [`gdn_kda_mamba_relationship.md`](./gdn_kda_mamba_relationship.md). This
supersedes the "direction B" (per-token delta prefix cache) idea in
[`kda_b_design.md`](./kda_b_design.md) — see §1.

---

## 1. Why not "direction B" (per-token deltas): the recovery-point-density rule

The optimal cache representation for a recurrent state depends on **how densely
you must recover the state**:

$$
\underbrace{\text{1 recovery / sequence}}_{\text{prefix cache}}
\;\;\cdots\;\;
\underbrace{\text{1 recovery / }\sim\!125\text{ tok}}_{\text{break-even}}
\;\;\cdots\;\;
\underbrace{\text{1 recovery / token}}_{\text{spec-decode / beam}}
$$

A full checkpoint (one bf16 photo of $S$) costs the same as ~125 per-token deltas
(measured: full state ≈ 70 MB across all linear layers; one token's
$(k_g,v_{\text{new}})$ delta ≈ 0.56 MB). So:

- **Dense recovery (every token)** → deltas win 50–125× → that's spec-decode
  target-verify / beam (PR #27658 / "direction A").
- **Sparse recovery (once per cached sequence)** → the full checkpoint already is
  minimal; per-token deltas cost **~8× MORE**. Prefix cache is here.

The M0 measurement (`tools/kda_b/m0_sweep.py`, Kimi-Linear-48B-A3B) showed the real bottleneck
is **capacity of the full-checkpoint pool**, not granularity: mamba pool holds
~1200 full checkpoints; reuse_frac collapses 0.94 → 0.62 → **0.055** as distinct
~1K-token prefixes grow 1024 → 1536 → 2048, while the 4.5M-token KV pool is barely
touched. So the fix is **cheaper checkpoints**, not deltas.

## 2. Why int8 (and why it is safe)

A cached checkpoint is **loaded once on a cache hit, then decoding continues in
bf16/fp32** — the quantization error is a single rounding of $S$; it never
re-enters the recurrence-with-quant that makes decode-kernel fp8 both lossy and
slow (amax overhead off the roofline). So the storage codec can be aggressive.

Probed on Kimi-Linear-48B-A3B (`tools/kda_b/`):

| codec | bytes | state relerr | decode-out relerr (128 step) | vs bf16 |
|---|---|---|---|---|
| bf16 (baseline) | 2 | 1.7e-3 | 9.2e-4 | 1× |
| **int8 per-(head,k-chan)** | **1** | 6.5e-3 | **2.1e-3** | **2.3×** |
| fp8-e4m3 | 1 | 2.6e-2 | 7.1e-3 | 7.8× |

**int8 beats fp8 at the same 1 byte** — the KDA state is ~uniformly distributed,
so fp8 wastes bits on the exponent; int8's uniform grid fits it. End-to-end:

> **GSM8K: int8 = 0.888 vs bf16 = 0.898** (500 q, ±1.4% sampling noise) →
> **statistically indistinguishable → quality-safe.**
> (probe: `tools/kda_b/patch_int8_ckpt.py`, file-flag int8-round-trips the temporal
> state in `MambaPool.copy_from`, the cache-hit COW path, on the live server.)

**Net: ~2× cached-prefix capacity at fixed memory** → M0 collapse point moves
1200 → ~2400. Composes with host-offload (HiMambaRadixCache): int8 also halves
the host↔device transfer it bottlenecks on.

## 3. Deeper lever (lower priority): decay-aware bit allocation

Kimi-Linear-48B-A3B's per-channel static decay $\alpha_0$ spans memory horizons **1 .. 40000+
tokens** (real `A_log`/`dt_bias`, `tools/kda_b/alog_structure_probe.py`): ~46% of
channels have memory < 32 tok (fast, transient, small state), ~27% > 256 tok
(slow, persistent, large — the int8 precision floor). A decay-aware codec (int4
fast / int8 slow) reaches int8 accuracy at **~6.5 bits** (≈ +20% beyond int8;
`decay_aware_codec_probe.py`), but needs mixed-bit storage + int4 kernels → keep
as a documented follow-up, not v1.

## 4. Design — separate int8 checkpoint store

`int8_checkpoint_store.py :: Int8CheckpointStore`. It is the **cache layer**, NOT
the compute layer:

```
running req (decode/prefill): state S in ACTIVE MambaPool (bf16/fp32), kernels read it
                              ── int8 store NEVER touches the running state ──
   req finishes / chunk ends  →  quantize bf16→int8, store in Int8CheckpointStore   (per cached prefix, 1×)
   cache HIT on that prefix   →  dequantize int8→bf16 into a fresh active slot       (per hit, 1×)
```

- Active pool (bf16, size $A$) serves running requests; kernels unchanged.
- `Int8CheckpointStore` (int8, size $C$): `qdata [L, C, H, d_v, d_k]` int8 +
  `scale [L, C, H, 1, d_k]` (per layer/slot/head/k-channel, reduce over $d_v$).
  `bytes/slot ≈ ½` bf16 → for the freed memory, $C ≈ 2A$.
- Today the radix `mamba_value` indexes an **active** slot (request donates it),
  so cached states compete with running ones. The change: radix caches into the
  int8 store, freeing active slots.

### 4.1 Integration points (HybridReqToTokenPool + MambaRadixCache)
1. `HybridReqToTokenPool._init_mamba_pool` (flag `--enable-int8-mamba-checkpoint`,
   `--int8-mamba-ckpt-multiplier` default 2): build `Int8CheckpointStore` +
   `MambaSlotAllocator` sized `multiplier × mamba_size`.
2. **store** (`cache_finished_req` / `cache_unfinished_req`): instead of donating
   the active slot, `store_from_bf16_pool(active.temporal, active_slot, ckpt_slot)`;
   `mamba_value = ckpt_slot`; free the active slot.
3. **hit COW** (`_match_post_processor` → forward `copy_from`): replace the
   active→active copy with `copy_to_bf16_pool(active.temporal, ckpt_slot, dst_active_slot)`.
4. **evict** (`evict_mamba`): free from the ckpt allocator.
5. conv state stays bf16 in the active pool (tiny, $W{-}1$ window); only the SSM
   temporal state is int8'd.

### 4.2 Correctness / risk
- Lossless-enough proven (GSM8K). Per-channel symmetric int8, scale reduces over
  $d_v$ — matches the per-k-channel decay axis.
- Edge: a state cached then immediately hit must round-trip consistently (tested
  in `test_int8_checkpoint_store.py::test_cow_helpers`).
- Spec-decode intermediate states are a separate path (full state per draft token)
  — out of scope here.

## 5. M0 reproduction plan (capacity → reuse recovery)
After wiring: run `tools/kda_b/m0_sweep.py` at K = 512..3072 with int8 ON vs OFF at
the **same GPU memory**. Expect the collapse knee to move from K≈1200 to K≈2400,
i.e. reuse_frac at K=2048 recovers from ~0.05 toward ~0.94. Plus GSM8K to confirm
quality holds at the new capacity.

## 6. Status
- ✅ Codec + `Int8CheckpointStore` module + tests (`test/srt/mem_cache/test_int8_checkpoint_store.py`, 6/6 pass incl. GPU decode-error).
- ✅ int8 quality validated end-to-end (GSM8K 0.888 vs 0.898) via the COW-path probe.
- ⏳ Production wiring (§4.1) + M0 reproduction (§5).

## Files
| file | role |
|---|---|
| `python/sglang/srt/mem_cache/int8_checkpoint_store.py` | the int8 store (codec + storage + COW helpers) |
| `test/srt/mem_cache/test_int8_checkpoint_store.py` | unit + GPU decode-error tests |
| `tools/kda_b/fp8_checkpoint_probe.py` | int8-vs-fp8-vs-bf16 codec accuracy probe |
| `tools/kda_b/alog_structure_probe.py` | per-channel decay structure (decay-aware headroom) |
| `tools/kda_b/decay_aware_codec_probe.py` | decay-aware int4/int8 codec frontier |
| `tools/kda_b/patch_int8_ckpt.py` | live-server int8 quality probe (COW-path round-trip) |
| `tools/kda_b/m0_sweep.py` | mamba prefix-reuse collapse measurement |
