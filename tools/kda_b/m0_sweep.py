"""M0 client-only probe: does linear-attn (mamba) prefix reuse COLLAPSE as the
number of distinct cached prefixes grows past the mamba state-pool capacity,
while the token KV pool still has plenty of room?

Method (no server instrumentation, no restart):
  for each K in K_LIST:
    flush cache
    WARM:  send K distinct ~P-token prefixes once  (caches KV + 1 mamba checkpoint each)
    PROBE: send each prefix again with a different short suffix; read meta_info
           cached_tokens (= mamba-LIMITED reused prefix) and prompt_tokens.
    reuse_ratio = sum(cached_probe) / sum(prefix_tokens)

Interpretation:
  - KV pool (max_total_num_tokens ~4.5M) >> K*P, so token KV survives.
  - If the mamba state pool holds all K checkpoints -> reuse_ratio ~ chunk-align
    ceiling (~0.95, only <64-token alignment loss).
  - If K exceeds the mamba pool S_m -> earlier checkpoints evicted (KV alive) ->
    reuse_ratio drops. The K where it falls off ~ S_m. The depth = the prefix
    that's cached-but-recomputed -> exactly what direction B recovers.

    K_LIST=128,256,512,1024,2048 P=1000 CONC=24 python m0_sweep.py
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.environ.get("BASE", "http://127.0.0.1:30010")
URL = BASE + "/generate"
P = int(os.environ.get("P", "1000"))
CONC = int(os.environ.get("CONC", "24"))
SUF = int(os.environ.get("SUF", "8"))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "128,256,512,1024").split(",")]

VOCAB = (
    "time year people way day man thing woman life child world school state "
    "family student group country problem hand part place case week company "
    "system program question work government number night point home water room "
    "mother area money story fact month lot right study book eye job word business "
    "issue side kind head house service friend father power hour game line end member "
    "law car city community name president team minute idea body information back "
    "parent face others level office door health person art war history party result "
    "change morning reason research girl guy moment air teacher force education"
).split()


def make_prefix(i: int) -> str:
    rng = random.Random(1000 + i)
    return f"Document number {i} unique tag {i * 7919 % 100000}. " + " ".join(
        rng.choice(VOCAB) for _ in range(P)
    )


def send(prompt: str):
    try:
        r = requests.post(
            URL,
            json={"text": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}},
            timeout=300,
        )
        mi = r.json().get("meta_info", {})
        return (mi.get("prompt_tokens"), mi.get("cached_tokens"))
    except Exception as e:  # noqa: BLE001
        return ("err", str(e)[:80])


def flush():
    try:
        requests.post(BASE + "/flush_cache", timeout=60)
        time.sleep(1.5)
    except Exception:
        pass


def run_k(K: int):
    flush()
    prefixes = [make_prefix(i) for i in range(K)]
    warm = [p + " " + " ".join(random.Random(7 * i).choice(VOCAB) for _ in range(SUF))
            for i, p in enumerate(prefixes)]
    probe = [p + " " + " ".join(random.Random(13 * i + 1).choice(VOCAB) for _ in range(SUF))
             for i, p in enumerate(prefixes)]
    t0 = time.time()
    with ThreadPoolExecutor(CONC) as ex:
        list(ex.map(send, warm))  # WARM (populate cache)
        res = list(ex.map(send, probe))  # PROBE
    dt = time.time() - t0
    ok = [(pt, ct) for (pt, ct) in res if pt != "err" and pt]
    errs = [r for r in res if r[0] == "err"]
    if not ok:
        print(f"K={K:5d}  ALL ERR e.g. {errs[:1]}")
        return
    sum_prompt = sum(pt for pt, _ in ok)
    sum_cached = sum((ct or 0) for _, ct in ok)
    # per-prefix reuse ratio (cached / (prompt - suffix)); suffix ~SUF tokens
    ratios = [(ct or 0) / max(1, pt - SUF) for pt, ct in ok]
    ratios.sort()
    med = ratios[len(ratios) // 2]
    lowcnt = sum(1 for r in ratios if r < 0.5)
    print(
        f"K={K:5d}  reuse_frac={sum_cached/max(1,sum_prompt):.3f}  "
        f"median_ratio={med:.3f}  frac_prefix_reuse<0.5={lowcnt/len(ok):.3f}  "
        f"avg_prompt_tok={sum_prompt/len(ok):.0f}  probe_n={len(ok)} errs={len(errs)} ({dt:.0f}s)"
    )


def main():
    print(f"P={P} SUF={SUF} CONC={CONC} K_LIST={K_LIST} URL={URL}")
    print("(reuse_frac high & flat = mamba NOT a bottleneck; collapse as K grows = "
          "mamba pool S_m exceeded -> B's opportunity)")
    for K in K_LIST:
        run_k(K)


if __name__ == "__main__":
    main()
