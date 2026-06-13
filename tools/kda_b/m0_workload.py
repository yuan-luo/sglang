"""M0 branchy workload: many DISTINCT long prefixes (multi-doc / multi-session
pattern). Drives the Kimi-Linear-48B-A3B server so its radix tree caches K distinct prefixes.
If the mamba state pool can't hold K checkpoints, earlier ones get evicted
(internal tombstone) while the token KV survives -> on re-access the token prefix
still matches but the SSM state is gone -> forced recompute. The server-side
[M0STATS] log (from m0_patch.py) captures full vs mamba match tokens.

Knobs (env): K distinct prefixes, P prefix words (~tokens), ROUNDS re-access
passes, CONC concurrency, SUF unique-suffix words. Tune K > mamba_pool_slots and
K*P_tokens < kv_token_capacity so the asymmetry is exposed.

    K=96 P=1500 ROUNDS=4 CONC=8 python m0_workload.py
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = os.environ.get("URL", "http://127.0.0.1:30010/generate")
K = int(os.environ.get("K", "96"))
P = int(os.environ.get("P", "1500"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))
CONC = int(os.environ.get("CONC", "8"))
SUF = int(os.environ.get("SUF", "12"))

# Fixed vocab -> stable tokenization, genuinely distinct prefixes per seed.
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
    # unique leading marker so prefixes don't accidentally share a long head
    return f"Document number {i} unique tag {i*7919 % 100000}. " + " ".join(
        rng.choice(VOCAB) for _ in range(P)
    )


PREFIXES = [make_prefix(i) for i in range(K)]


def send(prompt: str):
    try:
        r = requests.post(
            URL,
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            },
            timeout=300,
        )
        mi = r.json().get("meta_info", {})
        return {
            "prompt_tokens": mi.get("prompt_tokens"),
            "cached_tokens": mi.get("cached_tokens"),
        }
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:120]}


def build_requests():
    reqs = []
    for rnd in range(ROUNDS):
        order = list(range(K))
        random.Random(rnd).shuffle(order)
        for i in order:
            rng = random.Random(99999 * rnd + i)
            suffix = " " + " ".join(rng.choice(VOCAB) for _ in range(SUF))
            reqs.append(PREFIXES[i] + suffix)
    return reqs


def main():
    reqs = build_requests()
    print(f"K={K} P={P} ROUNDS={ROUNDS} CONC={CONC} total_reqs={len(reqs)} URL={URL}")
    t0 = time.time()
    with ThreadPoolExecutor(CONC) as ex:
        metas = list(ex.map(send, reqs))
    dt = time.time() - t0
    errs = [m for m in metas if "err" in m]
    ok = [m for m in metas if "err" not in m and m.get("prompt_tokens")]
    if errs:
        print(f"errors={len(errs)} e.g. {errs[0]}")
    if ok:
        tot_prompt = sum(m["prompt_tokens"] for m in ok)
        tot_cached = sum((m.get("cached_tokens") or 0) for m in ok)
        # cached_tokens is the server-reported reused prefix (mamba-limited).
        print(
            f"done {len(ok)} ok in {dt:.1f}s | sum_prompt_tok={tot_prompt} "
            f"sum_cached_tok={tot_cached} client_cache_frac={tot_cached/max(1,tot_prompt):.3f}"
        )
        print(
            "NOTE: client cached_tokens is mamba-limited reuse; compare with "
            "server [M0STATS] waste_frac for the full-vs-mamba gap."
        )


if __name__ == "__main__":
    main()
