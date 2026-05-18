"""Benchmark for jit_kernel persistent_topk vs DSv4 production topk paths.

DSv4 indexer runs on compressed K (compress_ratio=4), so:
  topk seq_len = ceil(model_ctx / 4)

  - DSv4-Pro   index_topk=1024 → routes to topk_1024.cuh / topk_v2.cuh
  - DSv4-Flash index_topk=512  → routes to topk.cuh       / topk_v2.cuh

Both production wrappers do **fused topk + page-table gather**; our
persistent_topk returns raw indices only. The time difference therefore
mixes pure topk cost with the per-call gather, but it directly answers
"would a drop-in replacement be faster end-to-end at this layer".

Providers (K is the index_topk value):
  - persistent_topk_k512   : this PR, k=512 (Flash target)
  - persistent_topk_k1024  : this PR, k=1024 (Pro target)
  - dsv4_topk_k512         : topk_transform_512 dispatched to topk.cuh
                             (Flash production single-CTA-per-row baseline)
  - dsv4_topk_k1024        : topk_transform_512 dispatched to topk_1024.cuh
                             (Pro production single-CTA-per-row baseline)
  - dsv4_topk_v2_k1024     : topk_transform_512_v2 with SGL_TOPK=1024
                             (Pro production cluster / multi-CTA baseline)
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.deepseek_v4 import (
    PERSISTENT_TOPK_WORKSPACE_BYTES,
    persistent_topk,
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="base-b-kernel-benchmark-1-gpu-large")

NUM_ROWS_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 32, 64],
    ci_range=[1, 16],
)
SEQ_LEN_RANGE = get_benchmark_range(
    full_range=[2048, 8192, 32768, 65536, 131072, 262144],
    ci_range=[8192, 32768],
)
# DSv4 indexer K cache uses paged layout — pick a representative page size.
PAGE_SIZE = 64

configs = list(itertools.product(NUM_ROWS_RANGE, SEQ_LEN_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_rows", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "persistent_topk_k512",
            "dsv4_topk_k512",
            "persistent_topk_k1024",
            "dsv4_topk_k1024",
            "dsv4_topk_v2_k1024",
        ],
        line_names=[
            "persistent_topk k=512 (new, Flash)",
            "dsv4_topk k=512 (current, Flash)",
            "persistent_topk k=1024 (new, Pro)",
            "dsv4_topk k=1024 (current, Pro)",
            "dsv4_topk_v2 k=1024 (current, Pro v2)",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("blue", "--"),
            ("green", "--"),
            ("orange", "--"),
        ],
        ylabel="us",
        plot_name="persistent-topk-vs-dsv4-baselines",
        args={},
    )
)
def benchmark(num_rows: int, seq_len: int, provider: str):
    k = 512 if provider.endswith("k512") else 1024

    scores = torch.randn((num_rows, seq_len), dtype=torch.float32, device=DEFAULT_DEVICE)
    seq_lens = torch.full((num_rows,), seq_len, dtype=torch.int32, device=DEFAULT_DEVICE)

    # persistent_topk: raw-indices output + 1MB workspace.
    workspace = torch.empty(
        PERSISTENT_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    out_raw = torch.empty((num_rows, k), dtype=torch.int32, device=DEFAULT_DEVICE)

    # dsv4_topk / dsv4_topk_v2: fused gather inputs.
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_tables = torch.arange(
        num_rows * num_pages, dtype=torch.int32, device=DEFAULT_DEVICE
    ).view(num_rows, num_pages)
    out_page_indices = torch.empty(
        (num_rows, k), dtype=torch.int32, device=DEFAULT_DEVICE
    )

    if provider.startswith("persistent_topk_"):
        fn = lambda: persistent_topk(
            logits=scores,
            lengths=seq_lens,
            topk=k,
            max_seq_len=seq_len,
            workspace=workspace,
            out=out_raw,
        )
    elif provider.startswith("dsv4_topk_v2_"):
        # Plan once per config (planning is amortized across layers in prod).
        metadata = plan_topk_v2(seq_lens)
        fn = lambda: topk_transform_512_v2(
            scores, seq_lens, page_tables, out_page_indices, PAGE_SIZE, metadata
        )
    else:  # dsv4_topk_{k512,k1024}
        fn = lambda: topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, PAGE_SIZE
        )

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
