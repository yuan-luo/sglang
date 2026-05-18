"""Benchmark for jit_kernel persistent_topk.

Compares against torch.topk. Primary win region is num_rows in [1, 32]
(decode at small batch). For larger num_rows the persistent kernel falls
back internally to CUB FilteredTopK.
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
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")

NUM_ROWS_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 32, 64],
    ci_range=[1, 16],
)
SEQ_LEN_RANGE = get_benchmark_range(
    full_range=[2048, 8192, 32768, 131072],
    ci_range=[8192, 32768],
)
K = 2048

configs = list(itertools.product(NUM_ROWS_RANGE, SEQ_LEN_RANGE))


def torch_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    _, idx = torch.topk(logits, k, dim=-1, sorted=False)
    return idx.to(torch.int32)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_rows", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit_persistent", "torch"],
        line_names=["SGL JIT persistent_topk", "torch.topk"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="persistent-topk-performance",
        args={},
    )
)
def benchmark(num_rows: int, seq_len: int, provider: str):
    logits = torch.randn((num_rows, seq_len), dtype=torch.float32, device=DEFAULT_DEVICE)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=DEFAULT_DEVICE)
    workspace = torch.empty(
        PERSISTENT_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    out = torch.empty((num_rows, K), dtype=torch.int32, device=DEFAULT_DEVICE)

    if provider == "jit_persistent":
        fn = lambda: persistent_topk(
            logits=logits,
            lengths=lengths,
            topk=K,
            max_seq_len=seq_len,
            workspace=workspace,
            out=out,
        )
    else:
        fn = lambda: torch_topk(logits, K)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
