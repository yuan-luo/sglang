"""Tests for jit_kernel persistent_topk (ported from vLLM csrc/persistent_topk.cuh).

The kernel returns the indices of the top-k logits per row. Tie-breaking is
not specified to match torch's exact ordering, so we compare the *sets of
gathered logit values* (and verify they all live above the kth-largest value
of the reference top-k slice).
"""

import pytest
import torch

from sglang.jit_kernel.deepseek_v4 import (
    PERSISTENT_TOPK_WORKSPACE_BYTES,
    persistent_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


def _reference_topk(logits: torch.Tensor, lengths: torch.Tensor, k: int) -> torch.Tensor:
    """Reference top-k that handles per-row lengths and -inf padding.

    Returns indices sorted by descending logit value. For positions beyond
    `lengths[i]`, the reference treats those slots as -inf so they will never
    be picked when length >= k. When length < k the reference fills with -1
    at trailing positions, matching the kernel's trivial-case behavior.
    """
    num_rows = logits.shape[0]
    stride = logits.shape[1]
    out = torch.full((num_rows, k), -1, dtype=torch.int32, device=logits.device)
    for r in range(num_rows):
        length = int(lengths[r].item())
        if length <= 0:
            continue
        if length <= k:
            out[r, :length] = torch.arange(length, dtype=torch.int32, device=logits.device)
            continue
        # Mask values beyond `length` so they cannot enter top-k.
        row = logits[r, :length]
        vals, idx = torch.topk(row, k, sorted=True)
        out[r, :] = idx.to(torch.int32)
    return out


def _assert_topk_values_match(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    actual_idx: torch.Tensor,
    expected_idx: torch.Tensor,
    k: int,
):
    """Check that `actual_idx` selects the same multiset of logit values as
    `expected_idx` (modulo ties), respecting per-row lengths."""
    num_rows = logits.shape[0]
    for r in range(num_rows):
        length = int(lengths[r].item())
        if length <= 0:
            continue
        if length <= k:
            # Trivial case: must be 0..length-1 then -1 padding.
            got = actual_idx[r].tolist()
            assert sorted(got[:length]) == list(range(length)), (
                f"row {r}: expected indices 0..{length-1}, got {got[:length]}"
            )
            for j in range(length, k):
                assert got[j] == -1, f"row {r}: pad slot {j} must be -1, got {got[j]}"
            continue

        # Indices must be in-range [0, length)
        ai = actual_idx[r]
        assert (ai >= 0).all() and (ai < length).all(), (
            f"row {r}: actual indices out of [0, {length}): "
            f"min={int(ai.min())}, max={int(ai.max())}"
        )
        ei = expected_idx[r]
        actual_vals, _ = torch.sort(logits[r, ai.long()])
        expected_vals, _ = torch.sort(logits[r, ei.long()])
        # Sums match (tie-resilient strong check) and sorted sequences match
        # (extra-strong check; safe since float32 random data has near-zero
        # tie probability outside of integer test fixtures).
        torch.testing.assert_close(actual_vals, expected_vals, rtol=0, atol=0)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "num_rows,seq_len",
    [
        # Decode regime (seq_len <= 8192): hits histogram_2048_topk
        (1, 4096),
        (4, 4096),
        (16, 4096),
        (32, 4096),
        # Medium regime (8K < seq_len <= 64K): hits histogram_256_topk
        (1, 16384),
        (8, 16384),
        (32, 16384),
        # Large/cooperative regime (seq_len > 32K): hits multi-CTA radix
        (1, 65536),
        (4, 65536),
        (16, 65536),
        # num_rows > 32 path (CUB FilteredTopK)
        (64, 8192),
        (128, 4096),
    ],
)
def test_persistent_topk_correctness(num_rows: int, seq_len: int, k: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xDEEDB00C ^ (num_rows * 131 + seq_len * 17 + k))

    stride = seq_len
    logits = torch.randn((num_rows, stride), dtype=torch.float32, device="cuda")
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")

    out = persistent_topk(
        logits=logits,
        lengths=lengths,
        topk=k,
        max_seq_len=int(lengths.max().item()),
    )

    expected = _reference_topk(logits, lengths, k)
    _assert_topk_values_match(logits, lengths, out, expected, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_persistent_topk_unaligned_max_seq_len(k: int):
    """Covers vLLM #42169: numerical correctness with unaligned max_seq_len."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(31337)

    num_rows = 8
    seq_len = 31337  # deliberately non-power-of-two and non-VEC_SIZE-aligned
    logits = torch.randn((num_rows, seq_len), dtype=torch.float32, device="cuda")
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")

    out = persistent_topk(
        logits=logits,
        lengths=lengths,
        topk=k,
        max_seq_len=seq_len,
    )
    expected = _reference_topk(logits, lengths, k)
    _assert_topk_values_match(logits, lengths, out, expected, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_persistent_topk_ragged_lengths(k: int):
    """Per-row varying lengths, mixing every code path within a single launch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xCAFEF00D)

    # 4 rows covering: short (decode), medium, just-below-radix, large
    row_lens = [4096, 16384, 32000, 65536]
    seq_len = max(row_lens)
    num_rows = len(row_lens)
    logits = torch.randn((num_rows, seq_len), dtype=torch.float32, device="cuda")
    lengths = torch.tensor(row_lens, dtype=torch.int32, device="cuda")

    out = persistent_topk(
        logits=logits,
        lengths=lengths,
        topk=k,
        max_seq_len=max(row_lens),
    )
    expected = _reference_topk(logits, lengths, k)
    _assert_topk_values_match(logits, lengths, out, expected, k)


def test_persistent_topk_lengths_2d():
    """Accept 2D lengths shape (vLLM allows 1D or 2D)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(123)

    num_rows, seq_len, k = 8, 4096, 512
    logits = torch.randn((num_rows, seq_len), dtype=torch.float32, device="cuda")
    lengths_1d = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
    lengths_2d = lengths_1d.view(num_rows, 1).contiguous()

    out_1d = persistent_topk(
        logits=logits, lengths=lengths_1d, topk=k, max_seq_len=seq_len
    )
    out_2d = persistent_topk(
        logits=logits, lengths=lengths_2d, topk=k, max_seq_len=seq_len
    )
    # Indices may permute for ties; compare via gathered values per row.
    for r in range(num_rows):
        v1, _ = torch.sort(logits[r, out_1d[r].long()])
        v2, _ = torch.sort(logits[r, out_2d[r].long()])
        torch.testing.assert_close(v1, v2, rtol=0, atol=0)


def test_persistent_topk_all_rows_shorter_than_k():
    """Trivial-case path: every row has length < k."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(7)

    num_rows, seq_len, k = 4, 256, 512  # seq_len < k
    logits = torch.randn((num_rows, k), dtype=torch.float32, device="cuda")
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")

    out = persistent_topk(
        logits=logits, lengths=lengths, topk=k, max_seq_len=seq_len
    )
    # Expect 0..seq_len-1 in some order, then -1 padding.
    for r in range(num_rows):
        row = out[r].tolist()
        assert sorted(row[:seq_len]) == list(range(seq_len))
        assert all(x == -1 for x in row[seq_len:])


def test_persistent_topk_user_workspace_too_small():
    """User-provided workspace below the minimum must raise a clear error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    logits = torch.randn((1, 4096), dtype=torch.float32, device="cuda")
    lengths = torch.tensor([4096], dtype=torch.int32, device="cuda")
    too_small = torch.empty(1024, dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError):
        persistent_topk(
            logits=logits,
            lengths=lengths,
            topk=512,
            max_seq_len=4096,
            workspace=too_small,
        )


def test_persistent_topk_returns_user_out():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    logits = torch.randn((2, 4096), dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    pre = torch.empty((2, 512), dtype=torch.int32, device="cuda")
    out = persistent_topk(
        logits=logits, lengths=lengths, topk=512, max_seq_len=4096, out=pre
    )
    assert out.data_ptr() == pre.data_ptr()


def test_persistent_topk_workspace_constant_is_reasonable():
    # Lightweight smoke: the public constant must be at least the doc minimum.
    assert PERSISTENT_TOPK_WORKSPACE_BYTES >= (1 << 20)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
