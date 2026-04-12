/*
 * KimiLinear Fused Gate Kernel: Router GEMM + Sigmoid + Grouped TopK
 *
 * Fuses the two-kernel pipeline (dsv3_router_gemm + biased_grouped_topk)
 * into a single kernel launch for KimiLinear MoE models.
 *
 * Phase 1: All 128 blocks compute GEMM [M, 5120] x [128, 5120]^T -> [M, 128]
 * Phase 2: Last completing block performs sigmoid + bias + grouped topk
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cfloat>

#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include "utils.h"

static constexpr int KIMI_LINEAR_NUM_EXPERTS = 128;
static constexpr int KIMI_LINEAR_HIDDEN_DIM = 5120;

// Convert 8 bfloat16 values from a uint4 to float array
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float(uint4 const& vec, float* dst) {
  __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

template <typename T, int kBlockSize, int VPT, int kNumTokens>
__global__ __launch_bounds__(128, 1) void kimi_linear_fused_gate_kernel(
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    T const* __restrict__ mat_a,
    T const* __restrict__ mat_b,
    float const* __restrict__ correction_bias,
    float* __restrict__ scratch,
    int* __restrict__ block_counter,
    int topk,
    int num_expert_group,
    int topk_group,
    bool renormalize) {
  constexpr int kNumExperts = KIMI_LINEAR_NUM_EXPERTS;
  constexpr int kHiddenDim = KIMI_LINEAR_HIDDEN_DIM;

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  // ========== Phase 1: GEMM ==========
  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  T const* b_col = mat_b + n_idx * kHiddenDim;

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);
    float b_float[VPT];
    bf16_uint4_to_float<VPT>(b_vec, b_float);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);
      float a_float[VPT];
      bf16_uint4_to_float<VPT>(a_vec, a_float);

#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  // Warp-level reduction
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  // Final reduction across warps and write to scratch
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }
      scratch[m * kNumExperts + n_idx] = final_sum;
    }
  }

  // ========== Synchronization: atomic completion counter ==========
  __threadfence();
  __syncthreads();

  int ticket = 0;
  if (tid == 0) {
    ticket = atomicAdd(block_counter, 1);
  }
  ticket = __shfl_sync(0xffffffff, ticket, 0);
  // Broadcast ticket to all threads via shared memory
  __shared__ int shared_ticket;
  if (tid == 0) {
    shared_ticket = ticket;
  }
  __syncthreads();
  ticket = shared_ticket;

  if (ticket != kNumExperts - 1) {
    return;  // Only the last block proceeds to Phase 2
  }

  // ========== Phase 2: Warp-per-Token Parallel Sigmoid + Grouped TopK ==========
  // Each warp independently processes one token per pass.
  // 128 experts / 32 lanes = 4 experts per thread, held in registers.
  // TopK uses intra-warp butterfly reduction (__shfl_xor_sync), no __syncthreads.

  constexpr int kExpertsPerThread = kNumExperts / kWarpSize;  // 128/32 = 4
  constexpr int kTokensPerPass = kNumWarps;  // 4

  // Shared memory for group selection path (only used when num_expert_group > 1)
  __shared__ float smem_group[kNumWarps][kNumExperts];

  for (int pass = 0; pass < (kNumTokens + kTokensPerPass - 1) / kTokensPerPass; pass++) {
    int m = pass * kTokensPerPass + warpId;
    if (m >= kNumTokens) break;

    // Step 1: Load logits from scratch, compute sigmoid + bias → registers
    float reg_score[kExpertsPerThread];
    float reg_biased[kExpertsPerThread];

#pragma unroll
    for (int e = 0; e < kExpertsPerThread; e++) {
      int expert_idx = laneId * kExpertsPerThread + e;
      float logit = scratch[m * kNumExperts + expert_idx];
      reg_score[e] = 1.0f / (1.0f + expf(-logit));
      reg_biased[e] = correction_bias ? (reg_score[e] + correction_bias[expert_idx]) : reg_score[e];
    }

    // Step 2: Group selection (only if num_expert_group > 1)
    if (num_expert_group > 1) {
      int experts_per_group = kNumExperts / num_expert_group;

      // Write biased scores to warp-local shared memory for lane 0 to process
#pragma unroll
      for (int e = 0; e < kExpertsPerThread; e++) {
        int expert_idx = laneId * kExpertsPerThread + e;
        smem_group[warpId][expert_idx] = reg_biased[e];
      }
      __syncwarp();

      // Lane 0 computes group top-2 sums and selects topk_group groups
      if (laneId == 0) {
        float group_sums[16];  // Max 16 groups
        for (int g = 0; g < num_expert_group; g++) {
          float top1 = -FLT_MAX, top2 = -FLT_MAX;
          for (int e_off = 0; e_off < experts_per_group; e_off++) {
            float val = smem_group[warpId][g * experts_per_group + e_off];
            if (val > top1) {
              top2 = top1;
              top1 = val;
            } else if (val > top2) {
              top2 = val;
            }
          }
          group_sums[g] = top1 + top2;
        }

        // Mark all groups enabled, then disable weakest groups
        for (int g = 0; g < num_expert_group; g++) {
          smem_group[warpId][g] = 1.0f;  // enabled
        }
        int groups_to_remove = num_expert_group - topk_group;
        for (int r = 0; r < groups_to_remove; r++) {
          float min_val = FLT_MAX;
          int min_group = 0;
          for (int g = 0; g < num_expert_group; g++) {
            if (smem_group[warpId][g] > 0 && group_sums[g] < min_val) {
              min_val = group_sums[g];
              min_group = g;
            }
          }
          smem_group[warpId][min_group] = 0.0f;  // disabled
        }
      }
      __syncwarp();

      // Each lane masks experts from non-selected groups
#pragma unroll
      for (int e = 0; e < kExpertsPerThread; e++) {
        int expert_idx = laneId * kExpertsPerThread + e;
        int my_group = expert_idx / experts_per_group;
        if (smem_group[warpId][my_group] == 0.0f) {
          reg_biased[e] = -FLT_MAX;
        }
      }
    }

    // Step 3: Iterative TopK via warp-level parallel argmax
    for (int k_idx = 0; k_idx < topk; k_idx++) {
      // Thread-local argmax over kExpertsPerThread values
      float my_val = -FLT_MAX;
      int my_expert = -1;
#pragma unroll
      for (int e = 0; e < kExpertsPerThread; e++) {
        if (reg_biased[e] > my_val) {
          my_val = reg_biased[e];
          my_expert = laneId * kExpertsPerThread + e;
        }
      }

      // Warp butterfly reduction for argmax (no __syncthreads!)
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float ov = __shfl_xor_sync(0xffffffff, my_val, offset);
        int oe = __shfl_xor_sync(0xffffffff, my_expert, offset);
        if (ov > my_val || (ov == my_val && oe < my_expert)) {
          my_val = ov;
          my_expert = oe;
        }
      }

      // Broadcast winner and retrieve original (unbiased) score
      int best = __shfl_sync(0xffffffff, my_expert, 0);
      int owning_lane = best / kExpertsPerThread;
      int local_idx = best % kExpertsPerThread;
      float best_score = __shfl_sync(0xffffffff,
          (laneId == owning_lane) ? reg_score[local_idx] : 0.0f, owning_lane);

      if (laneId == 0) {
        topk_ids[m * topk + k_idx] = best;
        topk_weights[m * topk + k_idx] = best_score;
      }

      // Invalidate selected expert in owning lane's register
      if (laneId == owning_lane) {
        reg_biased[local_idx] = -FLT_MAX;
      }
    }

    // Step 4: Renormalize weights (lane 0 only)
    if (renormalize && laneId == 0) {
      float sum = 0.0f;
      for (int k = 0; k < topk; k++) {
        sum += topk_weights[m * topk + k];
      }
      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          topk_weights[m * topk + k] /= sum;
        }
      }
    }
  }

  // Ensure all warps finished writing results before counter reset
  __syncthreads();

  // Reset counter for next invocation
  if (tid == 0) {
    *block_counter = 0;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Loop unroller for kNumTokens 1..16
template <typename T, int kNumTokens>
struct FusedGateDispatcher {
  static void dispatch(
      float* topk_weights,
      int* topk_ids,
      T const* mat_a,
      T const* mat_b,
      float const* correction_bias,
      float* scratch,
      int* block_counter,
      int topk,
      int num_expert_group,
      int topk_group,
      bool renormalize,
      cudaStream_t stream) {
    constexpr int VPT = 16 / sizeof(T);
    constexpr int kBlockSize = 128;

    cudaLaunchConfig_t config;
    config.gridDim = KIMI_LINEAR_NUM_EXPERTS;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(
        &config,
        kimi_linear_fused_gate_kernel<T, kBlockSize, VPT, kNumTokens>,
        topk_weights,
        topk_ids,
        mat_a,
        mat_b,
        correction_bias,
        scratch,
        block_counter,
        topk,
        num_expert_group,
        topk_group,
        renormalize);
  }
};

// Dispatch function pointer table for kNumTokens 1..16
template <typename T>
using DispatchFn = void (*)(
    float*, int*, T const*, T const*, float const*, float*, int*,
    int, int, int, bool, cudaStream_t);

template <typename T>
DispatchFn<T> getDispatchFn(int num_tokens) {
  static constexpr DispatchFn<T> table[16] = {
      FusedGateDispatcher<T, 1>::dispatch,
      FusedGateDispatcher<T, 2>::dispatch,
      FusedGateDispatcher<T, 3>::dispatch,
      FusedGateDispatcher<T, 4>::dispatch,
      FusedGateDispatcher<T, 5>::dispatch,
      FusedGateDispatcher<T, 6>::dispatch,
      FusedGateDispatcher<T, 7>::dispatch,
      FusedGateDispatcher<T, 8>::dispatch,
      FusedGateDispatcher<T, 9>::dispatch,
      FusedGateDispatcher<T, 10>::dispatch,
      FusedGateDispatcher<T, 11>::dispatch,
      FusedGateDispatcher<T, 12>::dispatch,
      FusedGateDispatcher<T, 13>::dispatch,
      FusedGateDispatcher<T, 14>::dispatch,
      FusedGateDispatcher<T, 15>::dispatch,
      FusedGateDispatcher<T, 16>::dispatch,
  };
  return table[num_tokens - 1];
}

// Persistent scratch buffers (allocated once per device)
static thread_local torch::Tensor scratch_buffer;
static thread_local torch::Tensor counter_buffer;

std::tuple<torch::Tensor, torch::Tensor> kimi_linear_fused_gate(
    torch::Tensor hidden_states,
    torch::Tensor router_weights,
    std::optional<torch::Tensor> correction_bias,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool renormalize) {
  int num_tokens = hidden_states.size(0);
  int hidden_dim = hidden_states.size(1);
  int num_experts = router_weights.size(0);

  TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
              "kimi_linear_fused_gate: num_tokens must be 1..16, got ", num_tokens);
  TORCH_CHECK(hidden_dim == KIMI_LINEAR_HIDDEN_DIM,
              "kimi_linear_fused_gate: hidden_dim must be ", KIMI_LINEAR_HIDDEN_DIM, ", got ", hidden_dim);
  TORCH_CHECK(num_experts == KIMI_LINEAR_NUM_EXPERTS,
              "kimi_linear_fused_gate: num_experts must be ", KIMI_LINEAR_NUM_EXPERTS, ", got ", num_experts);
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16,
              "kimi_linear_fused_gate: hidden_states must be bfloat16");
  TORCH_CHECK(router_weights.dtype() == torch::kBFloat16,
              "kimi_linear_fused_gate: router_weights must be bfloat16");

  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device());
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device());

  auto topk_weights_out = torch::empty({num_tokens, topk}, options_f32);
  auto topk_ids_out = torch::empty({num_tokens, topk}, options_i32);

  // Allocate scratch buffers if needed
  int scratch_size = 16 * KIMI_LINEAR_NUM_EXPERTS;  // Max tokens * experts
  if (!scratch_buffer.defined() || scratch_buffer.device() != hidden_states.device()) {
    scratch_buffer = torch::zeros({scratch_size}, options_f32);
    counter_buffer = torch::zeros({1}, options_i32);
  }

  float const* bias_ptr = correction_bias.has_value() ? correction_bias->data_ptr<float>() : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto fn = getDispatchFn<__nv_bfloat16>(num_tokens);
  fn(topk_weights_out.data_ptr<float>(),
     topk_ids_out.data_ptr<int>(),
     reinterpret_cast<__nv_bfloat16 const*>(hidden_states.data_ptr()),
     reinterpret_cast<__nv_bfloat16 const*>(router_weights.data_ptr()),
     bias_ptr,
     scratch_buffer.data_ptr<float>(),
     counter_buffer.data_ptr<int>(),
     static_cast<int>(topk),
     static_cast<int>(num_expert_group),
     static_cast<int>(topk_group),
     renormalize,
     stream);

  return std::make_tuple(topk_weights_out, topk_ids_out);
}
