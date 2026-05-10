# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# Importing thop loads the shared library that registers torch.ops.trtllm.
from tensorrt_llm.bindings.internal import thop as _thop  # noqa: F401


_SAGE3_BLOCK_SIZE = 128


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _pad_seq(x: torch.Tensor, multiple: int = _SAGE3_BLOCK_SIZE) -> torch.Tensor:
    pad_len = (_round_up(x.size(-2), multiple) - x.size(-2))
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def _check_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("SageAttention3 requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("SageAttention3 only supports fp16/bf16 QKV tensors")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("SageAttention3 requires Q, K, and V to have the same dtype")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise RuntimeError("SageAttention3 expects Q, K, V shaped [batch, heads, seq, head_dim]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise RuntimeError("SageAttention3 requires matching batch sizes")
    if q.size(1) != k.size(1) or q.size(1) != v.size(1):
        raise RuntimeError("SageAttention3 v3 integration does not support MQA/GQA yet")
    if q.size(-1) != k.size(-1) or q.size(-1) != v.size(-1):
        raise RuntimeError("SageAttention3 requires matching Q/K/V head dimensions")
    if q.size(-1) not in (64, 128):
        raise RuntimeError("SageAttention3 currently supports head dimensions 64 and 128")
    cc = torch.cuda.get_device_capability(q.device)
    if cc not in ((10, 0), (12, 0), (12, 1)):
        raise RuntimeError(
            f"SageAttention3 requires Blackwell SM100/SM120/SM121, got capability {cc}"
        )
    if not hasattr(torch.ops.trtllm, "sage_attention3_fwd"):
        raise RuntimeError(
            "SageAttention3 Torch ops are not built. Rebuild TensorRT-LLM with "
            "-DENABLE_SAGE_ATTENTION3=ON and CUDA 12.8+.")


def _preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    per_block_mean: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = k - k.mean(dim=-2, keepdim=True)
    q = _pad_seq(q)
    k = _pad_seq(k)
    v = _pad_seq(v)

    if per_block_mean:
        b, h, s, d = q.shape
        num_groups = s // _SAGE3_BLOCK_SIZE
        q_grouped = q.reshape(b, h, num_groups, _SAGE3_BLOCK_SIZE, d)
        qm = q_grouped.mean(dim=3)
        q = (q_grouped - qm.unsqueeze(3)).reshape_as(q).contiguous()
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = (q - qm).contiguous()

    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k.contiguous(), v.contiguous(), delta_s


def sage_attention3_blackwell(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    per_block_mean: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run SageAttention3 Blackwell NVFP4 attention.

    Q, K, and V must be fp16/bf16 tensors shaped [batch, heads, seq, head_dim].
    This wrapper performs the SageAttention3 preprocessing and FP4 packing, then
    dispatches to TensorRT-LLM's experimental SageAttention3 CUDA op.
    """

    if is_causal:
        raise NotImplementedError(
            "SageAttention3 causal mode is disabled in this integration because "
            "the upstream Blackwell kernel does not currently match SDPA numerics "
            "on SM120. Use non-causal/full attention for the NVFP4 AIGV path.")

    _check_supported(q, k, v)

    q_len = q.size(-2)
    k_len = k.size(-2)
    head_dim = q.size(-1)
    is_bf16 = q.dtype == torch.bfloat16

    q, k, v, delta_s = _preprocess_qkv(q, k, v, per_block_mean=per_block_mean)
    q_fp4, sf_q = torch.ops.trtllm.sage_attention3_scaled_fp4_quantize(q, 1)
    k_fp4, sf_k = torch.ops.trtllm.sage_attention3_scaled_fp4_permute_quantize(k, 1)
    v_fp4, sf_v = torch.ops.trtllm.sage_attention3_scaled_fp4_transpose_quantize(v, 1)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    out, _softmax_lse = torch.ops.trtllm.sage_attention3_fwd(
        q_fp4,
        k_fp4,
        v_fp4,
        sf_q,
        sf_k,
        sf_v,
        delta_s,
        k_len,
        None,
        softmax_scale,
        is_causal,
        per_block_mean,
        is_bf16,
    )
    return out[:, :, :q_len, :].contiguous()
