# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

"""
SageAttention3 Backend for Visual Generation Models

Wraps the TensorRT-LLM SageAttention3 Blackwell NVFP4 entry point for
diffusion attention. Visual-gen modules pass tensors in NHD layout
([B, S, H, D]); the underlying SageAttention3 wrapper expects BHSD
([B, H, S, D]).
"""

import math
from typing import Optional, Tuple

import torch

from ...attention_backend.interface import PredefinedAttentionMask
from ...attention_backend.sage_attention3 import sage_attention3_blackwell
from .interface import AttentionBackend, AttentionTensorLayout


class SageAttention3Attention(AttentionBackend):
    """
    SageAttention3 Blackwell backend for diffusion models.

    Supports full/non-causal self-attention and cross-attention in NHD layout.
    The Blackwell kernel path currently requires matching Q/K/V head counts
    and fp16/bf16 input tensors.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)

        if self.num_heads != self.num_kv_heads:
            raise ValueError("SageAttention3 visual-gen backend does not support MQA/GQA yet")
        if self.head_dim not in (64, 128):
            raise ValueError(
                "SageAttention3 visual-gen backend currently supports head dimensions "
                "64 and 128"
            )

        self._preferred_layout = AttentionTensorLayout.NHD

    @staticmethod
    def _validate_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int) -> None:
        if q.dim() != 4:
            raise ValueError(f"SageAttention3 expects q shaped [B, S, H, D], got {q.shape}")
        if k.dim() != 4:
            raise ValueError(f"SageAttention3 expects k shaped [B, S, H, D], got {k.shape}")
        if v.dim() != 4:
            raise ValueError(f"SageAttention3 expects v shaped [B, S, H, D], got {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("SageAttention3 visual-gen backend requires matching batch sizes")
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError("SageAttention3 visual-gen backend does not support MQA/GQA yet")
        if q.shape[3] != head_dim or k.shape[3] != head_dim or v.shape[3] != head_dim:
            raise ValueError(
                f"SageAttention3 visual-gen backend expects head_dim={head_dim}, got "
                f"q={q.shape[3]}, k={k.shape[3]}, v={v.shape[3]}"
            )
        if k.shape[1] != v.shape[1]:
            raise ValueError(
                "SageAttention3 visual-gen backend requires matching K/V sequence lengths"
            )

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[PredefinedAttentionMask],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
        if attention_mask is None:
            attention_mask = PredefinedAttentionMask.FULL
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            raise NotImplementedError(
                "SageAttention3 causal mode is disabled in this integration because "
                "the upstream Blackwell kernel does not currently match SDPA numerics "
                "on SM120. Visual-gen diffusion attention should use FULL attention."
            )
        if attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError(
                f"SageAttention3 visual-gen backend only supports FULL attention, "
                f"got {attention_mask}."
            )

        self._validate_shapes(q, k, v, self.head_dim)

        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("SageAttention3 visual-gen backend requires matching Q/K/V dtypes")

        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return q, k, v, origin_dtype

    @torch.compiler.disable
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        output = sage_attention3_blackwell(
            q,
            k,
            v,
            is_causal=False,
            softmax_scale=self.scale,
        )
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using SageAttention3 Blackwell.

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_heads, head_dim]
            attention_mask: FULL is supported; CAUSAL is rejected.

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        q, k, v, origin_dtype = self._prepare_inputs(q, k, v, attention_mask)
        output = self._fwd(q, k, v)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
