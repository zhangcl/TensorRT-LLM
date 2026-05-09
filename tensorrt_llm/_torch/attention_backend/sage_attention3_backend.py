# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch

from .interface import (AttentionForwardArgs, AttentionMask, AttentionMetadata,
                        PredefinedAttentionMask, merge_attention_forward_args)
from .sage_attention3 import sage_attention3_blackwell
from .vanilla import VanillaAttention, VanillaAttentionMetadata


class SageAttention3Metadata(VanillaAttentionMetadata):
    pass


class SageAttention3Attention(VanillaAttention):
    """Dense no-cache attention backend backed by SageAttention3 Blackwell."""

    Metadata = SageAttention3Metadata

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_heads != self.num_kv_heads:
            raise ValueError("SageAttention3 does not support MQA/GQA yet")
        if self.head_dim not in (64, 128):
            raise ValueError(
                "SageAttention3 currently supports head dimensions 64 and 128"
            )

    def no_kv_cache_forward(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            v: Optional[torch.Tensor],
            num_heads: int,
            num_kv_heads: int,
            metadata: AttentionMetadata,
            *,
            attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
            position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if position_ids is not None:
            raise ValueError(
                "SageAttention3 backend expects RoPE to be applied before attention"
            )
        if attention_mask not in (PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL):
            raise ValueError(
                f"SageAttention3 does not support attention mask {attention_mask}"
            )
        if metadata.seq_lens is None:
            raise ValueError(
                "SageAttention3 no-cache attention requires seq_lens metadata"
            )

        if k is None or v is None:
            if k is not None or v is not None:
                raise ValueError("Both K and V must be None for fused QKV input")
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q, k, v = q.split([q_size, kv_size, kv_size], dim=-1)

        assert k is not None
        assert v is not None
        if num_heads != self.num_heads or num_kv_heads != self.num_kv_heads:
            raise ValueError(
                "SageAttention3 backend was called with inconsistent head counts"
            )

        seq_lens = metadata.seq_lens.to(device="cpu")
        if seq_lens.numel() == 0:
            return q.new_empty((0, self.num_heads * self.head_dim))
        seq_len = int(seq_lens[0].item())
        if not torch.all(seq_lens == seq_len).item():
            raise ValueError(
                "SageAttention3 no-cache backend currently requires uniform sequence lengths"
            )

        batch_size = int(seq_lens.numel())
        expected_tokens = batch_size * seq_len
        if (q.size(0) != expected_tokens or k.size(0) != expected_tokens
                or v.size(0) != expected_tokens):
            raise ValueError(
                "SageAttention3 no-cache backend requires dense Q/K/V matching seq_lens"
            )

        q = q.reshape(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, seq_len, self.num_kv_heads,
                      self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, seq_len, self.num_kv_heads,
                      self.head_dim).transpose(1, 2).contiguous()

        softmax_scale = None
        if self.q_scaling is not None:
            softmax_scale = 1.0 / (math.sqrt(self.head_dim) * self.q_scaling)

        out = sage_attention3_blackwell(
            q,
            k,
            v,
            is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
            softmax_scale=softmax_scale,
        )
        return out.transpose(1, 2).reshape(
            expected_tokens, self.num_heads * self.head_dim).contiguous()

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: SageAttention3Metadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if metadata.kv_cache_manager is not None:
            raise ValueError(
                "SageAttention3 backend is only wired for no-KV-cache dense attention"
            )
        return self.no_kv_cache_forward(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            metadata=metadata,
            attention_mask=forward_args.attention_mask,
        )
