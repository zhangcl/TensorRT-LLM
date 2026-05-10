# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import (
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.sage_attention3 import (
    sage_attention3_blackwell,
)
from tensorrt_llm._torch.attention_backend.sage_attention3_backend import (
    SageAttention3Attention as LlmSageAttention3Attention,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.visual_gen.attention_backend.interface import (
    AttentionTensorLayout,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sage_attention3 import (
    SageAttention3Attention as VisualGenSageAttention3Attention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import (
    create_attention as create_visual_gen_attention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import (
    get_visual_gen_attention_backend,
)
from tensorrt_llm.bindings.internal import thop as _thop  # noqa: F401


def _cuda_cc():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return -1, -1


def _sage3_ops_available():
    return hasattr(torch.ops.trtllm, "sage_attention3_fwd")


def test_sage_attention3_backend_registry():
    assert get_attention_backend("SAGE3") is LlmSageAttention3Attention
    assert get_attention_backend("SAGE_ATTENTION3") is LlmSageAttention3Attention


def test_visual_gen_sage_attention3_backend_registry():
    assert get_visual_gen_attention_backend("SAGE3") is VisualGenSageAttention3Attention
    assert (
        get_visual_gen_attention_backend("SAGE_ATTENTION3")
        is VisualGenSageAttention3Attention
    )

    with pytest.raises(ValueError, match="Unknown visual-gen attention backend"):
        get_visual_gen_attention_backend("SAGE33")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(
    _cuda_cc() not in ((10, 0), (12, 0), (12, 1)),
    reason="SageAttention3 requires Blackwell.",
)
@pytest.mark.skipif(
    not _sage3_ops_available(),
    reason="SageAttention3 ops require a build with -DENABLE_SAGE_ATTENTION3=ON.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_sage_attention3_blackwell_matches_sdpa(dtype, head_dim, seq_len):
    torch.manual_seed(1234)
    q = torch.randn(1, 4, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = sage_attention3_blackwell(q, k, v, is_causal=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos > 0.95


def test_sage_attention3_causal_mode_rejected():
    q = torch.empty(1, 4, 128, 64)
    with pytest.raises(NotImplementedError):
        sage_attention3_blackwell(q, q, q, is_causal=True)


def test_visual_gen_sage_attention3_causal_mode_rejected():
    attn = create_visual_gen_attention(
        "SAGE3",
        layer_idx=0,
        num_heads=4,
        head_dim=64,
        num_kv_heads=4,
    )
    q = torch.empty(1, 128, 4, 64)

    with pytest.raises(NotImplementedError):
        attn.forward(q, q, q, attention_mask=PredefinedAttentionMask.CAUSAL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(
    _cuda_cc() not in ((10, 0), (12, 0), (12, 1)),
    reason="SageAttention3 requires Blackwell.",
)
@pytest.mark.skipif(
    not _sage3_ops_available(),
    reason="SageAttention3 ops require a build with -DENABLE_SAGE_ATTENTION3=ON.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_visual_gen_sage_attention3_matches_sdpa(dtype, head_dim, seq_len):
    torch.manual_seed(1234)
    attn = create_visual_gen_attention(
        "SAGE3",
        layer_idx=0,
        num_heads=4,
        head_dim=head_dim,
        num_kv_heads=4,
        dtype=dtype,
    )
    assert attn.preferred_layout == AttentionTensorLayout.NHD

    q = torch.randn(1, seq_len, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = attn.forward(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(
    _cuda_cc() not in ((10, 0), (12, 0), (12, 1)),
    reason="SageAttention3 requires Blackwell.",
)
@pytest.mark.skipif(
    not _sage3_ops_available(),
    reason="SageAttention3 ops require a build with -DENABLE_SAGE_ATTENTION3=ON.",
)
def test_visual_gen_sage_attention3_cross_attention_matches_sdpa():
    torch.manual_seed(1234)
    dtype = torch.bfloat16
    head_dim = 64
    num_heads = 4
    q_len = 128
    kv_len = 256
    attn = create_visual_gen_attention(
        "SAGE3",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,
        dtype=dtype,
    )

    q = torch.randn(1, q_len, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, kv_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    out = attn.forward(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos > 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(
    _cuda_cc() not in ((10, 0), (12, 0), (12, 1)),
    reason="SageAttention3 requires Blackwell.",
)
@pytest.mark.skipif(
    not _sage3_ops_available(),
    reason="SageAttention3 ops require a build with -DENABLE_SAGE_ATTENTION3=ON.",
)
def test_sage_attention3_backend_no_kv_cache():
    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 128
    num_heads = 4
    head_dim = 64
    dtype = torch.bfloat16
    q = torch.randn(batch_size * seq_len,
                    num_heads * head_dim,
                    device="cuda",
                    dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    metadata = LlmSageAttention3Attention.Metadata(
        max_num_requests=batch_size,
        max_num_tokens=batch_size * seq_len,
        kv_cache_manager=None,
        seq_lens=torch.tensor([seq_len] * batch_size, dtype=torch.int),
        num_contexts=batch_size,
    )
    attn = LlmSageAttention3Attention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,
    )

    out = attn.forward(q,
                       k,
                       v,
                       metadata,
                       attention_mask=PredefinedAttentionMask.FULL)
    assert out.shape == q.shape
