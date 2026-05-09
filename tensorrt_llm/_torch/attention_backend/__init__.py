from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, AttentionForwardArgs, AttentionMetadata
from .sage_attention3_backend import (SageAttention3Attention,
                                      SageAttention3Metadata)
from .sage_attention3 import sage_attention3_blackwell
from .sparse import get_sparse_attn_kv_cache_manager
from .trtllm import AttentionInputType, TrtllmAttention, TrtllmAttentionMetadata
from .vanilla import VanillaAttention, VanillaAttentionMetadata

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "AttentionForwardArgs",
    "AttentionInputType",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "SageAttention3Attention",
    "SageAttention3Metadata",
    "VanillaAttention",
    "VanillaAttentionMetadata",
    "get_sparse_attn_kv_cache_manager",
    "sage_attention3_blackwell",
]

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer import FlashInferAttention, FlashInferAttentionMetadata
    from .star_flashinfer import StarAttention, StarAttentionMetadata
    __all__ += [
        "FlashInferAttention", "FlashInferAttentionMetadata", "StarAttention",
        "StarAttentionMetadata"
    ]
