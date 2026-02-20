"""FlashAttention-style helpers for ROCm GPUs."""

from typing import Optional

import torch
import torch.nn.functional as F


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """A FlashAttention-like utility that wraps :func:`torch.nn.functional.scaled_dot_product_attention`.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value tensors of shape ``(batch, seq_len, num_heads, head_dim)``.
    mask : torch.Tensor, optional
        Attention mask broadcastable to ``(batch, num_heads, seq_len, seq_len)``.
        Should contain ``0`` for tokens to keep and ``-inf`` for tokens to mask.
    bias : torch.Tensor, optional
        Additive attention bias broadcastable to ``(batch, num_heads, seq_len, seq_len)``.
    dropout_p : float, optional
        Dropout probability applied during attention, by default ``0.0``.
    causal : bool, optional
        Whether to apply causal masking, by default ``False``.
    scale : float, optional
        Scale factor to apply to the query tensor before computing attention.

    Returns
    -------
    torch.Tensor
        The attention output tensor.
    """
    # PyTorch expects (B, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_mask = mask
    if bias is not None:
        attn_mask = bias if attn_mask is None else attn_mask + bias

    # Pass scale to SDPA directly instead of pre-multiplying q.
    # Pre-multiplying q AND letting SDPA apply 1/sqrt(head_dim) causes
    # double scaling (attention logits divided by head_dim, not sqrt(head_dim)).
    sdpa_kwargs = dict(
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
    )
    if scale is not None:
        sdpa_kwargs["scale"] = scale

    out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

    return out.transpose(1, 2)


class FlashMHA(torch.nn.Module):
    """Multi-head attention using :func:`flash_attention`."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn = flash_attention(
            q,
            k,
            v,
            mask=attention_mask,
            dropout_p=self.dropout,
            causal=kwargs.get("causal", False),
            scale=self.scaling,
        )

        attn = attn.reshape(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn)
