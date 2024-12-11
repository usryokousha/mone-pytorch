import torch
import torch.nn as nn
import torch.nn.functional as F
from mone_pytorch.layers.mlp import MLP
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Attention Pooling based on https://arxiv.org/pdf/1810.00825
    """

    def __init__(
        self,
        dim: int,
        latent_size: int = 1,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        proj_bias: bool = True,
        drop_rate: float = 0.0,
        mlp_ratio: float = 4.0,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
        mlp_layer: Optional[nn.Module] = None,
        post_norm: bool = False,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.latent = nn.Parameter(torch.zeros(1, latent_size, dim))
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        self.scale = self.head_dim**-0.5
        self.drop = nn.Dropout(drop_rate)
        if mlp_layer is not None:
            self.mlp = mlp_layer(dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
        else:
            self.mlp = None
        if post_norm:
            self.norm = norm_layer(dim) if norm_layer is not None else nn.Identity()
        else:
            self.norm = nn.Identity()
    
    def forward(self, x):
        B, N, _ = x.shape
        latent = self.latent.expand(B, -1, -1)
        q = self.q(latent).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        k, v = kv.unbind(dim=0)
        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, self.scale)
        x = x.transpose(1, 2).reshape(B, self.latent_size, self.dim)
        x = self.proj(x)
        x = self.drop(x)
        x = self.norm(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        if self.latent_size > 1:
            x = x.mean(dim=1)
        return x
