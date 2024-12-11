# adapted from dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from torch import IntTensor, BoolTensor
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    flex_attention,
    create_block_mask,
    BlockMask,
)
from typing import Optional, Tuple, Callable, Union
from .nested_linear import (
    nested_linear_expand,
    nested_linear_contract,
    _check_nested_linear,
)

attention_fn = {
    "flash": F.scaled_dot_product_attention,
    "flex": flex_attention,
}


# NATTEN attention mask with support for cls/register tokens
def generate_natten_ext(
    grid_w: int,
    grid_h: int,
    window_w: int,
    window_h: int,
    n_cls: int = 1,
) -> _mask_mod_signature:
    """Generates a NATTEN attention mask with support for cls/register tokens.

    Args:
        grid_w: The width of the grid.
        grid_h: The height of the grid.
        window_w: The width of the window.
        window_h: The height of the window.
        n_cls: The number of cls/register tokens prepended to the sequence.
    """

    def get_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        idx_adj = idx - n_cls  # Adjust indices to account for prepended tokens
        x = idx_adj // grid_w
        y = idx_adj % grid_w
        x = torch.where(idx_adj >= 0, x, -1)  # Assign -1 to special tokens
        y = torch.where(idx_adj >= 0, y, -1)
        return x, y

    def natten_mask_mod(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_x, q_y = get_x_y(q_idx)
        kv_x, kv_y = get_x_y(kv_idx)

        q_is_cls = (q_x == -1) & (q_y == -1)
        kv_is_cls = (kv_x == -1) & (kv_y == -1)

        # Native attention mask between image tokens
        both_image_tokens = (~q_is_cls) & (~kv_is_cls)
        kernel_center_x = q_x.clamp(window_w // 2, (grid_w - 1) - window_w // 2)
        kernel_center_y = q_y.clamp(window_h // 2, (grid_h - 1) - window_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= window_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= window_h // 2
        natten_mask = hori_mask & vert_mask & both_image_tokens

        # Allow cls/register tokens to attend to all tokens
        cls_query_mask = q_is_cls
        cls_key_mask = kv_is_cls

        final_mask = natten_mask | cls_query_mask | cls_key_mask
        return final_mask

    return natten_mask_mod


@lru_cache
def update_natten_mask(
    grid_size: Tuple[int, int], window_size: Tuple[int, int], n_cls: int
) -> torch.Tensor:
    return create_block_mask(
        generate_natten_ext(
            grid_size[0], grid_size[1], window_size[0], window_size[1], n_cls
        )
    )


@lru_cache
def update_local_window_mask(window_size: int, n_cls: int) -> torch.Tensor:
    return create_block_mask(generate_local_window_mask(window_size, n_cls))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
        attn_drop: float,
        proj_drop: float,
        qk_scale: Optional[float] = None,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_fn: Callable = F.scaled_dot_product_attention,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.scale = head_dim**-0.5 if qk_scale is None else qk_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.attn_fn = attn_fn

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[Union[torch.Tensor, BlockMask]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        num_experts: int = 1,
    ) -> torch.Tensor:
        B, N, C = x.shape
        if _check_nested_linear(expert_mask, num_experts):
            qkv = nested_linear_expand(
                x, self.qkv.weight, expert_mask, self.qkv.bias, num_experts
            )
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if isinstance(self.attn_fn, F.scaled_dot_product_attention):
            assert isinstance(
                attn_mask, torch.Tensor
            ), "mask must be a tensor for scaled_dot_product_attention"
            x = self.attn_fn(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop, scale=self.scale
            )
        elif isinstance(self.attn_fn, flex_attention):
            assert isinstance(
                attn_mask, BlockMask
            ), "mask must be a BlockMask for flex_attention"
            x = flex_attention(q, k, v, block_mask=attn_mask, scale=self.scale)
            x = F.dropout(x, p=self.attn_drop, training=self.training)
        else:
            raise ValueError(f"Unsupported attention function: {self.attn_fn}")
        if _check_nested_linear(expert_mask, num_experts):
            x = nested_linear_contract(
                x, self.proj.weight, expert_mask, self.proj.bias, num_experts
            )
        else:
            x = self.proj(x)
        x = F.dropout(x, p=self.proj_drop, training=self.training)
        return x


def generate_local_window_mask(
    tokens_per_window: int,
    n_cls: int = 1,
) -> _mask_mod_signature:
    """Generates an attention mask for local window attention with fixed window sizes.

    Args:
        tokens_per_window: Number of tokens in each window after dropout
        n_cls: Number of cls/register tokens prepended to the sequence
    """

    def local_mask_mod(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        # Adjust indices to account for cls tokens
        q_idx_adj = q_idx - n_cls
        kv_idx_adj = kv_idx - n_cls

        # Identify cls tokens (indices < 0 after adjustment)
        q_is_cls = q_idx_adj < 0
        kv_is_cls = kv_idx_adj < 0

        # Calculate window indices for regular tokens
        q_window = torch.where(q_is_cls, -1, q_idx_adj // tokens_per_window)
        kv_window = torch.where(kv_is_cls, -1, kv_idx_adj // tokens_per_window)

        # Local window mask: tokens can only attend within their window
        window_mask = (q_window == kv_window) & (~q_is_cls) & (~kv_is_cls)

        # Allow cls tokens to attend to all tokens and be attended by all tokens
        cls_query_mask = q_is_cls
        cls_key_mask = kv_is_cls

        final_mask = window_mask | cls_query_mask | cls_key_mask
        return final_mask

    return local_mask_mod
