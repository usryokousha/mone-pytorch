"""
Adapted from from timm library.
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
"""

import math

import torch
import torch.nn as nn

from mone_pytorch.layers.patch_embed import PatchEmbed
from mone_pytorch.layers.block import NestedSequentialBlock, NestedParallelBlock
from mone_pytorch.layers.mlp import NestedMLP, NestedSwiGLUMLP
from mone_pytorch.layers.routing import EPR

from typing import Union, Tuple

mlp_layer = {
    "mlp": NestedMLP,
    "swiglu": NestedSwiGLUMLP,
}

block_layer = {
    "sequential": NestedSequentialBlock,
    "parallel": NestedParallelBlock,
}


def nested_vit(block_type: str = "sequential", mlp_type: str = "mlp", **kwargs):
    return NestedVisionTransformer(
        block_layer=block_layer[block_type], mlp_layer=mlp_layer[mlp_type], **kwargs
    )


class NestedVisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_experts=4,
        capacity_dist=[1.0, 0.0, 0.0, 0.0],
        num_routers=1,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_alpha=False,
        norm_layer=nn.LayerNorm,
        mlp_layer=NestedMLP,
        block_layer=NestedSequentialBlock,
        shared_router=False,
        router_layer=EPR,
        mask_window_size=None,
    ):
        super().__init__()
        assert (
            len(capacity_dist) == num_experts
        ), "Capacity distribution must be of length num_experts"
        self.num_features = self.embed_dim = embed_dim
        self.num_routers = num_routers
        self.shared_router = shared_router
        self.mask_window_size = mask_window_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.alpha = None
        if use_alpha:
            self.alpha = nn.Parameter(torch.zeros((1,)))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([])
        self.routers = nn.ModuleList([])
        # possible improvement on original implementation
        # router weights are shared across blocks
        self.blocks_per_router = depth // num_routers
        for i in range(depth):
            if not self.shared_router:
                add_router = i % self.blocks_per_router == 0
            else:
                add_router = i == 0
            if add_router:
                self.routers.append(router_layer(embed_dim, capacity_dist))
            self.blocks.append(
                block_layer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    qk_scale=qk_scale,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    num_experts=num_experts,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                )
            )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_patch_size(self, new_size: Union[int, Tuple[int, int]]) -> None:
        self.patch_embed.update_patch_size(new_size)
        for block in self.blocks:
            block.update_block_mask(new_size, self.mask_window_size, 0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed.to(x.dtype)
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        capacity_dist: Tuple[float] = None,
        jitter_noise: float = 0.0,
    ) -> torch.Tensor:
        x = self.prepare_tokens(x)
        expert_mask = None
        router_probs = None
        for i, blk in enumerate(self.blocks):
            if i % self.blocks_per_router == 0:
                if not self.shared_router:
                    idx_router = i // self.blocks_per_router
                else:
                    idx_router = 0
                expert_mask, router_probs = self.routers[idx_router](
                    x, capacity_dist=capacity_dist, jitter_noise=jitter_noise
                )
            x = blk(x, expert_mask, router_probs, self.alpha)
        x = self.norm(x)
        # global average pooling
        return self.head(x.mean(dim=1))
