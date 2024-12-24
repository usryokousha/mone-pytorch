import math
from functools import partial
from typing import Callable, Set

import torch
import torch.nn as nn
import torch.distributed as dist

from mone_pytorch.layers.patch_embed import PatchEmbed
from mone_pytorch.layers.block import (
    Block,
    NmoeBlock,
    NmoeParallelBlock,
    MoneBlock,
)
from mone_pytorch.layers.routing import (
    ExpertPreferredRouter,
    compute_capacity_distribution,
)


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm implementation"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=True,
        init_values=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        block_layer=Block,
        num_experts=4,
        capacity_factor=1.0,
    ):
        super().__init__()

        # Add type checking for block_layer and ensure it contain two different block types
        if isinstance(block_layer, list):
            if len(block_layer) != 2 or not all(
                issubclass(b, nn.Module) for b in block_layer
            ):
                raise ValueError(
                    "block_layer must contain two different block type if list"
                )

        if isinstance(block_layer, MoneBlock):
            self.router = ExpertPreferredRouter(
                dim=embed_dim, num_experts=num_experts, bias=True
            )
        else:
            self.router = None

        if self.router is not None:
            self.capacity_distribution = torch.tensor([0.0] * (num_experts - 1) + [1.0])
        else:
            self.capacity_distribution = None

        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()

        # Create interspersed blocks if block_layer is a list
        if isinstance(block_layer, list):
            self.block_interleave = True
            for i in range(depth):
                block_cls = block_layer[i % 2]  # Alternate between block types
                self.blocks.append(
                    block_cls(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        init_values=init_values,
                    )
                )
        else:
            # Original behavior for single block type
            self.block_interleave = False
            self.blocks = nn.ModuleList(
                [
                    block_layer(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        init_values=init_values,
                    )
                    for i in range(depth)
                ]
            )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.gradient_checkpointing = False
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}
    
    @torch.jit.ignore
    def set_gradient_checkpointing(self, enabled: bool):
        self.gradient_checkpointing = enabled

    def update_capacity(self, capacity_factor):
        if self.router is not None:
            if dist.is_initialized() and dist.get_rank() == 0:
                self.capacity_distribution = compute_capacity_distribution(
                    self.num_experts, capacity_factor
                )
            dist.broadcast(self.capacity_distribution, 0)
        else:
            for i, blk in enumerate(self.blocks):
                if isinstance(blk, (NmoeBlock, NmoeParallelBlock)):
                    blk.capacity_factor = capacity_factor
                else:
                    blk.capacity_factor = capacity_factor

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def interpolate_pos_encoding(self, x, w, h, offset=0.1):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        w0, h0 = w0 + offset, h0 + offset
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
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        if self.router is not None:
            route_mask, route_prob = self.router(x, self.capacity_distribution)
            for blk in self.blocks:
                if self.training and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, route_mask, route_prob)
                else:
                    x = blk(x, route_mask, route_prob)
        else:
            for blk in self.blocks:
                if self.training and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # Get routing information if using MoNE
        if self.router is not None:
            route_mask, route_prob = self.router(x, self.capacity_distribution)
        
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            # Apply routing for MoNE blocks
            if self.router is not None:
                x = blk(x, route_mask, route_prob)
            else:
                x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def mone_vit_tiny(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_layer=MoneBlock,
        num_experts=num_experts,
        **kwargs,
    )
    return model

def mone_vit_small(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_layer=MoneBlock,
        num_experts=num_experts,
        **kwargs,
    )
    return model

def mone_vit_base(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_layer=MoneBlock,
        num_experts=num_experts,
        **kwargs,
    )
    return model


def nmoe_vit_tiny(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_layer=[Block, NmoeBlock],
        num_experts=num_experts,
        **kwargs,
    )
    return model


def nmoe_vit_small(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_layer=[Block, NmoeBlock],
        num_experts=num_experts,
        **kwargs,
    )
    return model


def nmoe_vit_base(patch_size=16, num_experts=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_layer=[Block, NmoeBlock],
        num_experts=num_experts,
        **kwargs,
    )
    return model
