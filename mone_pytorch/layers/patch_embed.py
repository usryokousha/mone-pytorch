# copy and pasted from dinov2: https://github.com/facebookresearch/dinov2

import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


# direct translation of FlexiViT code from:
# https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py
def resample_patch_embed(patch_embed: torch.Tensor, new_size: tuple) -> torch.Tensor:
    """Resample the weights of the patch embedding kernel to target resolution."""
    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    if tuple(patch_embed.shape[:2]) == tuple(new_size):
        return patch_embed

    logging.info(f"FlexiViT: resize embedding {patch_embed.shape} to {new_size}")

    def resize(x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        return (
            F.interpolate(
                x.unsqueeze(0).unsqueeze(-1),
                size=new_shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(-1)
        )

    def get_resize_mat(old_shape: tuple, new_shape: tuple) -> torch.Tensor:
        mat = []
        for i in range(old_shape[0] * old_shape[1]):
            basis_vec = torch.zeros(old_shape, device=patch_embed.device)
            basis_vec.view(-1)[i] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        return torch.stack(mat).T

    resize_mat = get_resize_mat(patch_embed.shape[:2], new_size)
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)

    def resample_kernel(kernel: torch.Tensor) -> torch.Tensor:
        resampled = resize_mat_pinv @ kernel.reshape(-1)
        return resampled.reshape(new_size)

    # Double vmap over both C_in and C_out dimensions
    v_resample_kernel = torch.vmap(torch.vmap(resample_kernel, in_dims=1), in_dims=1)
    return v_resample_kernel(patch_embed)

# modified from pytorch-image-models/timm/layers/patch_embed.py
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,  # if False, output will be in HWC format
            bias: bool = True,
            strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = make_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.flatten = flatten
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = make_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = make_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            else:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        else:
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        return x
