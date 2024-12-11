# copy and pasted from dinov2: https://github.com/facebookresearch/dinov2

import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


# direct translation of FlexiViT code from:
# https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py
def resample_patch_embed(patch_embed: torch.Tensor, new_size: tuple) -> torch.Tensor:
    """Resample the weights of the patch embedding kernel to target resolution."""
    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    if tuple(patch_embed.shape[:2]) == tuple(new_size):
        return patch_embed

    logger.info(f"FlexiViT: resize embedding {patch_embed.shape} to {new_size}")

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
    dtype = patch_embed.dtype
    return v_resample_kernel(patch_embed).to(dtype)

# taken from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/patch_embed.py
class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

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
                new_proj.weight.copy_(
                    resample_patch_embed(self.proj.weight, new_patch_size)
                )
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(
                img_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
