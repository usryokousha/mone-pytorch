from functools import lru_cache
import torch
from torch import IntTensor, BoolTensor
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_block_mask,
    BlockMask,
)
from typing import Tuple


def generate_natten_mod(
    canvas_w: int,
    canvas_h: int,
    kernel_w: int,
    kernel_h: int,
    n_cls: int = 1,
) -> _mask_mod_signature:
    """Generates a NATTEN attention mask with support for cls/register tokens.

    Args:
        canvas_w: The width of the canvas.
        canvas_h: The height of the canvas.
        kernel_w: The width of the kernel.
        kernel_h: The height of the kernel.
        n_cls: The number of cls/register tokens prepended to the sequence.
    """

    def get_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        idx_adj = idx - n_cls  # Adjust indices to account for prepended tokens
        x = idx_adj // canvas_w
        y = idx_adj % canvas_w
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
        kernel_center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        kernel_center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_h // 2
        natten_mask = hori_mask & vert_mask & both_image_tokens

        # Allow cls/register tokens to attend to all tokens
        cls_query_mask = q_is_cls
        cls_key_mask = kv_is_cls

        final_mask = natten_mask | cls_query_mask | cls_key_mask
        return final_mask

    return natten_mask_mod

@lru_cache
def create_natten_mask(
    batch_size: int,
    num_heads: int,
    canvas_w: int,
    canvas_h: int,
    kernel_w: int,
    kernel_h: int,
    n_cls: int = 1,
    compile: bool = False,
    device: str = "cuda",
) -> BlockMask:
    mask_mod = generate_natten_mod(canvas_w, canvas_h, kernel_w, kernel_h, n_cls)
    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=num_heads,
        Q_LEN=canvas_w,
        KV_LEN=canvas_h,
        device=device,
        _compile=compile,
    )
