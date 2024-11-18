import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class NestedLinearExpand(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        num_groups=1,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        return nested_linear_expand(
            x, self.weight, expert_mask, self.bias, self.num_experts
        )


class NestedLinearContract(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        return nested_linear_contract(
            x, self.weight, self.bias, expert_mask, self.num_experts
        )


@torch.compile
def nested_linear_expand(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    input_shape = x.shape
    in_dim = x.shape[-1]
    batch_seq = x.shape[:-1].numel()
    out_dim = w.shape[0]
    output = torch.zeros((batch_seq, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m = in_dim >> (num_experts - m - 1)

        # slice the input and weight
        x_m = x[valid_mask, :D_m]
        w_m = w[:, :D_m]

        # project up to the expert dim
        output[valid_mask, :] = F.linear(x_m, w_m, b)

    return output.reshape(input_shape[:-1] + (out_dim,))


@torch.compile
def nested_linear_contract(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    input_shape = x.shape
    in_dim = x.shape[-1]
    batch_seq = x.shape[:-1].numel()
    out_dim = w.shape[0]
    output = torch.zeros((batch_seq, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m = out_dim >> (num_experts - m - 1)

        # get the m-th expert's input and sliced weight
        x_m = x[valid_mask]
        w_m = w[:D_m, :]

        # project down to the expert dim
        if b is not None:
            b_m = b[:D_m]
        else:
            b_m = None

        # Avoid explicit padding
        y = F.linear(x_m, w_m, b_m)
        output[valid_mask, :D_m] = y

    return output.reshape(input_shape[:-1] + (out_dim,))
