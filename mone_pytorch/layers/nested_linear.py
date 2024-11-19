import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable


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
    in_dim = w.shape[1]
    out_dim = w.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()
    
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
    in_dim = w.shape[1]
    out_dim = w.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()
    
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

@torch.compile
def nested_feedforward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: Callable = F.gelu,
    drop_rate: float = 0.0,
    num_experts: int = 4,
    training: bool = True,
) -> torch.Tensor:
    """More efficient implementation of nested feedforward"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()
    
    output = torch.zeros((batch_seq, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m_in = in_dim >> (num_experts - m - 1)
        D_m_out = out_dim >> (num_experts - m - 1)

        # get the m-th expert's input and sliced weight
        x_m = x[valid_mask]
        w_m = w1[:, :D_m_in]

        x_m = F.linear(x_m, w_m, b1)
        x_m = activation(x_m)
        x_m = F.dropout(x_m, drop_rate, training)
        w_m = w2[:D_m_out, :]
        y = F.linear(x_m, w_m, b2)
        output[valid_mask, :D_m_out] = F.dropout(y, drop_rate, training)

    return output.reshape(input_shape[:-1] + (out_dim,))

@torch.compile
def nested_feedforward_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    """More efficient implementation of nested feedforward with SwiGLU"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    output = torch.zeros((batch_seq, out_dim), device=x.device, dtype=x.dtype)
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m_in = in_dim >> (num_experts - m - 1)
        D_m_out = out_dim >> (num_experts - m - 1)

        x_m = x[valid_mask, :D_m_in]
        w_m = w1[:, :D_m_in]

        x_m = F.linear(x_m, w_m, b1)
        x1, x2 = x_m.chunk(2, dim=-1)
        x_m = x1 * F.silu(x2)

        w_m = w2[:D_m_out, :]
        y = F.linear(x_m, w_m, b2)
        output[valid_mask, :D_m_out] = y

    return output.reshape(input_shape[:-1] + (out_dim,))
