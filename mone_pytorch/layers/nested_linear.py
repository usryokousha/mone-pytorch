import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

class NestedLinearExpand(nn.Linear):
    """Performs a linear projection expansion with a nested expert mask"""
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
        if torch.all(expert_mask == self.num_experts - 1):
            return F.linear(x, self.weight, self.bias)
        else:
            return nested_linear_expand(
                x, self.weight, expert_mask, self.bias, self.num_experts
            )

class NestedLinearContract(nn.Linear):
    """Performs a linear projection contraction with a nested expert mask"""
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
        if torch.all(expert_mask == self.num_experts - 1):
            return F.linear(x, self.weight, self.bias)
        else:
            return nested_linear_contract(
                x, self.weight, expert_mask, self.bias, self.num_experts
            )

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

    # Initialize output tensor with same dtype as input
    output = torch.zeros(
        (batch_seq, out_dim), device=x.device, dtype=x.dtype
    )
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

    # Initialize output tensor with same dtype as input
    output = torch.zeros(
        (batch_seq, out_dim), device=x.device, dtype=x.dtype
    )
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


def nested_mlp(
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
    """More efficient implementation of nested MLP"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    # Initialize output tensor with same dtype as input
    output = torch.zeros(
        (batch_seq, out_dim), device=x.device, dtype=x.dtype
    )
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        valid_mask = (expert_mask == m).view(batch_seq)
        D_m_in = in_dim >> (num_experts - m - 1)
        D_m_out = out_dim >> (num_experts - m - 1)

        x_m = x[valid_mask, :D_m_in]
        w_m = w1[:, :D_m_in]
        x_m = F.linear(x_m, w_m, b1)
        x_m = activation(x_m)
        x_m = F.dropout(x_m, drop_rate, training)
        w_m = w2[:D_m_out, :]
        if b2 is not None:
            b_m = b2[:D_m_out]
        else:
            b_m = None
        y_m = F.linear(x_m, w_m, b_m)
        output[valid_mask, :D_m_out] = F.dropout(y_m, drop_rate, training)

    return output.reshape(input_shape[:-1] + (out_dim,))

def nested_swiglu_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    """More efficient implementation of nested MLP with SwiGLU"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    # Initialize output tensor with same dtype as input
    output = torch.zeros(
        (batch_seq, out_dim), device=x.device, dtype=x.dtype
    )
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
        if b2 is not None:
            b_m = b2[:D_m_out]
        else:
            b_m = None
        y = F.linear(x_m, w_m, b_m)
        output[valid_mask, :D_m_out] = y

    return output.reshape(input_shape[:-1] + (out_dim,))


if __name__ == "__main__":
    import os

    os.environ["TORCH_CUDNN_V9_API_ENABLED"] = "1"
    x = torch.randn(256, 768, 768, dtype=torch.float32).cuda()
    w = torch.randn(768, 768, dtype=torch.float32).cuda()
    expert_mask = torch.randint(0, 4, (256, 768)).cuda()
    # let's compare latency
    import time

    dense_times = []
    for _ in range(20):
        start_time = time.time()
        F.linear(x, w)
        torch.cuda.synchronize()
        end_time = time.time()
        dense_times.append(end_time - start_time)
    print(f"Dense average time: {sum(dense_times)/20:.6f} seconds")

    # Average over 10 executions for dense
    dense_times = []
    for _ in range(20):
        start_time = time.time()
        nested_linear_expand(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        dense_times.append(end_time - start_time)
    print(f"Dense expand average time: {sum(dense_times)/20:.6f} seconds")

    dense_times = []
    for _ in range(20):
        start_time = time.time()
        nested_linear_contract(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        dense_times.append(end_time - start_time)
    print(f"Dense contract average time: {sum(dense_times)/20:.6f} seconds")

    nested_mlp_times = []
    for _ in range(20):
        start_time = time.time()
        nested_mlp(x, w, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        nested_mlp_times.append(end_time - start_time)
    print(f"Nested MLP average time: {sum(nested_mlp_times)/20:.6f} seconds")

    # compare nested mlp vs expand + contract mlp
    combination_times = []
    for _ in range(20):
        start_time = time.time()
        out = nested_linear_expand(x, w, expert_mask)
        out = F.gelu(out)
        out = F.dropout(out, 0.0, training=True)
        nested_linear_contract(out, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        combination_times.append(end_time - start_time)
    print(f"Combination average time: {sum(combination_times)/20:.6f} seconds")
