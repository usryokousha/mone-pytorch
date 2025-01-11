import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from routing import ExpertsChooseMaskedRouter
from typing import Optional


class ExpertsChooseMaskedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bt...,eio->bteo", x, self.weight) + self.bias


def dispatch(
    x: torch.Tensor, 
    dispatch_mask: torch.Tensor, 
    residual: bool = False
) -> torch.Tensor:
    B, T, D = x.shape
    E, C = dispatch_mask.size(2), dispatch_mask.size(3)
    if residual:
        x = x.repeat(1, 1, 2)
        expert_dim = 2 * D // E
    else:
        expert_dim = D // E
    x = x.view(B, T, E, expert_dim)
    x = torch.einsum("btei,btec->bcei", x, dispatch_mask)
    x = x.view(B, C, E, expert_dim)
    if residual:
        return torch.chunk(x, 2, dim=2)
    return x


def combine(
    x: torch.Tensor,
    combine_array: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, C, E, D = x.shape
    T = combine_array.size(1)
    output_dim = D * E
    if residual is not None:
        x = torch.cat([x, residual], dim=2)
    x = torch.einsum("bceo,btec->bteo", x, combine_array)
    if residual is not None:
        return torch.sum(x.view(B, T, 2, output_dim), dim=2)
    return x.view(B, T, output_dim)


if __name__ == "__main__":
    batch_size = 16
    num_experts = 4
    embed_dim = 768
    num_tokens = 196
    inputs = torch.randn(batch_size, num_tokens, embed_dim).cuda()

    # Create the router
    router = ExpertsChooseMaskedRouter(dim=embed_dim, num_experts=num_experts).cuda()
    
    # Create the linear layer
    linear = ExpertsChooseMaskedLinear(
        in_features=embed_dim // num_experts,
        out_features=embed_dim // num_experts,
        num_experts=num_experts,
    ).cuda()

    # Dispatch the experts
    dispatch_mask, combine_array = router(inputs, c=num_tokens // 2)

    # Dispatch the inputs to the experts
    x = dispatch(inputs, dispatch_mask)

    # Apply the linear layer
    x = linear(x)

    # Combine the experts
    x = combine(x, combine_array)

    print(x)

    # Now we need to do the residual connection
    num_experts = 8
    inputs = torch.randn(batch_size, num_tokens, embed_dim).cuda()

    # Create the router
    router = ExpertsChooseMaskedRouter(dim=embed_dim, num_experts=num_experts).cuda()
    
    # Create the linear layer with half the experts
    linear = ExpertsChooseMaskedLinear(
        in_features=embed_dim // (num_experts // 2),
        out_features=embed_dim // (num_experts // 2),
        num_experts=num_experts // 2,
    ).cuda()

    # Dispatch the experts
    dispatch_mask, combine_array = router(inputs, c=num_tokens // 2)

    # Dispatch the inputs to the experts
    x, res = dispatch(inputs, dispatch_mask, residual=True)
    x = linear(x)
    x = combine(x, combine_array, residual=res)
    print(x)
