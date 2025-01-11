import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = (
            nn.Parameter(torch.empty(num_experts, out_features)) if bias else None
        )
        self.reset_parameters()

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, num_experts={self.num_experts}, bias={self.bias is not None}"

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(2)
        output = torch.einsum("btei,eoi->bteo", x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output


class ExpertLinearValve(nn.Module):
    """A trainable low-rank adaptation module that approximates SVD through learned routing.

    ExpertLinearValve implements a LoRA-like adapter that incorporates a trainable predictor to
    dynamically predict coefficients for the singular value decomposition (SVD) approximation.
    Instead of using fixed singular values, it learns to route and combine low-rank
    representations through a learned coefficient prediction mechanism.  This can also be combined
    with a MoE router to allow for better predictions of the coefficients.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        rank (int): Rank of the low-rank approximation
        k (int, optional): Number of top coefficients to select
        num_experts (int, optional): Number of experts
        bias (bool, optional): Whether to add a bias term to the output
        pred_bias (bool, optional): Whether to add a bias term to the predictor

    Attributes:
        u (nn.Parameter): Down-projection matrix ($U$ in SVD terms)
        v_star (nn.Parameter): Up-projection matrix ($V^T$ in SVD terms)
        predictor (nn.Linear): Trainable router that predicts combination coefficients

    Note:
        While traditional SVD decomposes a matrix into $U\Sigma V^T$ with fixed singular values,
        ExpertLinearValve learns to predict the coefficients dynamically, allowing for more
        flexible and context-dependent low-rank adaptations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        k: Optional[int] = None,
        num_experts: int = 1,
        bias: bool = True,
        pred_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.k = rank if k is None else k
        self.up = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.down = nn.Parameter(torch.empty(num_experts, out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, out_features))
        else:
            self.bias = None
        self.pred = nn.Linear(in_features, num_experts * rank, bias=pred_bias)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize the router weights
        nn.init.uniform_(self.up, a=-2e-2, b=2e-2)
        nn.init.uniform_(self.down, a=-2e-2, b=2e-2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.uniform_(self.pred.weight, a=-2e-2, b=2e-2)
        if self.pred.bias is not None:
            nn.init.zeros_(self.pred.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a router that predicts the coefficients
        coeff = F.softplus(self.pred(x.mean(dim=(0, 1))))
        coeff = coeff.view(self.num_experts, self.rank)
        coeff, indices = torch.topk(coeff, self.k, dim=-1)  # (num_experts, k)
        mask = F.one_hot(indices, num_classes=self.rank).float() # (num_experts, k, rank)
        mask = torch.einsum('ek,ekr->ekr', coeff, mask) # (num_experts, k, rank)
        
        # Use einsum to select weight components
        u_selected = torch.einsum('ekr,eri->eki', mask, self.up)
        v_selected = torch.einsum('ekr,eor->eko', mask, self.down)
        
        # Compute final weight matrix (num_experts, in_features, out_features)
        weight = torch.einsum("eok,eki->eoi", v_selected, u_selected) 
        output = torch.einsum('btei,eoi->bteo', x, weight)
        if self.bias is not None:
            output += self.bias
        return output
