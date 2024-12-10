import torch
import torch.nn as nn
import math
from typing import Tuple
import torch.nn.functional as F

class ExpertPreferredRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, bias: bool = False):
        super(ExpertPreferredRouter, self).__init__()
        self.num_experts = num_experts
        self.router_pred = nn.Linear(embedding_dim, num_experts, bias=bias)
        self._init_weights()

    def _init_weights(self):
        # uniform distribution for the router weights
        nn.init.uniform_(self.router_pred.weight, a=-2e-2, b=2e-2)
        if self.router_pred.bias is not None:
            nn.init.zeros_(self.router_pred.bias)

    @torch.amp.autocast(enabled=False, device_type="cuda")
    def _compute_router_probs(
            self, input_tokens: torch.Tensor, jitter_noise: float = 0.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the router probabilities for the input tokens.
        Keeps computation in float32 for numerical stability.
        """
        logits = self.router_pred(input_tokens)
        if jitter_noise > 0.0:
            noise = (
                torch.randn_like(logits, dtype=logits.dtype, device=logits.device)
                * jitter_noise
            )
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs

    def forward(self, x: torch.Tensor, c: torch.Tensor, jitter_noise: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        E = self.num_experts
        T = N

        # Compute router predictions
        r_probs = self._compute_router_probs(x, jitter_noise) # (B, N, E)
        r_modified = r_probs.permute(0, 2, 1).clone()  # (B, E, N)

        # Initialize assignments to the smallest model (expert = 0)
        M = torch.zeros((B, N), dtype=torch.long, device=x.device)

        # Iterate from largest model (E-1) down to smallest (0)
        for j in reversed(range(E)):
            kj = math.floor(c[j] * T)
            if kj > 0:
                # Get top-k indices for expert j
                _, indices = torch.topk(r_modified[:, j, :], k=kj, dim=-1)  # (B, kj)

                # Scatter assignments into M
                assign_vals = torch.full_like(indices, j, dtype=torch.long)
                M.scatter_(1, indices, assign_vals)

                # Scatter -inf into r_modified for these chosen tokens across all experts
                # indices shape: (B, kj)
                # Expand indices to (B, E, kj) to null out for all experts
                expanded_indices = indices.unsqueeze(1).expand(B, E, kj) # (B, E, kj)
                chosen_tokens = torch.full((B, E, kj), float('-inf'), device=x.device)
                r_modified.scatter_(2, expanded_indices, chosen_tokens)

        # Gather probabilities of the assigned experts
        M_probs = r_probs.gather(2, M.unsqueeze(-1)).squeeze(-1)  # (B, N)

        return M, M_probs


# Example usage
if __name__ == "__main__":
    B = 8   # batch size
    E = 4   # number of experts
    D = 128 # embedding dimension
    N = 10  # number of tokens per batch element

    x = torch.randn(B, N, D)
    c = torch.tensor([0.2, 0.21, 0.22, 0.37])

    epr = ExpertPreferredRouter(embedding_dim=D, num_experts=E)
    M, M_probs = epr(x, c)

    print("Assignments:", M)         # shape (B, N)
    print("Assigned Probs:", M_probs) # shape (B, N)

