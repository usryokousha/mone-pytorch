import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize

from typing import List, Tuple

"""Implementation of the routing algorithm for the MONE model."""


def get_expert_probs(router_probs: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Get probabilities for assigned experts based on logits and token mask"""
    expert_probs = router_probs.gather(2, token_mask.unsqueeze(-1)).squeeze(-1)
    return expert_probs


class EPR(nn.Module):
    """
    Expert Preferred Router for the MONE model.

    This router assigns input tokens to experts based on a capacity distribution
    and learned routing probabilities. It supports jitter noise for improved training stability.

    Args:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): A list of floats representing the capacity
                                             distribution across experts. Must sum to 1.0.
        dtype (torch.dtype, optional): The data type for the module. Defaults to torch.float32.

    Attributes:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): The capacity distribution across experts.
        dtype (torch.dtype): The data type for the module.
    """

    def __init__(
        self,
        dim: int,
        capacity_distribution: List[float],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim

        assert (
            sum(capacity_distribution) == 1.0
        ), "The sum of the capacity distribution must be 1.0"
        self.capacity_distribution = capacity_distribution
        self.dtype = dtype
        self.num_experts = len(capacity_distribution)
        self.router = nn.Linear(dim, self.num_experts, dtype=dtype)
        self._init_weights()

    def _init_weights(self):
        # uniform distribution for the router weights
        nn.init.uniform_(self.router.weight, a=-2e-2, b=2e-2)
        nn.init.zeros_(self.router.bias)

    def _compute_router_probs(
        self, input_tokens: torch.Tensor, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the router probabilities for the input tokens.
        """
        logits = self.router(input_tokens)
        if jitter_noise > 0.0:
            # add jitter noise to the logits
            noise = torch.randn_like(logits, dtype=self.dtype) * jitter_noise
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs

    def _assign_tokens_to_experts(
        self, router_probs: torch.Tensor, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """
        Assign tokens to experts based on router probabilities and capacity constraints.

        Args:
            router_probs: Tensor of shape [batch_size, num_tokens, num_experts] containing router probabilities
            num_tokens: Number of tokens to assign
            device: Device to place tensors on

        Returns:
            token_mask: Tensor of shape [batch_size, num_tokens] containing expert assignments
        """
        batch_size = router_probs.shape[0]

        # Initialize token assignments with -1 (unassigned)
        token_mask = torch.full(
            (batch_size, num_tokens), -1, dtype=torch.long, device=device
        )
        unassigned_mask = torch.ones(
            (batch_size, num_tokens), dtype=torch.bool, device=device
        )

        # For each expert, assign tokens
        for j in reversed(range(self.num_experts)):
            # Check if any capacity is left for this expert
            capacity = int(math.floor(self.capacity_distribution[j] * num_tokens))
            if capacity == 0:
                continue

            # Get the router probabilities for expert j
            r_j = router_probs[:, :, j]  # [batch_size, num_tokens]

            # Mask already assigned tokens
            r_j = r_j.masked_fill(~unassigned_mask, float("-inf"))

            # Get the top-k tokens for this expert
            _, topk_indices = torch.topk(r_j, capacity, dim=1)

            # Assign tokens to the expert
            token_mask[:, topk_indices] = j

            # Update unassigned mask
            unassigned_mask[:, topk_indices] = False

        # Assign any remaining unassigned tokens to expert 0
        token_mask[token_mask == -1] = 0

        return token_mask

    def forward(
        self, input_tokens: torch.Tensor, prev_logits: torch.Tensor = None, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = input_tokens.shape
        device = input_tokens.device
        dtype = input_tokens.dtype

        # Get router probabilities
        router_probs = self._compute_router_probs(
            input_tokens.to(self.dtype), prev_logits, jitter_noise
        )

        # Assign tokens to experts
        token_mask = self._assign_tokens_to_experts(router_probs, num_tokens, device)

        expert_probs = get_expert_probs(router_probs, token_mask)

        return token_mask, expert_probs.to(dtype)


class NestedCombine(nn.Module):
    """
    Combine the output of the experts.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros((1,), dtype=dtype))
        self.dtype = dtype

    def forward(
        self,
        z: torch.Tensor,
        router_probs: torch.Tensor,
        z_prime: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z: Output of attention layer.
            z_prime: Output of feedforward layer.
            router_probs: Probabilities assigned by the router.
        """
        if z_prime is None:
            z_prime = 0.0
        return z + (self.alpha * router_probs + 1) * z_prime


def compute_capacity_distribution(e_c, E, delta=2, beta=10):
    """
    Computes the capacity distribution c_i for the MoNE framework.

    Args:
        e_c (float): Effective capacity, a value between 0 and 1.
        E (int): Number of nested experts.
        delta (float): Incentive parameter for larger models (>1). Default is 2.
        beta (float): Entropy regularization parameter (>0). Default is 10.

    Returns:
        c (np.ndarray): Capacity distribution array of shape (E,).
    """
    assert 0 < e_c < 1, "Effective capacity e_c must be between 0 and 1."
    assert E >= 1, "Number of experts E must be at least 1."
    assert delta > 1, "Delta must be greater than 1."
    assert beta > 0, "Beta must be greater than 0."

    # Initial guess for c_i
    c_init = np.full(E, 1.0 / E)

    # Model dimensions d_i (assuming d_i = D / 2^(E - i))
    d_ratios = np.array([2 ** (E - i - 1) for i in range(E)])
    d_ratios = d_ratios / d_ratios[0]  # Normalize so that d_1 = 1

    # Objective function to minimize (negative of the maximization problem)
    def objective(c):
        term1 = -np.sum(c * delta ** np.arange(E))
        term2 = beta * np.sum(
            c * np.log(c + 1e-12)
        )  # Add small epsilon to avoid log(0)
        return term1 + term2

    # Equality constraints
    constraints = [
        {"type": "eq", "fun": lambda c: np.sum(c) - 1},  # Sum of capacities equals 1
        {
            "type": "eq",
            "fun": lambda c: np.sum(c * d_ratios)
            - e_c,  # Effective capacity constraint
        },
    ]

    # Bounds for c_i (between 0 and 1)
    bounds = [(0, 1) for _ in range(E)]

    # Solve the optimization problem
    result = minimize(
        objective,
        c_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "disp": False, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    c = result.x
    c = np.clip(c, 0, 1)  # Ensure c_i are within [0, 1]

    # Normalize to ensure sum to 1 due to possible numerical errors
    c /= c.sum()

    return c


if __name__ == "__main__":
    import time

    def benchmark(router, input_tokens):
        start_time = time.time()
        token_mask, assigned_probs = router(input_tokens)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        end_time = time.time()
        return end_time - start_time

    # Define input tokens
    batch_size = 32
    num_tokens = 100
    dim = 128
    input_tokens = torch.randn(batch_size, num_tokens, dim).cuda()

    # Define capacity distribution
    capacity_distribution = [0.2, 0.3, 0.5]  # Sum must be 1.0

    # Initialize router
    router = ExpertPreferredRouter(
        dim=dim, capacity_distribution=capacity_distribution
    ).cuda()

    # Forward pass
    token_mask, router_probs = router(input_tokens)

    print("Token Mask:", token_mask.cpu())
    print("Router Probabilities:", router_probs.cpu())

    # benchmark
    print(f"Execution time: {benchmark(router, input_tokens):.6f} seconds")

    # Parameters
    e_c = 0.6  # Effective capacity (between 0 and 1)
    E = 3  # Number of experts
    delta = 2  # Incentive parameter (>1)
    beta = 10  # Entropy regularization parameter (>0)

    # Compute capacity distribution
    capacity_distribution = compute_capacity_distribution(e_c, E, delta, beta)

    print("Capacity Distribution c_i:")
    for i, c_i in enumerate(capacity_distribution):
        print(f"Expert {i+1}: c_{i+1} = {c_i:.4f}")

    # Verify constraints
    print("\nVerification:")
    print(f"Sum of c_i: {np.sum(capacity_distribution):.4f} (should be 1.0)")
    effective_capacity = np.sum(
        capacity_distribution * np.array([2 ** (E - i - 1) for i in range(E)])
    ) / (2 ** (E - 1))
    print(f"Effective Capacity e_c: {effective_capacity:.4f} (should be {e_c})")
