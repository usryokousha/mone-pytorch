import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize

from typing import List

"""Implementation of the routing algorithm for the MONE model."""


class ExpertPreferredRouter(nn.Module):
    """
    Expert Preferred Router for the MONE model.

    This router assigns input tokens to experts based on a capacity distribution
    and learned routing probabilities. It supports jitter noise for improved training stability.

    Args:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): A list of floats representing the capacity
                                             distribution across experts. Must sum to 1.0.
        jitter_noise (float, optional): The amount of jitter noise to add to logits. Defaults to 0.0.
        dtype (torch.dtype, optional): The data type for the module. Defaults to torch.float32.

    Attributes:
        dim (int): The dimension of the input tokens.
        capacity_distribution (List[float]): The capacity distribution across experts.
        jitter_noise (float): The amount of jitter noise to add to logits.
        dtype (torch.dtype): The data type for the module.
    """

    def __init__(
        self,
        dim: int,
        capacity_distribution: List[float],
        jitter_noise: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim

        assert (
            sum(capacity_distribution) == 1.0
        ), "The sum of the capacity distribution must be 1.0"
        self.capacity_distribution = torch.tensor(capacity_distribution, dtype=dtype)
        self.jitter_noise = jitter_noise
        self.dtype = dtype
        self.num_experts = len(capacity_distribution)
        self.router = nn.Linear(dim, self.num_experts, dtype=dtype)
        self._init_weights()

    def _init_weights(self):
        # uniform distribution for the router weights
        nn.init.uniform_(self.router.weight, a=-2e-2, b=2e-2)
        nn.init.zeros_(self.router.bias)

    def _compute_router_probabilities(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute the router probabilities for the input tokens.
        """
        logits = self.router(input_tokens)

        if self.jitter_noise > 0.0:
            # add jitter noise to the logits
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute router probabilities and token assignments without looping over the batch,
        and without cloning the router probabilities.
        """
        batch, num_tokens, dim = input_tokens.shape
        # Compute router probabilities
        router_probs = self._compute_router_probabilities(input_tokens)

        # Compute token capacities
        k = (self.capacity_distribution * num_tokens).floor().long()

        # Initialize token assignments with -1 (unassigned)
        token_mask = torch.full(
            (batch, num_tokens), -1, dtype=torch.long, device=input_tokens.device
        )
        unassigned_mask = torch.ones(
            (batch, num_tokens), dtype=torch.bool, device=input_tokens.device
        )

        for j in range(self.num_experts - 1, -1, -1):
            # Compute the number of unassigned tokens per batch item
            num_unassigned = unassigned_mask.sum(dim=1)  # Shape: [batch]
            k_j = k[j].unsqueeze(0).expand(batch)  # Shape: [batch]
            k_i = torch.minimum(k_j, num_unassigned)  # Shape: [batch]

            if k_i.sum() == 0:
                continue

            # Extract router probabilities for expert j
            r_j = router_probs[:, :, j]  # Shape: [batch, num_tokens]

            # Mask already assigned tokens
            r_j = r_j.masked_fill(~unassigned_mask, float("-inf"))

            # Get sorted indices of router probabilities
            _, sorted_indices = torch.sort(
                r_j, dim=1, descending=True
            )  # Shape: [batch, num_tokens]

            # Create a range tensor for indices
            range_tensor = (
                torch.arange(num_tokens, device=input_tokens.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )

            # Create a mask for selecting top k_i tokens per batch item
            k_i_expanded = k_i.unsqueeze(1)  # Shape: [batch, 1]
            top_k_mask = range_tensor < k_i_expanded  # Shape: [batch, num_tokens], bool

            # Select the indices to assign
            selected_indices = sorted_indices[
                top_k_mask
            ]  # 1D tensor of selected token indices
            selected_batch_indices = (
                torch.arange(batch, device=input_tokens.device)
                .unsqueeze(1)
                .expand(-1, num_tokens)[top_k_mask]
            )

            # Assign to token_mask
            token_mask[selected_batch_indices, selected_indices] = j

            # Update unassigned_mask
            unassigned_mask[selected_batch_indices, selected_indices] = False

        # Assign any remaining unassigned tokens to expert 0
        token_mask[token_mask == -1] = 0

        return token_mask, torch.gather(
            router_probs, 2, token_mask.unsqueeze(-1)
        ).squeeze(-1)


class Combine(nn.Module):
    """
    Combine the output of the experts.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))
        self.dtype = dtype

    def forward(
        self,
        z: torch.Tensor,
        z_prime: torch.Tensor,
        router_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: Output of attention layer.
            z_prime: Output of feedforward layer.
            router_probs: Probabilities assigned by the router.
        """
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


# Define input tokens
batch_size = 2
num_tokens = 10
dim = 16
input_tokens = torch.randn(batch_size, num_tokens, dim)

# Define capacity distribution
capacity_distribution = [0.2, 0.3, 0.5]  # Sum must be 1.0

# Initialize router
router = ExpertPreferredRouter(dim=dim, capacity_distribution=capacity_distribution)

# Forward pass
token_mask, router_probs = router(input_tokens)

print("Token Mask:", token_mask)
print("Router Probabilities:", router_probs)

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
