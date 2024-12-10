import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize

from typing import Tuple

"""Implementation of the routing algorithm for the MONE model."""


class EPR(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, bias: bool = False):
        super(EPR, self).__init__()
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

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        E = self.num_experts
        T = N

        # Compute router predictions
        r_probs = self._compute_router_probs(x, jitter_noise)  # (B, N, E)
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
                expanded_indices = indices.unsqueeze(1).expand(B, E, kj)  # (B, E, kj)
                chosen_tokens = torch.full((B, E, kj), float("-inf"), device=x.device)
                r_modified.scatter_(2, expanded_indices, chosen_tokens)

        # Gather probabilities of the assigned experts
        M_probs = r_probs.gather(2, M.unsqueeze(-1)).squeeze(-1)  # (B, N)

        return M, M_probs


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


def compute_capacity_distribution(E, e_c, delta=2.0, beta=10.0):
    """
    Solve the capacity distribution optimization problem:

    maximize    sum_{i=1}^E c_i * delta^(i-1) - beta * sum_{i=1}^E c_i * log(c_i)
    subject to: sum_{i=1}^E c_i = 1
                sum_{i=1}^E c_i / 2^(E-i) = e_c
                0 <= c_i <= 1

    Parameters
    ----------
    E : int
        Number of experts.
    e_c : float
        Effective capacity (0 < e_c < 1).
    delta : float, optional
        Weighting factor that incentivizes usage of larger experts. Default is 2.0.
    beta : float, optional
        Entropy coefficient that promotes uniformity. Default is 10.0.

    Returns
    -------
    c : ndarray
        Optimal capacity distribution array of shape (E,).
    """

    # Initial guess: uniform distribution
    c0 = np.ones(E) / E

    def objective(c):
        # sum_{i=1}^E c_i * delta^(i-1)
        term1 = sum(c[i] * (delta**i) for i in range(E))

        # Avoid log(0) by adding a tiny epsilon
        eps = 1e-15
        term2 = sum(c[i] * np.log(c[i] + eps) for i in range(E))

        # Objective to maximize: term1 - beta * term2
        # We'll minimize the negative: -(term1 - beta * term2) = -term1 + beta * term2
        return -(term1 - beta * term2)

    # Constraints:
    # sum(c_i) = 1
    def constraint_sum(c):
        return np.sum(c) - 1.0

    # sum(c_i / 2^(E-i)) = e_c
    # Note: For i in Python 0-based: c[i] = c_(i+1), so exponent is (E-(i+1)) = E-i-1
    def constraint_ec(c):
        return np.sum(c[i] / (2.0 ** (E - i - 1)) for i in range(E)) - e_c

    constraints = [
        {"type": "eq", "fun": constraint_sum},
        {"type": "eq", "fun": constraint_ec},
    ]

    # Bounds for each c_i: [0, 1]
    bounds = [(0.0, 1.0) for _ in range(E)]

    # Solve the optimization problem
    result = minimize(
        objective, c0, method="SLSQP", constraints=constraints, bounds=bounds
    )

    if not result.success:
        raise ValueError("Optimization did not converge: " + result.message)

    return result.x


def _exponential_annealing(
    step: int,
    start_step: int,
    max_steps: int,
    min_val: float = 0.1,
    max_val: float = 1.0,
) -> float:
    """Anneal the value exponentially from max_val to min_val over max_steps."""
    # Calculate decay rate to reach min_temp at max_steps with a delay of start_step
    decay_rate = -math.log(min_val) / (max_steps - start_step)
    return max(min_val, max_val * math.exp(-decay_rate * (step - start_step)))


def _cosine_annealing(
    step: int,
    start_step: int,
    max_steps: int,
    min_val: float = 0.1,
    max_val: float = 1.0,
) -> float:
    """Anneal the value cosinely from max_val to min_val over max_steps with a delay of start_step."""
    if step < start_step:
        return max_val

    # Adjust step to account for start_step offset
    adjusted_step = step - start_step
    total_steps = max_steps - start_step

    return min_val + 0.5 * (max_val - min_val) * (
        1 + math.cos(math.pi * adjusted_step / total_steps)
    )


def _linear_annealing(
    step: int,
    start_step: int,
    max_steps: int,
    min_val: float = 0.1,
    max_val: float = 1.0,
) -> float:
    """Anneal the value linearly from max_val to min_val over max_steps."""
    if step < start_step:
        return max_val

    # Adjust step to account for start_step offset
    adjusted_step = step - start_step
    total_steps = max_steps - start_step

    # Linear decay from max_val to min_val
    return max_val - (max_val - min_val) * (adjusted_step / total_steps)


class CapacityScheduler:
    def __init__(
        self,
        patch_size: int,
        image_size: int,
        max_epochs: int,
        min_capacity: float = 0.1,
        max_capacity: float = 1.0,
        num_experts: int = 4,
        delta: float = 2,
        beta: float = 10,
        annealing_type: str = "cosine",
    ):
        self.patch_size = patch_size
        self.image_size = image_size
        self.max_epochs = max_epochs
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.num_experts = num_experts
        self.delta = delta
        self.beta = beta
        self.annealing_type = annealing_type
        self.start_epoch = 0
        self.epoch = 0

    def update(self) -> float:
        # Update the epoch
        self.epoch += 1

        # Compute the effective capacity for the current step
        if self.annealing_type == "exponential":
            self.effective_capacity = _exponential_annealing(
                self.epoch,
                self.start_epoch,
                self.max_epochs,
                self.min_capacity,
                self.max_capacity,
            )
        elif self.annealing_type == "cosine":
            self.effective_capacity = _cosine_annealing(
                self.epoch,
                self.start_epoch,
                self.max_epochs,
                self.min_capacity,
                self.max_capacity,
            )
        elif self.annealing_type == "linear":
            self.effective_capacity = _linear_annealing(
                self.epoch,
                self.start_epoch,
                self.max_epochs,
                self.min_capacity,
                self.max_capacity,
            )

        # Compute the capacity distribution based on new effective capacity
        self.capacity_distribution = compute_capacity_distribution(
            self.num_experts, self.effective_capacity, self.delta, self.beta
        )

        return self.capacity_distribution, self.adjusted_patch_size

    @property
    def adjusted_patch_size(self) -> int:
        # Calculate patch size based on effective capacity and ensure it's even
        patch_size = max(1, int(self.patch_size * self.effective_capacity))
        return patch_size if patch_size % 2 == 0 else patch_size + 1


if __name__ == "__main__":
    # Parameters
    e_c = 0.6  # Effective capacity (between 0 and 1)
    E = 4  # Number of experts
    delta = 2  # Incentive parameter (>1)
    beta = 10  # Entropy regularization parameter (>0)

    # Compute capacity distribution
    capacity_distribution = compute_capacity_distribution(E, e_c, delta, beta)
    print("Optimal capacity distribution c:", capacity_distribution)
    print("Sum of capacities:", f"{np.sum(capacity_distribution):.4f}")
    # Check the effective capacity constraint
    check_ec = np.sum([capacity_distribution[i] / 2.0 ** (E - i - 1) for i in range(E)])
    print("Computed e_c:", f"{check_ec:.4f}")

    import time

    def benchmark(router, input_tokens, c):
        start_time = time.time()
        router(input_tokens, c=c)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        end_time = time.time()
        return end_time - start_time

    # Define input tokens
    batch_size = 16
    num_tokens = 10
    dim = 128
    input_tokens = torch.randn(batch_size, num_tokens, dim).cuda()

    # Capacity distribution from above

    # Initialize router
    router = EPR(embedding_dim=dim, num_experts=E).cuda()

    # Forward pass
    token_mask, router_probs = router(
        input_tokens, c=torch.tensor(capacity_distribution).to(input_tokens.device)
    )

    print("Token Mask:", token_mask.cpu())
    print("Router Probabilities:", router_probs.cpu())

    # benchmark
    print(
        f"Execution time EPR: {benchmark(router, input_tokens, c=torch.tensor(capacity_distribution).to(input_tokens.device)):.6f} seconds"
    )

    capacity_scheduler = CapacityScheduler(
        patch_size=16,
        image_size=224,
        max_epochs=100,
        min_capacity=0.45,
        max_capacity=1.0,
        num_experts=4,
        delta=2,
        beta=10,
        annealing_type="linear",
    )
    # Let's look at effective capacity and patch size over epochs by graphing
    import matplotlib.pyplot as plt

    effective_capacity_list = []
    patch_size_list = []
    for _ in range(100):
        capacity_distribution, patch_size = capacity_scheduler.update()
        effective_capacity_list.append(capacity_scheduler.effective_capacity)
        patch_size_list.append(patch_size)
    # Plot the results with different graphs
    # Label the axes
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # First subplot remains the same
    ax1.set_ylabel("Effective Capacity")
    ax1.set_xlabel("Epoch")
    ax1.plot(effective_capacity_list, label="Effective Capacity")
    ax1.legend()

    # Second subplot with dual y-axes
    ax2.set_ylabel("Adjusted Patch Size")
    ax2.set_xlabel("Epoch")

    # Primary y-axis for patch size
    line1 = ax2.plot(patch_size_list, label="Adjusted Patch Size", color="blue")

    # Only label when patch size changes
    last_patch_size = None
    for i, patch_size in enumerate(patch_size_list):
        if patch_size != last_patch_size:
            ax2.annotate(
                f"p:{patch_size}",
                (i, patch_size),
                textcoords="offset points",
                xytext=(-20, 5),
                ha="center",
                fontsize=8,
            )
            last_patch_size = patch_size

    # Secondary y-axis for grid size
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Grid Size (N×N)", color="red")
    grid_sizes = [224 // patch_size for patch_size in patch_size_list]
    line2 = ax2_twin.plot(grid_sizes, color="red", label="Grid Size")

    # Only label when grid size changes
    last_grid_size = None
    for i, grid_size in enumerate(grid_sizes):
        if grid_size != last_grid_size:
            ax2_twin.annotate(
                f"g:{grid_size}×{grid_size}",
                (i, grid_size),
                textcoords="offset points",
                xytext=(20, -15),
                ha="center",
                fontsize=8,
                color="red",
            )
            last_grid_size = grid_size

    # Combine legends from both y-axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.show()
