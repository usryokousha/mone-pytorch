import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize

from typing import Tuple, Optional

"""Implementation of the routing algorithm for the MONE model."""


def capacity(
    num_experts: int,
    num_tokens: int,
    capacity_factor: float = 2.0,
) -> int:
    return int(num_tokens * capacity_factor / num_experts)


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.bias = bias
        self.num_experts = num_experts
        self.router_weights = nn.Linear(dim, num_experts, bias=bias)
        self._init_weights()

    def _init_weights(self):
        # uniform distribution for the router weights
        nn.init.uniform_(self.router_weights.weight, a=-2e-2, b=2e-2)
        if self.router_weights.bias is not None:
            nn.init.zeros_(self.router_weights.bias)

    @torch.amp.autocast(enabled=False, device_type="cuda")
    def compute_router_probs(
        self, input_tokens: torch.Tensor, jitter_noise: float = 0.0
    ) -> torch.Tensor:
        """
        Compute the router probabilities for the input tokens.
        Keeps computation in float32 for numerical stability.
        """
        logits = self.router_weights(input_tokens)
        if jitter_noise > 0.0:
            noise = (
                torch.randn_like(logits, dtype=logits.dtype, device=logits.device)
                * jitter_noise
            )
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        return probs

    def compute_routing_instructions(
        self, router_probs: torch.Tensor, capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to compute routing instructions.
        Must be implemented by child classes.
        """
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute router predictions
        r_probs = self.compute_router_probs(x, jitter_noise)
        return self.compute_routing_instructions(r_probs, c)


class ExpertPreferredRouter(Router):
    def compute_routing_instructions(
        self, router_probs: torch.Tensor, capacity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expert assignments using the EPR algorithm.

        Args:
            router_probs: Tensor of shape (B, N, E) containing router probabilities
            capacity: Tensor of shape (E,) containing capacity fractions for each expert

        Returns:
            Tuple[Tensor, Tensor]: Expert assignments and corresponding probabilities
        """
        B, N, E = router_probs.shape
        T = N

        r_modified = router_probs.permute(0, 2, 1).clone()  # (B, E, N)

        # Initialize assignments to the smallest model (expert = 0)
        M = torch.zeros((B, N), dtype=torch.long, device=router_probs.device)

        # Iterate from largest model (E-1) down to smallest (0)
        for j in reversed(range(E)):
            kj = math.floor(capacity[j] * T)
            if kj > 0:
                # Get top-k indices for expert j
                _, indices = torch.topk(r_modified[:, j, :], k=kj, dim=-1)  # (B, kj)

                # Scatter assignments into M
                assign_vals = torch.full_like(indices, j, dtype=torch.long)
                M.scatter_(1, indices, assign_vals)

                # Scatter -inf into r_modified for these chosen tokens across all experts
                expanded_indices = indices.unsqueeze(1).expand(B, E, kj)  # (B, E, kj)
                chosen_tokens = torch.full(
                    (B, E, kj), float("-inf"), device=router_probs.device
                )
                r_modified.scatter_(2, expanded_indices, chosen_tokens)

        # Gather probabilities of the assigned experts
        M_probs = router_probs.gather(2, M.unsqueeze(-1)).squeeze(-1)  # (B, N)

        return M, M_probs


class ExpertsChooseMaskedRouter(Router):
    """
    PyTorch implementation of 'experts choose' routing strategy,
    subclassing the given Router class. This strategy assigns tokens
    by letting each expert pick its top tokens independently.

    The `_compute_routing_instructions` method returns:
    - dispatch_mask: [batch, num_tokens, num_experts, capacity]
    - combine_array: [batch, num_tokens, num_experts, capacity]

    Here, 'capacity' is the number of tokens that each expert selects.
    """

    def compute_routing_instructions(
        self, router_probs: torch.Tensor, capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing instructions using "experts choose" routing.

        Args:
            router_probs (torch.Tensor): [B, T, E] router probabilities
                                         B: batch
                                         T: num_tokens
                                         E: num_experts
            capacity (torch.Tensor): scalar tensor indicating the top-k capacity.

        Returns:
            dispatch_mask (torch.Tensor): [B, T, E, C] binary mask indicating which
                                          tokens are chosen by which experts.
            combine_array (torch.Tensor): [B, T, E, C] probabilities scaled by dispatch_mask.
        """
        # Ensure capacity is an integer
        if isinstance(capacity, torch.Tensor):
            capacity = int(capacity.item())

        B, T, E = router_probs.shape

        # transpose router_probs to [B, E, T]
        router_probs = router_probs.permute(0, 2, 1)

        if capacity == T:
            # If full capacity we don't need to dispatch experts [B, T, E, C]
            return None, router_probs.unsqueeze(-1)

        expert_gate, expert_indices = torch.topk(router_probs, k=capacity, dim=-1)

        # Create a one-hot dispatch mask of shape [B, E, C, T]
        dispatch_mask = F.one_hot(expert_indices, num_classes=T).float()

        # Permute to [B, T, E, C] to match desired output shape
        dispatch_mask = dispatch_mask.permute(0, 3, 1, 2).contiguous()

        # The combine array is the dispatch mask scaled by the selected probabilities
        combine_array = torch.einsum("...ec,...tec->...tec", expert_gate, dispatch_mask)

        return dispatch_mask, combine_array


def normalize(x: torch.Tensor, axis: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize tensor along dimension to unit norm."""
    m = torch.rsqrt(torch.square(x).sum(axis=axis, keepdim=True) + eps)
    return x * m


class SoftMergingRouter(nn.Module):
    """Soft router merging tokens as inputs/outputs of the experts.

    Implementation of the routing algorithm from:
    "From Sparse to Soft Mixture of Experts" (https://arxiv.org/abs/2308.00951)
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_slots: Optional[int] = None,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.capacity_factor = capacity_factor
        self.mu = nn.Parameter(
            torch.zeros(dim, self.num_experts, self.num_slots, dtype=self.dtype)
        )
        self.scale = nn.Parameter(torch.ones(1, dtype=self.dtype))
        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(1.0 / self.dim)
        nn.init.normal_(self.mu, std=std)

    def forward(
        self, inputs: torch.Tensor, jitter_noise: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tensor of shape (B, T, D) where:
                B = batch size
                T = number of tokens
                D = dimension of input features
            jitter_noise: If > 0, add noise to the logits

        Returns:
            Tuple of:
                dispatch_weights: Tensor of shape (B, T, E, C)
                combine_weights: Tensor of shape (B, T, E, C)
        """
        with torch.amp.autocast(
            device_type="cuda", enabled=False, device=inputs.device
        ):
            # Normalize inputs to have unit norm
            inputs = normalize(inputs, axis=-1)

            # Normalize mu to have unit norm
            mu = normalize(self.mu, axis=0)

            # Determine number of slots if not specified
            if self.num_slots is None:
                _, num_tokens, _ = inputs.shape
                self.num_slots = round(
                    num_tokens * self.capacity_factor / self.num_experts
                )

            # Scale inputs/mu before computing logits
            if inputs.numel() < mu.numel():
                inputs = inputs * self.scale
            else:
                mu = mu * self.scale

            # Compute router logits
            logits = torch.einsum("gmd,dnp->gmnp", inputs, mu)

            # Add noise during training if specified
            if jitter_noise > 0:
                noise = torch.randn_like(logits, dtype=logits.dtype) * jitter_noise
                logits = logits + noise

            # Compute dispatch and combine weights
            dispatch_weights = F.softmax(logits, dim=1)
            combine_weights = F.softmax(logits, dim=(2, 3))

        return dispatch_weights, combine_weights


def compute_capacity_distribution(
    E, e_c, delta=2.0, beta=10.0, device: Optional[torch.device] = None
):
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

    return torch.tensor(result.x, device=device)


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


class CapacityScheduler(nn.Module):
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
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.max_epochs = max_epochs
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.num_experts = num_experts
        self.delta = delta
        self.beta = beta
        self.annealing_type = annealing_type

        # Register buffers for stateful variables
        self.register_buffer("start_epoch", torch.tensor(0))
        self.register_buffer("epoch", torch.tensor(0))
        self.register_buffer("effective_capacity", torch.tensor(max_capacity))
        self.register_buffer(
            "capacity_distribution", torch.zeros(num_experts, dtype=torch.float32)
        )

    def update(self) -> Tuple[torch.Tensor, int]:
        # Update the epoch
        self.epoch += 1

        # Compute the effective capacity for the current step
        if self.annealing_type == "exponential":
            self.effective_capacity = torch.tensor(
                _exponential_annealing(
                    int(self.epoch.item()),
                    int(self.start_epoch.item()),
                    self.max_epochs,
                    self.min_capacity,
                    self.max_capacity,
                )
            )
        elif self.annealing_type == "cosine":
            self.effective_capacity = torch.tensor(
                _cosine_annealing(
                    int(self.epoch.item()),
                    int(self.start_epoch.item()),
                    self.max_epochs,
                    self.min_capacity,
                    self.max_capacity,
                )
            )
        elif self.annealing_type == "linear":
            self.effective_capacity = torch.tensor(
                _linear_annealing(
                    int(self.epoch.item()),
                    int(self.start_epoch.item()),
                    self.max_epochs,
                    self.min_capacity,
                    self.max_capacity,
                )
            )

        # Compute the capacity distribution based on new effective capacity
        self.capacity_distribution = torch.tensor(
            compute_capacity_distribution(
                self.num_experts,
                float(self.effective_capacity.item()),
                self.delta,
                self.beta,
            )
        )

        return self.capacity_distribution, self.adjusted_patch_size

    def reset(self):
        """Reset the scheduler state"""
        self.epoch.zero_()
        self.effective_capacity.fill_(self.max_capacity)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # Add any additional state that's not captured by buffers
        state_dict.update(
            {
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "max_epochs": self.max_epochs,
                "min_capacity": self.min_capacity,
                "max_capacity": self.max_capacity,
                "num_experts": self.num_experts,
                "delta": self.delta,
                "beta": self.beta,
                "annealing_type": self.annealing_type,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        # Extract configuration parameters
        self.patch_size = state_dict.pop("patch_size")
        self.image_size = state_dict.pop("image_size")
        self.max_epochs = state_dict.pop("max_epochs")
        self.min_capacity = state_dict.pop("min_capacity")
        self.max_capacity = state_dict.pop("max_capacity")
        self.num_experts = state_dict.pop("num_experts")
        self.delta = state_dict.pop("delta")
        self.beta = state_dict.pop("beta")
        self.annealing_type = state_dict.pop("annealing_type")

        # Load the remaining state (buffers)
        super().load_state_dict(state_dict, strict=strict)

    @property
    def adjusted_patch_size(self) -> torch.Tensor:
        # Calculate patch size based on effective capacity and ensure it's the next even number
        patch_size = torch.maximum(
            torch.tensor(1), (self.patch_size * self.effective_capacity).int()
        )
        return patch_size + (patch_size % 2)


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
